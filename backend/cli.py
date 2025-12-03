"""Terminal-first CLI for interacting with the LLM Council."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import subprocess
import uuid
from typing import Optional

from . import council, storage, fs_tools
from .council_config import load_council_config


EXTERNAL_CLI_ENV_VARS = {
    "codex": "CODEX_CLI_CMD",
    "claude": "CLAUDE_CLI_CMD",
    "gemini": "GEMINI_CLI_CMD",
}

EXTERNAL_CLI_DEFAULTS = {
    "codex": "codex",
    "claude": "claude",
    "gemini": "gemini",
}


_FILE_PATTERN = re.compile(r"\[\[file:([^\]]+)\]\]")
_TREE_PATTERN = re.compile(r"\[\[tree:([^\]]+)\]\]")
_WRITE_PATTERN = re.compile(r"\[\[write:([^|]+)\|([^\]]*)\]\]")


def parse_special_command(line: str) -> Optional[str]:
    """Parse a user line that may represent a special command.

    Returns one of {"codex", "claude", "gemini", "exit"} or None.
    """
    stripped = line.strip().lower()
    if stripped.startswith(":"):
        stripped = stripped[1:].strip()

    command = stripped
    if command in ("codex", "claude", "gemini", "exit"):
        return command
    return None


def get_external_cli_command(kind: str) -> str:
    """Resolve the command to invoke for an external dev CLI."""
    env_var = EXTERNAL_CLI_ENV_VARS.get(kind)
    default_cmd = EXTERNAL_CLI_DEFAULTS.get(kind, kind)
    if env_var is None:
        return default_cmd
    return os.getenv(env_var, default_cmd)


def run_external_cli(kind: str) -> int:
    """Invoke an external dev CLI (codex / claude / gemini)."""
    cmd = get_external_cli_command(kind)
    if not cmd:
        print(f"No command configured for external CLI '{kind}'.")
        return 1
    try:
        return subprocess.call(cmd, shell=True)
    except FileNotFoundError:
        print(f"Failed to run external CLI '{kind}': command '{cmd}' not found.")
        return 1


def expand_file_placeholders(text: str) -> str:
    """Expand [[file:/path]] placeholders with file previews.

    This helper is pure with respect to the council pipeline: it only
    enriches the text that will be sent to models, without affecting
    stored conversation history.
    """

    def _replace(match: re.Match) -> str:
        raw_path = match.group(1).strip()
        try:
            preview = fs_tools.read_file_preview(raw_path)
            label = f"[FILE {raw_path}]"
            return f"\n\n{label}\n{preview}\n\n"
        except Exception as e:
            return f"[Error reading file {raw_path}: {e}]"

    return _FILE_PATTERN.sub(_replace, text)


def expand_tree_placeholders(text: str) -> str:
    """Expand [[tree:/path]] placeholders with directory trees."""

    def _replace(match: re.Match) -> str:
        raw_path = match.group(1).strip()
        try:
            tree = fs_tools.list_directory_tree(raw_path)
            label = f"[DIR {raw_path}]"
            return f"\n\n{label}\n{tree}\n\n"
        except Exception as e:
            return f"[Error listing directory {raw_path}: {e}]"

    return _TREE_PATTERN.sub(_replace, text)


def apply_write_placeholders(text: str) -> str:
    """Process [[write:/path|content]] placeholders by writing files.

    Placeholders are replaced with a concise status marker so that the
    models see the outcome (success or failure) rather than the raw
    control syntax.
    """

    def _replace(match: re.Match) -> str:
        raw_path = match.group(1).strip()
        body = match.group(2)
        try:
            fs_tools.write_text_file(raw_path, body)
            return f"[Wrote file {raw_path}]"
        except Exception as e:
            return f"[Error writing file {raw_path}: {e}]"

    return _WRITE_PATTERN.sub(_replace, text)


async def _handle_user_message(conversation_id: str, content: str) -> None:
    """Run the full council pipeline for a single user message."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        conversation = storage.create_conversation(conversation_id)

    is_first_message = len(conversation["messages"]) == 0

    storage.add_user_message(conversation_id, content)

    if is_first_message:
        title = await council.generate_conversation_title(content)
        storage.update_conversation_title(conversation_id, title)

    # First apply any write placeholders so that file-system side effects
    # happen before the council is invoked. Then enrich the question with
    # [[tree:/path]] and [[file:/path]] placeholders. The stored
    # conversation content remains the original user input.
    processed = apply_write_placeholders(content)
    processed = expand_tree_placeholders(processed)
    council_input = expand_file_placeholders(processed)

    stage1_results, stage2_results, stage3_result, metadata = await council.run_full_council(
        council_input
    )

    storage.add_assistant_message(
        conversation_id,
        stage1_results,
        stage2_results,
        stage3_result,
    )

    # Render a concise but informative summary to the terminal.
    print("\n--- Stage 1: Individual Responses ---")
    for idx, result in enumerate(stage1_results, start=1):
        model = result.get("model", "unknown")
        response = result.get("response", "")
        print(f"[{idx}] {model}")
        print(response)
        print()

    print("--- Stage 2: Aggregate Rankings ---")
    aggregate = metadata.get("aggregate_rankings", []) if isinstance(metadata, dict) else []
    if aggregate:
        for entry in aggregate:
            model = entry.get("model")
            avg_rank = entry.get("average_rank")
            count = entry.get("rankings_count")
            print(f"{model}: average rank {avg_rank} over {count} rankings")
    else:
        print("(no rankings available)")

    print("\n--- Stage 3: Chairman Synthesis ---")
    chairman_model = stage3_result.get("model", "chairman")
    chairman_response = stage3_result.get("response", "")
    print(f"Chairman ({chairman_model}):\n")
    print(chairman_response)
    print()


async def _run_session(conversation_id: Optional[str]) -> None:
    if conversation_id is None:
        conversation_id = str(uuid.uuid4())
        storage.create_conversation(conversation_id)
        print(f"Started new conversation: {conversation_id}")
    else:
        existing = storage.get_conversation(conversation_id)
        if existing is None:
            storage.create_conversation(conversation_id)
            print(f"Created conversation: {conversation_id}")
        else:
            print(f"Resuming conversation: {conversation_id}")
            print(f"Title: {existing.get('title', 'New Conversation')}")

    # Show effective council configuration summary for this session.
    try:
        cfg = load_council_config()
        if cfg.council_models:
            models_list = ", ".join(cfg.council_models)
            print(f"Council models: {models_list}")
            if cfg.chairman_model:
                print(f"Chairman model: {cfg.chairman_model}")
            if cfg.title_model and cfg.title_model != cfg.chairman_model:
                print(f"Title model: {cfg.title_model}")
        else:
            print("Warning: no valid council models configured.")

        if cfg.failures:
            print("Configuration issues:")
            for failure in cfg.failures:
                model_spec = failure.get("model_spec")
                error_type = failure.get("error_type")
                missing = ", ".join(failure.get("missing", []))
                provider = failure.get("provider")
                details = f"provider={provider}" if provider else ""
                if missing:
                    details = (details + f" missing={missing}").strip()
                print(f"  - {model_spec}: {error_type} ({details})")
    except Exception as e:
        # Config summary is best-effort; do not block the session.
        print(f"Warning: failed to load council configuration: {e}")

    print("Type your message and press Enter.")
    print("Use :codex / :claude / :gemini to launch external CLIs, :exit to quit.")

    loop = asyncio.get_event_loop()
    while True:
        try:
            # Use run_in_executor so that Ctrl+C behaviour stays responsive.
            line = await loop.run_in_executor(None, lambda: input("You: "))
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not line.strip():
            continue

        command = parse_special_command(line)
        if command == "exit":
            print("Goodbye.")
            break
        if command in ("codex", "claude", "gemini"):
            run_external_cli(command)
            continue

        await _handle_user_message(conversation_id, line)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Council - Terminal CLI")
    parser.add_argument(
        "--conversation-id",
        help="Existing conversation ID to resume (default: create new)",
    )
    args = parser.parse_args()

    asyncio.run(_run_session(args.conversation_id))


if __name__ == "__main__":
    main()
