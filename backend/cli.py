"""Terminal-first CLI for interacting with the LLM Council."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Optional

from . import council, storage, fs_tools, project_context
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
_CODEBASE_PATTERN = re.compile(r"\[\[codebase:([^|\]]+)(?:\|([^\]]+))?\]\]")
_PATH_IN_TEXT_PATTERN = re.compile(r"(?P<path>(?:~|/)[A-Za-z0-9_\-./]+)")


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


def parse_path_command(line: str) -> Optional[Path]:
    """Interpret a line as a filesystem path command if applicable.

    A valid path command is a line that:
    - is non-empty
    - starts with "/" or "~"
    - resolves to an existing directory
    - is allowed by fs_tools.is_path_allowed()
    """
    stripped = line.strip()
    if not stripped:
        return None
    if not (stripped.startswith("/") or stripped.startswith("~")):
        return None

    candidate = Path(stripped).expanduser()
    try:
        if not candidate.is_dir():
            return None
    except OSError:
        return None

    if not fs_tools.is_path_allowed(candidate):
        return None

    return candidate


def _find_first_allowed_path_in_text(text: str) -> Optional[str]:
    """Find the first allowed directory path mentioned in free-form text."""
    for match in _PATH_IN_TEXT_PATTERN.finditer(text):
        raw = match.group("path").strip()
        # Strip common trailing punctuation that may be attached.
        candidate = raw.rstrip("，。！？:：；,;、")
        if not candidate:
            continue
        path_obj = Path(candidate).expanduser()
        try:
            if not path_obj.is_dir():
                continue
        except OSError:
            continue
        if not fs_tools.is_path_allowed(path_obj):
            continue
        return candidate
    return None


def enrich_content_with_path_placeholders(
    content: str,
    is_first_message: bool,
) -> str:
    """Inject tree/codebase placeholders for a detected path in the content.

    This only applies to the first message in a conversation and only when the
    user has not already provided explicit [[tree:]] or [[codebase:]] markers.
    """
    if not is_first_message:
        return content

    # If the user already specified tree/codebase placeholders, respect them.
    if _TREE_PATTERN.search(content) or _CODEBASE_PATTERN.search(content):
        return content

    raw_path = _find_first_allowed_path_in_text(content)
    if not raw_path:
        return content

    prefix = f"[[tree:{raw_path}]]\n[[codebase:{raw_path}]]\n\n"
    return prefix + content


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


def handle_path_line(line: str) -> Optional[str]:
    """Handle a potential path line by returning a project context.

    When the line represents an allowed directory path, this returns the
    project context text produced by scan_project_path(). Otherwise it
    returns None and the caller should treat the line as a normal message.
    """
    path = parse_path_command(line)
    if path is None:
        return None
    return scan_project_path(str(path))


def scan_project_path(
    path_str: str,
    max_tree_depth: int | None = None,
    max_tree_entries: int | None = None,
    code_kwargs: dict | None = None,
) -> str:
    """Build a project context summary for a given filesystem path.

    This helper is a thin wrapper around project_context.build_project_context()
    that enforces fs_tools path permissions and expands user directories.
    """
    root = Path(path_str).expanduser()
    if not fs_tools.is_path_allowed(root):
        raise PermissionError(f"Access to {path_str} is not allowed")

    return project_context.build_project_context(
        root,
        max_tree_depth=max_tree_depth,
        max_tree_entries=max_tree_entries,
        code_kwargs=code_kwargs,
    )


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


def expand_codebase_placeholders(text: str) -> str:
    """Expand [[codebase:/path|k=v|...]] placeholders with code previews."""

    def _replace(match: re.Match) -> str:
        raw_path = match.group(1).strip()
        options_str = (match.group(2) or "").strip()

        patterns = None
        max_bytes = 8192
        max_files = 50
        max_depth = 8
        extra_ignore_dirs: list[str] = []

        try:
            if options_str:
                for segment in options_str.split("|"):
                    segment = segment.strip()
                    if not segment or "=" not in segment:
                        continue
                    key, value = segment.split("=", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key in ("pattern", "patterns"):
                        patterns = [p.strip() for p in value.split(",") if p.strip()]
                    elif key == "max_bytes":
                        max_bytes = int(value)
                    elif key == "max_files":
                        max_files = int(value)
                    elif key == "max_depth":
                        max_depth = int(value)
                    elif key == "ignore":
                        extra_ignore_dirs = [v.strip() for v in value.split(",") if v.strip()]

            code = fs_tools.collect_codebase_preview(
                raw_path,
                patterns=patterns,
                max_bytes_per_file=max_bytes,
                max_files=max_files,
                max_depth=max_depth,
                extra_ignore_dirs=extra_ignore_dirs,
            )
            label = f"[CODEBASE {raw_path}]"
            return f"\n\n{label}\n{code}\n\n"
        except Exception as e:
            return f"[Error collecting codebase {raw_path}: {e}]"

    return _CODEBASE_PATTERN.sub(_replace, text)


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


def prepare_council_input(content: str, is_first_message: bool) -> str:
    """Prepare the final text that will be sent to the council.

    This applies placeholder processing to the user content and, for the
    first message in a conversation, prefixes a project context summary
    describing the local codebase.
    """
    processed = apply_write_placeholders(content)
    processed = expand_tree_placeholders(processed)
    processed = expand_codebase_placeholders(processed)
    processed = expand_file_placeholders(processed)

    if not is_first_message:
        return processed

    try:
        ctx = project_context.build_initial_project_context()
    except Exception as e:
        ctx = f"[Error building project context: {e}]"

    return f"{ctx}\n\n{processed}"


async def _handle_user_message(
    conversation_id: str,
    content: str,
    summary_only: bool = False,
) -> None:
    """Run the full council pipeline for a single user message.

    When summary_only is True, only the chairman's synthesized answer
    is printed to the terminal (one-shot audit mode).
    """
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        conversation = storage.create_conversation(conversation_id)

    is_first_message = len(conversation["messages"]) == 0

    storage.add_user_message(conversation_id, content)

    if is_first_message:
        title = await council.generate_conversation_title(content)
        storage.update_conversation_title(conversation_id, title)

    # In summary-only mode for the first message, wrap the user content
    # with a short audit preamble that explicitly instructs models to
    # ground their analysis in this specific local codebase rather than
    # emitting generic checklists.
    effective_content = content
    if summary_only and is_first_message:
        effective_content = (
            "You are auditing the *current local llm-council codebase* whose "
            "project context (PROJECT ROOT/TREE/CODEBASE) is included above.\n"
            "Your answer MUST:\n"
            "- Ground every finding in specific files or modules that exist in this repository "
            "(for example backend/cli.py, backend/main.py, backend/council.py, "
            "backend/llm_client.py, backend/fs_tools.py, backend/storage.py, "
            "frontend/src/App.jsx, frontend/src/components/*, tests/*).\n"
            "- Avoid long, generic checklists or advice that is not clearly tied to this codebase.\n"
            "- Prefer concise, project-specific findings over broad best-practices.\n\n"
            "User request:\n"
            f"{content}"
        )

    # For the very first message, automatically inject tree/codebase
    # placeholders when the user mentions a concrete project path, so
    # that the council can see that codebase without requiring explicit
    # [[tree:]] / [[codebase:]] syntax.
    enriched_content = enrich_content_with_path_placeholders(
        effective_content,
        is_first_message=is_first_message,
    )

    # Prepare the full council input. For the first message in a
    # conversation, this includes a project context summary describing
    # the local codebase; subsequent messages only process placeholders.
    council_input = prepare_council_input(enriched_content, is_first_message)

    stage1_results, stage2_results, stage3_result, metadata = await council.run_full_council(
        council_input
    )

    storage.add_assistant_message(
        conversation_id,
        stage1_results,
        stage2_results,
        stage3_result,
    )

    if summary_only:
        chairman_model = stage3_result.get("model", "chairman")
        chairman_response = stage3_result.get("response", "")
        print("\nChairman ({model}):\n".format(model=chairman_model))
        print(chairman_response)
        print()
        return

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


async def _run_session(conversation_id: Optional[str], one_shot: bool = False) -> None:
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
    if one_shot:
        print("One-shot audit mode: first non-command message will run a single council pass and then exit.")

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

        # Handle a bare path line as a project scan command when applicable.
        path_context = handle_path_line(line)
        if path_context is not None:
            print(path_context)
            if one_shot:
                print("One-shot session complete. Exiting.")
                break
            continue

        await _handle_user_message(conversation_id, line, summary_only=one_shot)
        if one_shot:
            print("One-shot session complete. Exiting.")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Council - Terminal CLI")
    parser.add_argument(
        "--conversation-id",
        help="Existing conversation ID to resume (default: create new)",
    )
    parser.add_argument(
        "--one-shot",
        action="store_true",
        help="Run a single audit-style council pass (chairman summary only) and then exit",
    )

    subparsers = parser.add_subparsers(dest="command")
    scan_parser = subparsers.add_parser("scan", help="Scan a project path and print its context")
    scan_parser.add_argument(
        "path",
        help="Filesystem path to a project root to scan",
    )

    args = parser.parse_args()

    if args.command == "scan":
        ctx = scan_project_path(args.path)
        print(ctx)
        return

    asyncio.run(_run_session(args.conversation_id, one_shot=args.one_shot))


if __name__ == "__main__":
    main()
