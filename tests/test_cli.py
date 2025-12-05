import pytest
from backend import cli
import importlib
from pathlib import Path


def test_parse_special_command_recognizes_commands():
    # Leading colon is supported
    assert cli.parse_special_command(":codex") == "codex"
    assert cli.parse_special_command(" :claude ") == "claude"
    assert cli.parse_special_command(":gemini") == "gemini"
    assert cli.parse_special_command(":exit") == "exit"
    # Bare commands (without colon) are also recognized
    assert cli.parse_special_command("codex") == "codex"
    assert cli.parse_special_command(" claude ") == "claude"
    assert cli.parse_special_command("gemini") == "gemini"
    assert cli.parse_special_command("exit") == "exit"
    assert cli.parse_special_command("hello") is None


def test_get_external_cli_command_uses_env(monkeypatch):
    monkeypatch.setenv("CODEX_CLI_CMD", "my-codex-cli")
    cmd = cli.get_external_cli_command("codex")
    assert cmd == "my-codex-cli"


def test_run_external_cli_invokes_subprocess(monkeypatch):
    calls = []

    def fake_call(cmd, shell):
        calls.append((cmd, shell))
        return 0

    monkeypatch.setenv("CODEX_CLI_CMD", "codex-cli")
    monkeypatch.setattr(cli.subprocess, "call", fake_call)

    rc = cli.run_external_cli("codex")
    assert rc == 0
    assert calls
    assert calls[0][0] == "codex-cli"
    assert calls[0][1] is True


def test_expand_file_placeholders_injects_file_content(monkeypatch, tmp_path):
    # Allow tmp_path as an accessible root and create a file inside it.
    allowed_root = tmp_path / "root"
    allowed_root.mkdir()
    file_path = allowed_root / "note.txt"
    file_path.write_text("hello world", encoding="utf-8")

    monkeypatch.setenv("LLM_COUNCIL_FS_ALLOWED_ROOTS", str(allowed_root))

    # Reload modules so that fs_tools picks up the new env config.
    import backend.fs_tools as fs_tools_module

    importlib.reload(fs_tools_module)
    importlib.reload(cli)

    text = f"see this [[file:{file_path}]] please"
    expanded = cli.expand_file_placeholders(text)

    assert "hello world" in expanded
    assert "[FILE" in expanded


def test_expand_file_placeholders_handles_disallowed_path(monkeypatch, tmp_path):
    # Only allow a specific subdirectory, but reference a file outside it.
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    other_root = tmp_path / "other"
    other_root.mkdir()
    file_path = other_root / "note.txt"
    file_path.write_text("top secret", encoding="utf-8")

    monkeypatch.setenv("LLM_COUNCIL_FS_ALLOWED_ROOTS", str(allowed_root))

    import backend.fs_tools as fs_tools_module

    importlib.reload(fs_tools_module)
    importlib.reload(cli)

    text = f"see this [[file:{file_path}]] please"
    expanded = cli.expand_file_placeholders(text)

    # The placeholder should be replaced with an error message, not raise.
    assert "[Error reading file" in expanded
    assert "top secret" not in expanded


def test_expand_tree_placeholders_injects_directory_tree(monkeypatch, tmp_path):
    # Allow tmp_path as an accessible root and create a directory tree.
    allowed_root = tmp_path / "root"
    sub = allowed_root / "sub"
    sub.mkdir(parents=True)
    (allowed_root / "a.txt").write_text("a", encoding="utf-8")
    (sub / "b.txt").write_text("b", encoding="utf-8")

    monkeypatch.setenv("LLM_COUNCIL_FS_ALLOWED_ROOTS", str(allowed_root))

    import backend.fs_tools as fs_tools_module

    importlib.reload(fs_tools_module)
    importlib.reload(cli)

    text = f"tree here [[tree:{allowed_root}]] end"
    expanded = cli.expand_tree_placeholders(text)

    assert "a.txt" in expanded
    assert "sub/" in expanded
    assert "[DIR" in expanded


def test_apply_write_placeholders_writes_file_and_replaces(monkeypatch, tmp_path):
    # Allow tmp_path as an accessible root and choose a file path inside it.
    allowed_root = tmp_path / "root"
    allowed_root.mkdir()
    file_path = allowed_root / "note.txt"

    monkeypatch.setenv("LLM_COUNCIL_FS_ALLOWED_ROOTS", str(allowed_root))

    import backend.fs_tools as fs_tools_module

    importlib.reload(fs_tools_module)
    importlib.reload(cli)

    text = f"please [[write:{file_path}|hello world]] now"
    processed = cli.apply_write_placeholders(text)

    # The file should be written and the placeholder replaced with a status.
    assert file_path.exists()
    assert file_path.read_text(encoding="utf-8") == "hello world"
    assert "[Wrote file" in processed


def test_apply_write_placeholders_handles_disallowed_path(monkeypatch, tmp_path):
    # Only allow a specific subdirectory, but reference a file outside it.
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    other_root = tmp_path / "other"
    other_root.mkdir()
    file_path = other_root / "note.txt"

    monkeypatch.setenv("LLM_COUNCIL_FS_ALLOWED_ROOTS", str(allowed_root))

    import backend.fs_tools as fs_tools_module

    importlib.reload(fs_tools_module)
    importlib.reload(cli)

    text = f"please [[write:{file_path}|top secret]] now"
    processed = cli.apply_write_placeholders(text)

    # The write should fail gracefully and not create the file.
    assert not file_path.exists()
    assert "[Error writing file" in processed


def test_expand_codebase_placeholders_injects_codebase_preview(monkeypatch, tmp_path):
    # Allow tmp_path as an accessible root and create a small codebase.
    allowed_root = tmp_path / "root"
    src = allowed_root / "src"
    src.mkdir(parents=True)
    (src / "main.py").write_text("print('hello')", encoding="utf-8")

    monkeypatch.setenv("LLM_COUNCIL_FS_ALLOWED_ROOTS", str(allowed_root))

    import backend.fs_tools as fs_tools_module

    importlib.reload(fs_tools_module)
    importlib.reload(cli)

    text = f"scan [[codebase:{allowed_root}|pattern=*.py|max_files=5|max_depth=3]] now"
    expanded = cli.expand_codebase_placeholders(text)

    assert "[CODEBASE" in expanded
    assert "main.py" in expanded
    assert "print('hello')" in expanded


def test_expand_codebase_placeholders_handles_disallowed_path(monkeypatch, tmp_path):
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    other_root = tmp_path / "other"
    other_root.mkdir()

    monkeypatch.setenv("LLM_COUNCIL_FS_ALLOWED_ROOTS", str(allowed_root))

    import backend.fs_tools as fs_tools_module

    importlib.reload(fs_tools_module)
    importlib.reload(cli)

    text = f"scan [[codebase:{other_root}|pattern=*.py]] now"
    expanded = cli.expand_codebase_placeholders(text)

    assert "[Error collecting codebase" in expanded


def test_parse_path_command_recognizes_absolute_directory(monkeypatch, tmp_path):
    import backend.cli as cli

    root = tmp_path / "repo"
    root.mkdir()

    # Allow the temporary repo path.
    monkeypatch.setattr(cli.fs_tools, "is_path_allowed", lambda p: True)

    parsed = cli.parse_path_command(str(root))
    assert parsed == root


def test_parse_path_command_ignores_non_path_lines(monkeypatch):
    import backend.cli as cli

    # Non-path input should not be treated as a path command.
    assert cli.parse_path_command("hello world") is None
    assert cli.parse_path_command(":codex") is None


def test_scan_project_path_uses_build_project_context(monkeypatch, tmp_path):
    import backend.cli as cli

    root = tmp_path / "repo"
    root.mkdir()

    calls = {"allowed": None, "context": None}

    def fake_is_path_allowed(path):
        calls["allowed"] = path
        return True

    monkeypatch.setattr(cli.fs_tools, "is_path_allowed", fake_is_path_allowed)

    class FakeProjectContext:
        @staticmethod
        def build_project_context(root_path, max_tree_depth=None, max_tree_entries=None, code_kwargs=None):
            calls["context"] = (root_path, max_tree_depth, max_tree_entries, code_kwargs)
            return "CTX"

    monkeypatch.setattr(cli, "project_context", FakeProjectContext)

    result = cli.scan_project_path(str(root))

    # Should return the context text produced by build_project_context.
    assert result == "CTX"
    # is_path_allowed should be called with the resolved Path.
    assert calls["allowed"] == root
    # build_project_context should receive the same root and default limits.
    root_path, max_tree_depth, max_tree_entries, code_kwargs = calls["context"]
    assert root_path == root
    assert max_tree_depth is None
    assert max_tree_entries is None
    assert code_kwargs is None


def test_enrich_content_with_path_placeholders_injects_placeholders_for_first_message(
    monkeypatch,
    tmp_path,
):
    import backend.cli as cli

    root = tmp_path / "repo"
    root.mkdir()

    # Allow the temporary repo path.
    monkeypatch.setattr(cli.fs_tools, "is_path_allowed", lambda p: True)

    content = f"iterate through{root}repo,summarize the project no more than 10 sentences"

    enriched = cli.enrich_content_with_path_placeholders(
        content,
        is_first_message=True,
    )

    # Should prepend tree and codebase placeholders for the detected path.
    assert f"[[tree:{root}]]" in enriched
    assert f"[[codebase:{root}]]" in enriched
    # Original content should still be present.
    assert content in enriched


def test_enrich_content_with_path_placeholders_skips_when_no_path(monkeypatch):
    import backend.cli as cli

    content = "hello world"
    enriched = cli.enrich_content_with_path_placeholders(
        content,
        is_first_message=True,
    )
    assert enriched == content


def test_enrich_content_with_path_placeholders_respects_existing_placeholders(monkeypatch):
    import backend.cli as cli

    # When tree/codebase placeholders are already present, enrichment
    # should not add additional ones.
    content = "[[tree:/project]]\n[[codebase:/project]]\nplease audit"

    enriched = cli.enrich_content_with_path_placeholders(
        content,
        is_first_message=True,
    )

    assert enriched == content


def test_handle_path_line_uses_parse_and_scan(monkeypatch, tmp_path):
    import backend.cli as cli

    root = tmp_path / "repo"
    root.mkdir()

    calls = {"parse": None, "scan": None}

    def fake_parse(line: str):
        calls["parse"] = line
        return root

    def fake_scan(path_str: str, max_tree_depth=None, max_tree_entries=None, code_kwargs=None):
        calls["scan"] = (path_str, max_tree_depth, max_tree_entries, code_kwargs)
        return "CTX"

    monkeypatch.setattr(cli, "parse_path_command", fake_parse)
    monkeypatch.setattr(cli, "scan_project_path", fake_scan)

    result = cli.handle_path_line(str(root))

    assert result == "CTX"
    assert calls["parse"] == str(root)
    assert calls["scan"][0] == str(root)


def test_handle_path_line_returns_none_when_not_path(monkeypatch):
    import backend.cli as cli

    monkeypatch.setattr(cli, "parse_path_command", lambda line: None)

    result = cli.handle_path_line("hello world")
    assert result is None


def test_main_scan_subcommand_uses_scan_project_path(monkeypatch, tmp_path, capsys):
    import sys
    import backend.cli as cli

    root = tmp_path / "repo"
    root.mkdir()

    calls = {"args": None}

    def fake_scan(path_str: str, max_tree_depth=None, max_tree_entries=None, code_kwargs=None):
        calls["args"] = (path_str, max_tree_depth, max_tree_entries, code_kwargs)
        return "CTX"

    monkeypatch.setattr(cli, "scan_project_path", fake_scan)
    monkeypatch.setattr(sys, "argv", ["llm-council", "scan", str(root)])

    cli.main()

    out = capsys.readouterr().out
    # The scan subcommand should print the context text.
    assert "CTX" in out
    # scan_project_path should be called with the provided path.
    assert calls["args"][0] == str(root)


@pytest.mark.asyncio
async def test_run_session_one_shot_calls_handler_once_and_exits(monkeypatch):
    import backend.cli as cli

    # Avoid real config and storage side effects.
    class DummyCfg:
        council_models = ["model-a"]
        chairman_model = "model-a"
        title_model = None
        failures = []

    monkeypatch.setattr(cli, "load_council_config", lambda: DummyCfg())
    monkeypatch.setattr(cli.storage, "create_conversation", lambda cid: {"id": cid, "messages": []})
    monkeypatch.setattr(cli.storage, "get_conversation", lambda cid: {"id": cid, "messages": []})

    # Provide a single line of input; if the loop does not exit after
    # one iteration, a second call would fail.
    inputs = ["audit once"]

    def fake_input(prompt=""):
        return inputs.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)

    calls = {"messages": []}

    async def fake_handle(conversation_id, content, summary_only=False):
        calls["messages"].append((conversation_id, content, summary_only))

    monkeypatch.setattr(cli, "_handle_user_message", fake_handle)

    # Act: run a one-shot session; it should consume exactly one line.
    await cli._run_session(None, one_shot=True)

    # Assert: handler was called once with summary_only=True.
    assert len(calls["messages"]) == 1
    _, content, summary_only = calls["messages"][0]
    assert content == "audit once"
    assert summary_only is True


@pytest.mark.asyncio
async def test_handle_user_message_summary_only_wraps_audit_prompt(monkeypatch, capsys):
    import backend.cli as cli

    # Minimal in-memory conversation storage.
    conv = {"id": "conv-1", "messages": []}

    monkeypatch.setattr(cli.storage, "get_conversation", lambda cid: None)
    monkeypatch.setattr(cli.storage, "create_conversation", lambda cid: conv)
    monkeypatch.setattr(cli.storage, "add_user_message", lambda cid, content: conv["messages"].append({"role": "user", "content": content}))
    monkeypatch.setattr(cli.storage, "update_conversation_title", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli.storage, "add_assistant_message", lambda *args, **kwargs: None)

    async def fake_title(content: str) -> str:
        return "Title"

    monkeypatch.setattr(cli.council, "generate_conversation_title", fake_title)

    # Prepare_council_input should see the wrapped audit prompt.
    seen_inputs = []

    def fake_prepare_council_input(content: str, is_first_message: bool) -> str:
        seen_inputs.append((content, is_first_message))
        # Return the content unchanged so run_full_council receives it.
        return content

    monkeypatch.setattr(cli, "prepare_council_input", fake_prepare_council_input)

    async def fake_run_full_council(user_query: str):
        # Return minimal, valid council outputs.
        return [], [], {"model": "chairman", "response": "FINAL AUDIT"}, {"aggregate_rankings": []}

    monkeypatch.setattr(cli.council, "run_full_council", fake_run_full_council)

    # Act: first message in summary-only (audit) mode.
    await cli._handle_user_message("conv-1", "please audit", summary_only=True)

    # Assert: prepare_council_input saw an audit-wrapped prompt that
    # includes both the audit preamble and the original content.
    assert seen_inputs
    wrapped_content, first_flag = seen_inputs[0]
    assert first_flag is True
    assert "auditing the *current local llm-council codebase*" in wrapped_content
    assert "please audit" in wrapped_content

    # Only the chairman summary should be printed.
    out = capsys.readouterr().out
    assert "--- Stage 1:" not in out
    assert "--- Stage 2:" not in out
    assert "Chairman (chairman):" in out
    assert "FINAL AUDIT" in out


def test_prepare_council_input_includes_project_context_for_first_message(monkeypatch):
    import backend.cli as cli

    calls = {"ctx": 0}

    def fake_apply_write_placeholders(text: str) -> str:
        return "PROCESSED"

    monkeypatch.setattr(cli, "apply_write_placeholders", fake_apply_write_placeholders)
    monkeypatch.setattr(cli, "expand_tree_placeholders", lambda text: text)
    monkeypatch.setattr(cli, "expand_codebase_placeholders", lambda text: text)
    monkeypatch.setattr(cli, "expand_file_placeholders", lambda text: text)

    class FakeProjectContext:
        @staticmethod
        def build_initial_project_context() -> str:
            calls["ctx"] += 1
            return "CTX"

    monkeypatch.setattr(cli, "project_context", FakeProjectContext)

    result = cli.prepare_council_input("USER", is_first_message=True)

    assert calls["ctx"] == 1
    assert "CTX" in result
    assert "PROCESSED" in result
    assert result.index("CTX") < result.index("PROCESSED")


def test_prepare_council_input_skips_project_context_for_followup_message(monkeypatch):
    import backend.cli as cli

    calls = {"ctx": 0}

    monkeypatch.setattr(cli, "apply_write_placeholders", lambda text: "PROCESSED")
    monkeypatch.setattr(cli, "expand_tree_placeholders", lambda text: text)
    monkeypatch.setattr(cli, "expand_codebase_placeholders", lambda text: text)
    monkeypatch.setattr(cli, "expand_file_placeholders", lambda text: text)

    class FakeProjectContext:
        @staticmethod
        def build_initial_project_context() -> str:
            calls["ctx"] += 1
            return "CTX"

    monkeypatch.setattr(cli, "project_context", FakeProjectContext)

    result = cli.prepare_council_input("USER", is_first_message=False)

    assert calls["ctx"] == 0
    assert result == "PROCESSED"
