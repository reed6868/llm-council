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
