import importlib
from pathlib import Path


def test_read_file_preview_within_allowed_root(monkeypatch, tmp_path):
    # Arrange: allow tmp_path as root and create a file inside it.
    root = tmp_path / "root"
    root.mkdir()
    file_path = root / "test.txt"
    file_path.write_text("hello world", encoding="utf-8")

    monkeypatch.setenv("LLM_COUNCIL_FS_ALLOWED_ROOTS", str(root))

    from backend import fs_tools as fs_tools_module

    importlib.reload(fs_tools_module)

    # Act
    content = fs_tools_module.read_file_preview(str(file_path))

    # Assert
    assert "hello world" in content


def test_read_file_preview_outside_allowed_root_raises(monkeypatch, tmp_path):
    # Arrange: only allow a specific subdirectory, but place the file elsewhere.
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    other_root = tmp_path / "other"
    other_root.mkdir()
    file_path = other_root / "secret.txt"
    file_path.write_text("top secret", encoding="utf-8")

    monkeypatch.setenv("LLM_COUNCIL_FS_ALLOWED_ROOTS", str(allowed_root))

    from backend import fs_tools as fs_tools_module

    importlib.reload(fs_tools_module)

    # Act / Assert
    import pytest

    with pytest.raises(PermissionError):
        fs_tools_module.read_file_preview(str(file_path))


def test_list_directory_tree_within_allowed_root(monkeypatch, tmp_path):
    # Arrange: allow tmp_path as root and create a small directory tree.
    root = tmp_path / "root"
    sub = root / "sub"
    sub.mkdir(parents=True)
    (root / "a.txt").write_text("a", encoding="utf-8")
    (sub / "b.txt").write_text("b", encoding="utf-8")

    monkeypatch.setenv("LLM_COUNCIL_FS_ALLOWED_ROOTS", str(root))

    from backend import fs_tools as fs_tools_module

    importlib.reload(fs_tools_module)

    tree = fs_tools_module.list_directory_tree(str(root))

    # The tree should mention both files and the subdirectory.
    assert "a.txt" in tree
    assert "sub/" in tree
    assert "b.txt" in tree


def test_write_text_file_within_allowed_root(monkeypatch, tmp_path):
    # Arrange: allow tmp_path as root and choose a file path inside it.
    root = tmp_path / "root"
    root.mkdir()
    target = root / "note.txt"

    monkeypatch.setenv("LLM_COUNCIL_FS_ALLOWED_ROOTS", str(root))

    from backend import fs_tools as fs_tools_module

    importlib.reload(fs_tools_module)

    written_path = fs_tools_module.write_text_file(str(target), "hello write")

    assert target.exists()
    assert target.read_text(encoding="utf-8") == "hello write"
    assert written_path.endswith("note.txt")


def test_list_directory_tree_traverses_nested_directories_by_default(monkeypatch, tmp_path):
    # Arrange: build a nested directory tree root/a/b/c/d/deep.txt.
    root = tmp_path / "root"
    deep_dir = root / "a" / "b" / "c" / "d"
    deep_dir.mkdir(parents=True)
    (deep_dir / "deep.txt").write_text("x", encoding="utf-8")

    # Allow the root as an accessible filesystem root.
    monkeypatch.setenv("LLM_COUNCIL_FS_ALLOWED_ROOTS", str(root))

    from backend import fs_tools as fs_tools_module

    importlib.reload(fs_tools_module)

    tree = fs_tools_module.list_directory_tree(str(root))

    # All nested directories and the deepest file should be present.
    assert "a/" in tree
    assert "b/" in tree
    assert "c/" in tree
    assert "d/" in tree
    assert "deep.txt" in tree


def test_list_directory_tree_respects_optional_max_entries(monkeypatch, tmp_path):
    # Arrange: create a directory with several files.
    root = tmp_path / "root"
    root.mkdir()
    for name in ["a.txt", "b.txt", "c.txt"]:
        (root / name).write_text("x", encoding="utf-8")

    monkeypatch.setenv("LLM_COUNCIL_FS_ALLOWED_ROOTS", str(root))

    from backend import fs_tools as fs_tools_module

    importlib.reload(fs_tools_module)

    # With an explicit max_entries we should see truncation.
    limited = fs_tools_module.list_directory_tree(str(root), max_entries=2)
    assert "...[truncated tree]..." in limited

    # With default arguments we should see the full set of files without truncation.
    full = fs_tools_module.list_directory_tree(str(root))
    assert "...[truncated tree]..." not in full
    assert "a.txt" in full and "b.txt" in full and "c.txt" in full


def test_list_directory_tree_supports_tilde_paths(monkeypatch, tmp_path):
    # Arrange: create a faux HOME directory with a project subdirectory.
    home = tmp_path / "home"
    project_root = home / "project"
    sub = project_root / "sub"
    sub.mkdir(parents=True)
    (sub / "file.txt").write_text("data", encoding="utf-8")

    # Configure allowed roots explicitly and set HOME so that "~" expands correctly.
    monkeypatch.setenv("LLM_COUNCIL_FS_ALLOWED_ROOTS", str(home))
    monkeypatch.setenv("HOME", str(home))

    from backend import fs_tools as fs_tools_module

    importlib.reload(fs_tools_module)

    tree = fs_tools_module.list_directory_tree("~/project")

    assert "sub/" in tree
    assert "file.txt" in tree


def test_collect_codebase_preview_respects_patterns_and_ignore(monkeypatch, tmp_path):
    # Arrange: build a small project tree with mixed files and an ignored directory.
    root = tmp_path / "root"
    src = root / "src"
    src.mkdir(parents=True)
    (src / "a.py").write_text("print('a')", encoding="utf-8")
    (src / "b.txt").write_text("ignored", encoding="utf-8")
    node_modules = root / "node_modules"
    node_modules.mkdir()
    (node_modules / "lib.js").write_text("should_not_appear", encoding="utf-8")

    monkeypatch.setenv("LLM_COUNCIL_FS_ALLOWED_ROOTS", str(root))

    from backend import fs_tools as fs_tools_module

    importlib.reload(fs_tools_module)

    preview = fs_tools_module.collect_codebase_preview(
        str(root),
        patterns=["*.py"],
        max_bytes_per_file=1024,
        max_files=10,
        max_depth=4,
    )

    assert "a.py" in preview
    assert "print('a')" in preview
    assert "b.txt" not in preview
    assert "lib.js" not in preview


def test_collect_codebase_preview_honors_max_files(monkeypatch, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    for i in range(5):
        (root / f"f{i}.py").write_text(f"print({i})", encoding="utf-8")

    monkeypatch.setenv("LLM_COUNCIL_FS_ALLOWED_ROOTS", str(root))

    from backend import fs_tools as fs_tools_module

    importlib.reload(fs_tools_module)

    preview = fs_tools_module.collect_codebase_preview(
        str(root),
        patterns=["*.py"],
        max_bytes_per_file=1024,
        max_files=2,
    )

    # We should only see at most 2 file headings.
    assert preview.count("### ") <= 2


def test_collect_codebase_preview_disallowed_root_raises(monkeypatch, tmp_path):
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    other_root = tmp_path / "other"
    other_root.mkdir()

    monkeypatch.setenv("LLM_COUNCIL_FS_ALLOWED_ROOTS", str(allowed_root))

    from backend import fs_tools as fs_tools_module

    importlib.reload(fs_tools_module)

    import pytest

    with pytest.raises(PermissionError):
        fs_tools_module.collect_codebase_preview(str(other_root))
