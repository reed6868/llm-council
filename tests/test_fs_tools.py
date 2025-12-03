import importlib


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
