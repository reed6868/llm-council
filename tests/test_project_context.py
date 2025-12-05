from pathlib import Path

from backend import project_context


def test_resolve_project_root_defaults_to_cwd(monkeypatch, tmp_path):
    # When no explicit project root is configured, use the current
    # working directory as the project root.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_COUNCIL_PROJECT_ROOT", raising=False)

    root = project_context.resolve_project_root()

    assert root == tmp_path.resolve()


def test_resolve_project_root_uses_env(monkeypatch, tmp_path):
    # LLM_COUNCIL_PROJECT_ROOT should override the current directory.
    project_root = tmp_path / "project"
    project_root.mkdir()

    monkeypatch.setenv("LLM_COUNCIL_PROJECT_ROOT", str(project_root))

    root = project_context.resolve_project_root()

    assert root == project_root.resolve()


def test_build_initial_project_context_includes_tree_and_code(monkeypatch):
    # Build a project context that includes both a directory tree and a
    # codebase preview using the configured root.
    calls = {"root": None, "tree": 0, "code": 0}

    def fake_resolve_project_root() -> Path:
        calls["root"] = Path("/project")
        return calls["root"]

    def fake_list_directory_tree(path_str, max_depth=None, max_entries=None):
        calls["tree"] += 1
        return "TREE"

    def fake_collect_codebase_preview(path_str, **kwargs):
        calls["code"] += 1
        return "CODE"

    monkeypatch.setattr(project_context, "resolve_project_root", fake_resolve_project_root)
    monkeypatch.setattr(
        project_context.fs_tools,
        "list_directory_tree",
        fake_list_directory_tree,
    )
    monkeypatch.setattr(
        project_context.fs_tools,
        "collect_codebase_preview",
        fake_collect_codebase_preview,
    )

    ctx = project_context.build_initial_project_context()

    assert calls["root"] == Path("/project")
    assert calls["tree"] == 1
    assert calls["code"] == 1
    assert "[PROJECT ROOT]" in ctx
    assert "/project" in ctx
    assert "[PROJECT TREE]" in ctx
    assert "TREE" in ctx
    assert "[PROJECT CODEBASE]" in ctx
    assert "CODE" in ctx


def test_build_initial_project_context_handles_errors_gracefully(monkeypatch):
    # If collecting the tree or codebase fails, the error should be
    # captured in the context text instead of bubbling up.
    def fake_resolve_project_root() -> Path:
        return Path("/broken")

    def fail_list_directory_tree(*args, **kwargs):
        raise RuntimeError("tree failure")

    def fake_collect_codebase_preview(path_str, **kwargs):
        return "CODE"

    monkeypatch.setattr(project_context, "resolve_project_root", fake_resolve_project_root)
    monkeypatch.setattr(
        project_context.fs_tools,
        "list_directory_tree",
        fail_list_directory_tree,
    )
    monkeypatch.setattr(
        project_context.fs_tools,
        "collect_codebase_preview",
        fake_collect_codebase_preview,
    )

    ctx = project_context.build_initial_project_context()

    assert "[Error collecting project tree:" in ctx
    assert "[PROJECT CODEBASE]" in ctx
    assert "CODE" in ctx


def test_build_project_context_uses_given_root_and_kwargs(monkeypatch, tmp_path):
    # Given an explicit project root and custom limits, build_project_context
    # should call fs_tools helpers with the provided arguments and include
    # their outputs in the context text.
    calls = {"tree": [], "code": []}

    root = tmp_path / "project"
    root.mkdir()

    def fake_list_directory_tree(path_str, max_depth=None, max_entries=None):
        calls["tree"].append((path_str, max_depth, max_entries))
        return "TREE"

    def fake_collect_codebase_preview(path_str, **kwargs):
        calls["code"].append((path_str, kwargs))
        return "CODE"

    monkeypatch.setattr(
        project_context.fs_tools,
        "list_directory_tree",
        fake_list_directory_tree,
    )
    monkeypatch.setattr(
        project_context.fs_tools,
        "collect_codebase_preview",
        fake_collect_codebase_preview,
    )

    ctx = project_context.build_project_context(
        root,
        max_tree_depth=10,
        max_tree_entries=500,
        code_kwargs={"max_files": 100},
    )

    # list_directory_tree should be called once with the given root and limits.
    assert calls["tree"] == [(str(root), 10, 500)]
    # collect_codebase_preview should be called once with the given root and kwargs.
    assert calls["code"] == [(str(root), {"max_files": 100})]

    # The returned context should include the project root, tree, and code sections.
    assert "[PROJECT ROOT]" in ctx
    assert str(root) in ctx
    assert "[PROJECT TREE]" in ctx
    assert "TREE" in ctx
    assert "[PROJECT CODEBASE]" in ctx
    assert "CODE" in ctx
