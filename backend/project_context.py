"""Project context helpers for llm-council.

This module is responsible for resolving the project root directory and
building an initial, bounded view of the local codebase (directory tree
and code previews) that can be injected into model prompts.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import fs_tools


def resolve_project_root() -> Path:
    """Resolve the active project root directory.

    The root is determined by the LLM_COUNCIL_PROJECT_ROOT environment
    variable when set; otherwise the current working directory is used.
    """
    raw = os.getenv("LLM_COUNCIL_PROJECT_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.cwd().resolve()


def build_project_context(
    root: Path,
    max_tree_depth: int | None = 3,
    max_tree_entries: int | None = 200,
    code_kwargs: dict | None = None,
) -> str:
    """Build a textual summary of a project rooted at the given path.

    The summary includes:
    - project root path
    - a directory tree view (bounded by depth and entry count when provided)
    - a codebase preview using fs_tools.collect_codebase_preview()
    """
    sections: list[str] = [f"[PROJECT ROOT] {root}"]

    try:
        tree = fs_tools.list_directory_tree(
            str(root),
            max_depth=max_tree_depth,
            max_entries=max_tree_entries,
        )
        sections.append("[PROJECT TREE]")
        sections.append(tree)
    except Exception as e:
        sections.append(f"[Error collecting project tree: {e}]")

    try:
        kwargs = code_kwargs or {}
        code = fs_tools.collect_codebase_preview(str(root), **kwargs)
        sections.append("[PROJECT CODEBASE]")
        sections.append(code)
    except Exception as e:
        sections.append(f"[Error collecting project codebase: {e}]")

    return "\n".join(sections)


def build_initial_project_context(
    max_tree_depth: int = 3,
    max_tree_entries: int = 200,
) -> str:
    """Build a textual summary of the current project.

    This is a convenience wrapper around build_project_context() that
    uses the resolved project root and default bounds.
    """
    root = resolve_project_root()
    return build_project_context(
        root,
        max_tree_depth=max_tree_depth,
        max_tree_entries=max_tree_entries,
        code_kwargs=None,
    )


__all__ = [
    "resolve_project_root",
    "build_project_context",
    "build_initial_project_context",
]
