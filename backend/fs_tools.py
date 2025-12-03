"""Local filesystem helpers for llm-council.

This module provides a minimal, security-conscious surface for reading
local files so that higher-level entrypoints (CLI / HTTP) can expose
selected filesystem content to LLM models.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List


def _get_allowed_roots() -> List[Path]:
    """Return the list of allowed root paths for filesystem access.

    The roots are derived from the LLM_COUNCIL_FS_ALLOWED_ROOTS
    environment variable, which should be a path-separated list
    (``os.pathsep``) of absolute or relative paths. If no roots are
    configured, the current user's home directory is used as a sensible
    default (e.g. ``/home/username``).
    """
    raw = os.getenv("LLM_COUNCIL_FS_ALLOWED_ROOTS", "")
    roots: List[Path] = []

    for part in raw.split(os.pathsep):
        part = part.strip()
        if not part:
            continue
        roots.append(Path(part).expanduser().resolve())

    # If no explicit roots are configured, default to the user's home
    # directory so that typical paths like /home/<user>/* are accessible.
    if not roots:
        roots.append(Path.home().resolve())

    return roots


def is_path_allowed(path: Path) -> bool:
    """Return True if the given path is within an allowed root."""
    allowed_roots = _get_allowed_roots()
    if not allowed_roots:
        return False

    try:
        resolved = path.expanduser().resolve()
    except FileNotFoundError:
        # Even if the file does not exist yet, we still enforce that the
        # eventual path would be under an allowed root.
        resolved = path.expanduser().absolute()

    for root in allowed_roots:
        if resolved == root or root in resolved.parents:
            return True
    return False


def read_file_preview(path_str: str, max_bytes: int = 8192) -> str:
    """Read a UTF-8 preview of a file, enforcing allowed roots.

    Args:
        path_str: Path to the file.
        max_bytes: Maximum number of bytes to read; content is truncated
            with an indicator if the file is larger.

    Returns:
        File content as a UTF-8 string (possibly truncated).

    Raises:
        PermissionError: If the path is outside the allowed roots.
        FileNotFoundError: If the file does not exist.
        OSError: For other filesystem-level errors.
    """
    path = Path(path_str)

    if not is_path_allowed(path):
        raise PermissionError(f"Access to {path_str} is not allowed")

    if not path.is_file():
        raise FileNotFoundError(path_str)

    # Read up to max_bytes + 1 so that we can decide whether to mark truncation.
    with path.open("rb") as f:
        data = f.read(max_bytes + 1)

    truncated = len(data) > max_bytes
    if truncated:
        data = data[:max_bytes]

    text = data.decode("utf-8", errors="replace")
    if truncated:
        text += "\n...[truncated]..."
    return text


def list_directory_tree(path_str: str, max_depth: int = 3, max_entries: int = 1000) -> str:
    """Return a textual tree view of a directory within allowed roots.

    Args:
        path_str: Directory path to list.
        max_depth: Maximum directory depth to traverse (root is depth 0).
        max_entries: Maximum number of entries to include overall.

    Returns:
        A newline-separated tree view of the directory contents.

    Raises:
        PermissionError: If the path is outside the allowed roots.
        FileNotFoundError: If the path does not exist.
        NotADirectoryError: If the path is not a directory.
    """
    path = Path(path_str)

    if not is_path_allowed(path):
        raise PermissionError(f"Access to {path_str} is not allowed")

    if not path.exists():
        raise FileNotFoundError(path_str)

    if not path.is_dir():
        # If a file is passed, just return its name for convenience.
        return path.name

    root = path.resolve()
    lines: List[str] = [str(root)]
    count = 0

    def walk(current: Path, depth: int) -> None:
        nonlocal count
        if depth > max_depth or count >= max_entries:
            return

        try:
            entries = sorted(
                [p for p in current.iterdir() if not p.name.startswith(".")],
                key=lambda p: (p.is_file(), p.name.lower()),
            )
        except OSError:
            lines.append("  " * depth + "[Error reading directory]")
            return

        for entry in entries:
            if count >= max_entries:
                break
            indent = "  " * (depth + 1)
            if entry.is_dir():
                lines.append(f"{indent}{entry.name}/")
                count += 1
                walk(entry, depth + 1)
            else:
                lines.append(f"{indent}{entry.name}")
                count += 1

    walk(root, 0)
    if len(lines) == 1:
        lines.append("  (empty)")
    elif count >= max_entries:
        lines.append("  ...[truncated tree]...")
    return "\n".join(lines)


def write_text_file(path_str: str, content: str, overwrite: bool = False, max_bytes: int = 1024 * 1024) -> str:
    """Write UTF-8 text to a file within allowed roots.

    Args:
        path_str: Target file path.
        content: Text content to write.
        overwrite: Whether to allow overwriting an existing file.
        max_bytes: Maximum allowed size of the encoded content.

    Returns:
        The absolute path of the written file as a string.

    Raises:
        PermissionError: If the path is outside the allowed roots.
        FileExistsError: If the file exists and overwrite is False.
        ValueError: If content exceeds max_bytes when encoded.
        OSError: For other filesystem-level errors.
    """
    path = Path(path_str).expanduser()

    if not is_path_allowed(path):
        raise PermissionError(f"Access to {path_str} is not allowed")

    data = content.encode("utf-8")
    if len(data) > max_bytes:
        raise ValueError(f"Content too large to write (>{max_bytes} bytes)")

    # Ensure parent directory exists.
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {path_str}")

    with path.open("wb") as f:
        f.write(data)

    return str(path.resolve())


__all__ = ["is_path_allowed", "read_file_preview", "list_directory_tree", "write_text_file"]
