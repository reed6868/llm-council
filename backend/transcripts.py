"""Markdown transcript generation for conversations.

This module mirrors the JSON conversation structure in a VS Code–friendly
Markdown format so that the full council + chairman discussion is easy to
inspect and diff.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from .config import DATA_DIR as CONFIG_DATA_DIR

# Allow tests to override this (e.g. with a temporary directory).
DATA_DIR = CONFIG_DATA_DIR


def get_transcript_path(conversation_id: str) -> str:
    """Return the markdown transcript path for a conversation."""
    base = os.fspath(DATA_DIR)
    return os.path.join(base, f"{conversation_id}.md")


def _render_user_message(content: str) -> List[str]:
    return ["## User", "", content.strip()]


def _render_stage1(stage1: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = ["## Council Stage 1", ""]
    for result in stage1:
        model = result.get("model", "")
        response = result.get("response", "")
        if not model and not response:
            continue
        lines.append(f"### {model}")
        lines.append("")
        lines.append(str(response).strip())
        lines.append("")
    return lines


def _render_stage2(stage2: List[Dict[str, Any]]) -> List[str]:
    if not stage2:
        return []
    lines: List[str] = ["## Council Stage 2", ""]
    for ranking in stage2:
        model = ranking.get("model", "")
        ranking_text = ranking.get("ranking", "")
        lines.append(f"### {model}")
        lines.append("")
        lines.append(str(ranking_text).strip())
        lines.append("")
    return lines


def _render_stage3(stage3: Dict[str, Any]) -> List[str]:
    lines: List[str] = ["## Council Stage 3 (Chairman)", ""]
    model = stage3.get("model")
    response = stage3.get("response", "")
    if model:
        lines.append(f"Model: {model}")
        lines.append("")
    lines.append(str(response).strip())
    return lines


def write_markdown_transcript(conversation: Dict[str, Any]) -> None:
    """Write a markdown transcript for the given conversation."""
    os.makedirs(os.fspath(DATA_DIR), exist_ok=True)

    title = conversation.get("title") or "New Conversation"
    lines: List[str] = [f"# Conversation {title}", ""]

    for message in conversation.get("messages", []):
        role = message.get("role")

        if role == "user":
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            lines.extend(_render_user_message(content))
            lines.append("")
            continue

        if role == "assistant":
            stage1 = message.get("stage1") or []
            stage2 = message.get("stage2") or []
            stage3 = message.get("stage3") or {}

            if stage1:
                lines.append("")
                lines.extend(_render_stage1(stage1))
            if stage2:
                lines.append("")
                lines.extend(_render_stage2(stage2))
            if stage3:
                lines.append("")
                lines.extend(_render_stage3(stage3))
            lines.append("")

    path = Path(get_transcript_path(conversation["id"]))
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

