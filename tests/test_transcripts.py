from pathlib import Path

import importlib
import pytest

from backend import transcripts


def _make_sample_conversation():
    return {
        "id": "test-conv",
        "created_at": "2024-01-01T00:00:00Z",
        "title": "Sample Title",
        "messages": [
            {"role": "user", "content": "Hello council"},
            {
                "role": "assistant",
                "stage1": [
                    {"model": "model-a", "response": "Answer A"},
                    {"model": "model-b", "response": "Answer B"},
                ],
                "stage2": [
                    {
                        "model": "model-a",
                        "ranking": "Ranking text",
                        "parsed_ranking": ["Response A", "Response B"],
                    },
                ],
                "stage3": {
                    "model": "chairman-model",
                    "response": "Final answer",
                },
            },
        ],
    }


def test_write_markdown_transcript_creates_file(tmp_path, monkeypatch):
    # Redirect transcript output directory to a temporary path
    monkeypatch.setattr(transcripts, "DATA_DIR", tmp_path)

    conversation = _make_sample_conversation()

    transcripts.write_markdown_transcript(conversation)

    path = Path(transcripts.get_transcript_path(conversation["id"]))
    assert path.exists()

    content = path.read_text(encoding="utf-8")

    # Basic structure checks
    assert "# Conversation Sample Title" in content
    assert "## User" in content
    assert "Hello council" in content
    assert "## Council Stage 1" in content
    assert "model-a" in content
    assert "Answer A" in content
    assert "## Council Stage 3 (Chairman)" in content
    assert "Final answer" in content


def test_transcripts_uses_config_data_dir_env_override(tmp_path, monkeypatch):
    custom_dir = tmp_path / "conversations"
    monkeypatch.setenv("LLM_COUNCIL_DATA_DIR", str(custom_dir))

    from backend import config as config_module
    importlib.reload(config_module)

    # Reload transcripts so that it picks up the new DATA_DIR from config
    import backend.transcripts as transcripts_module
    importlib.reload(transcripts_module)

    conversation = _make_sample_conversation()
    transcripts_module.write_markdown_transcript(conversation)

    path = Path(transcripts_module.get_transcript_path(conversation["id"]))
    assert path.exists()
    # Ensure we're writing under the overridden directory
    assert str(custom_dir) in str(path)
