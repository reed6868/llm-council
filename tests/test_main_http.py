import asyncio

import pytest

from backend import main as api_main


@pytest.mark.asyncio
async def test_send_message_uses_prepare_council_input_for_first_message(monkeypatch):
    # Arrange a conversation with no prior messages so that this is
    # treated as the first message.
    conversation = {
        "id": "conv-1",
        "created_at": "2024-01-01T00:00:00",
        "title": "New Conversation",
        "messages": [],
    }

    monkeypatch.setattr(api_main.storage, "get_conversation", lambda cid: conversation)
    monkeypatch.setattr(api_main.storage, "add_user_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        api_main.storage,
        "update_conversation_title",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        api_main.storage,
        "add_assistant_message",
        lambda *args, **kwargs: None,
    )

    async def fake_generate_conversation_title(content: str) -> str:
        return "Title"

    monkeypatch.setattr(api_main, "generate_conversation_title", fake_generate_conversation_title)

    calls = {"prepared": None, "council_input": None}

    def fake_prepare_council_input(content: str, is_first_message: bool) -> str:
        prepared = f"CTX::{content}::{is_first_message}"
        calls["prepared"] = prepared
        return prepared

    monkeypatch.setattr(api_main, "prepare_council_input", fake_prepare_council_input)

    async def fake_run_full_council(council_input: str):
        calls["council_input"] = council_input
        return ["s1"], ["s2"], {"model": "m", "response": "r"}, {"meta": True}

    monkeypatch.setattr(api_main, "run_full_council", fake_run_full_council)

    request = api_main.SendMessageRequest(content="hello")

    # Act
    resp = await api_main.send_message("conv-1", request)

    # Assert: the council should see the prepared input, not the raw content.
    assert calls["prepared"] is not None
    assert calls["council_input"] == calls["prepared"]
    assert resp["stage1"] == ["s1"]
    assert resp["stage2"] == ["s2"]
    assert resp["stage3"]["model"] == "m"
    assert resp["metadata"] == {"meta": True}


@pytest.mark.asyncio
async def test_send_message_marks_followup_as_not_first(monkeypatch):
    # Arrange a conversation that already has at least one message so
    # that the next message is treated as a follow-up.
    conversation = {
        "id": "conv-2",
        "created_at": "2024-01-01T00:00:00",
        "title": "Existing",
        "messages": [{"role": "user", "content": "earlier"}],
    }

    monkeypatch.setattr(api_main.storage, "get_conversation", lambda cid: conversation)
    monkeypatch.setattr(api_main.storage, "add_user_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        api_main.storage,
        "update_conversation_title",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        api_main.storage,
        "add_assistant_message",
        lambda *args, **kwargs: None,
    )

    async def fake_run_full_council(council_input: str):
        return [], [], {"model": "m", "response": "r"}, {}

    monkeypatch.setattr(api_main, "run_full_council", fake_run_full_council)

    calls = {"flags": []}

    def fake_prepare_council_input(content: str, is_first_message: bool) -> str:
        calls["flags"].append(is_first_message)
        return content

    monkeypatch.setattr(api_main, "prepare_council_input", fake_prepare_council_input)

    request = api_main.SendMessageRequest(content="followup")

    # Act
    await api_main.send_message("conv-2", request)

    # Assert: this should be treated as a follow-up message.
    assert calls["flags"] == [False]


@pytest.mark.asyncio
async def test_send_message_stream_uses_prepare_council_input(monkeypatch):
    # Arrange a conversation with no messages so this is the first.
    conversation = {
        "id": "conv-stream",
        "created_at": "2024-01-01T00:00:00",
        "title": "Stream Conversation",
        "messages": [],
    }

    monkeypatch.setattr(api_main.storage, "get_conversation", lambda cid: conversation)
    monkeypatch.setattr(api_main.storage, "add_user_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        api_main.storage,
        "update_conversation_title",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        api_main.storage,
        "add_assistant_message",
        lambda *args, **kwargs: None,
    )

    calls = {"prepared": None, "council_input": None}

    def fake_prepare_council_input(content: str, is_first_message: bool) -> str:
        prepared = f"CTX::{content}::{is_first_message}"
        calls["prepared"] = prepared
        return prepared

    monkeypatch.setattr(api_main, "prepare_council_input", fake_prepare_council_input)

    async def fake_run_full_council(council_input: str):
        calls["council_input"] = council_input
        # Minimal but structurally valid return
        return (
            ["s1"],
            ["s2"],
            {"model": "m", "response": "r"},
            {
                "label_to_model": {"Response A": "model-a"},
                "aggregate_rankings": [
                    {"model": "model-a", "average_rank": 1.0, "rankings_count": 1}
                ],
            },
        )

    monkeypatch.setattr(api_main, "run_full_council", fake_run_full_council)

    request = api_main.SendMessageRequest(content="hello-stream")

    # Act: obtain the streaming response and consume enough chunks to
    # ensure run_full_council has been awaited.
    response = await api_main.send_message_stream("conv-stream", request)

    assert hasattr(response, "body_iterator")

    async for chunk in response.body_iterator:
        # The first yield occurs before run_full_council; once the
        # council call has happened, our fake will have recorded the
        # prepared input.
        if calls["council_input"] is not None:
            break

    # Assert: streaming path should also use the prepared input.
    assert calls["prepared"] is not None
    assert calls["council_input"] == calls["prepared"]

