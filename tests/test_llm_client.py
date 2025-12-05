import pytest
import httpx

from backend import llm_client
from backend.llm_client import ConfigError


def test_parse_model_spec_with_explicit_provider():
    provider, name = llm_client.parse_model_spec("openai:gpt-4.1-mini")
    assert provider == "openai"
    assert name == "gpt-4.1-mini"


def test_parse_model_spec_without_provider_defaults_to_openrouter():
    provider, name = llm_client.parse_model_spec("google/gemini-3-pro-preview")
    assert provider == "openrouter"
    assert name == "google/gemini-3-pro-preview"


def test_supported_providers_include_expected_keys():
    expected = {"openrouter", "openai", "gemini", "anthropic", "glm", "qwen", "grok"}
    for key in expected:
        assert key in llm_client.PROVIDERS


@pytest.mark.asyncio
async def test_query_models_parallel_uses_query_model(monkeypatch):
    called = []

    async def fake_query_model(model_spec, messages, timeout=120.0):
        called.append((model_spec, messages, timeout))
        return {"content": f"resp-{model_spec}"}

    monkeypatch.setattr(llm_client, "query_model", fake_query_model)

    models = ["openai:gpt-4.1-mini", "anthropic:claude-3.5-sonnet"]
    messages = [{"role": "user", "content": "hi"}]

    results = await llm_client.query_models_parallel(models, messages)

    assert set(results.keys()) == set(models)
    assert results["openai:gpt-4.1-mini"]["content"] == "resp-openai:gpt-4.1-mini"
    assert len(called) == 2


@pytest.mark.asyncio
async def test_query_model_returns_none_on_config_error(monkeypatch):
    async def fake_provider(model_name, messages, timeout):
        raise ConfigError("config problem")

    # Inject fake provider for 'openai'
    monkeypatch.setitem(llm_client.PROVIDERS, "openai", fake_provider)

    result = await llm_client.query_model(
        "openai:gpt-4.1-mini",
        [{"role": "user", "content": "hi"}],
    )
    assert result is None


@pytest.mark.asyncio
async def test_query_model_returns_none_on_http_401(monkeypatch):
    async def fake_provider(model_name, messages, timeout):
        request = httpx.Request("POST", "https://example.com")
        response = httpx.Response(401, request=request)
        raise httpx.HTTPStatusError("Unauthorized", request=request, response=response)

    monkeypatch.setitem(llm_client.PROVIDERS, "openai", fake_provider)

    result = await llm_client.query_model(
        "openai:gpt-4.1-mini",
        [{"role": "user", "content": "hi"}],
    )
    assert result is None


def test_to_anthropic_messages_merges_consecutive_roles_and_inlines_system():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a1"},
        {"role": "assistant", "content": "a2"},
        {"role": "tool", "content": "tool-msg"},
    ]

    converted = llm_client._to_anthropic_messages(messages)

    # Expect three messages:
    # 1. user: system + both user messages merged
    # 2. assistant: both assistant messages merged
    # 3. user: tool message (normalized to user)
    assert len(converted) == 3

    first = converted[0]
    assert first["role"] == "user"
    assert "[SYSTEM]" in first["content"]
    assert "sys" in first["content"]
    assert "u1" in first["content"]
    assert "u2" in first["content"]

    second = converted[1]
    assert second["role"] == "assistant"
    assert "a1" in second["content"]
    assert "a2" in second["content"]

    third = converted[2]
    assert third["role"] == "user"
    assert "tool-msg" in third["content"]


def test_to_gemini_contents_normalizes_roles_and_merges_consecutive():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "assistant", "content": "a2"},
    ]

    contents = llm_client._to_gemini_contents(messages)

    # System + first user message become a single "user" turn,
    # assistants are merged into a single "model" turn.
    assert len(contents) == 2

    first = contents[0]
    assert first["role"] == "user"
    user_text = first["parts"][0]["text"]
    assert "[SYSTEM]" in user_text
    assert "sys" in user_text
    assert "u1" in user_text

    second = contents[1]
    assert second["role"] == "model"
    model_text = second["parts"][0]["text"]
    assert "a1" in model_text
    assert "a2" in model_text
