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
