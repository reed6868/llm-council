import importlib

import httpx
import pytest


@pytest.mark.asyncio
async def test_query_model_openrouter_401_soft_failure(monkeypatch, capsys):
    # Ensure OpenRouter is considered configured.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    # Reload config and llm_client to pick up env changes.
    from backend import config as config_module

    importlib.reload(config_module)

    from backend import llm_client as llm_client_module

    importlib.reload(llm_client_module)

    # Fake AsyncClient that always returns a 401-like response.
    class DummyResponse:
        def __init__(self) -> None:
            self.status_code = 401
            self._request = httpx.Request(
                "POST", config_module.OPENROUTER_API_URL
            )

        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError(
                "401 Unauthorized",
                request=self._request,
                response=httpx.Response(
                    status_code=self.status_code,
                    request=self._request,
                ),
            )

        def json(self) -> dict:
            return {}

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            return DummyResponse()

    # Force llm_client to use the dummy client so that OpenRouter
    # calls raise HTTPStatusError(401).
    monkeypatch.setattr(
        llm_client_module.httpx,
        "AsyncClient",
        DummyAsyncClient,
    )

    messages = [{"role": "user", "content": "test question"}]

    result = await llm_client_module.query_model(
        "google/gemini-3-pro-preview",
        messages,
        timeout=5.0,
    )

    captured = capsys.readouterr()

    # The unified client should treat 401 as a soft failure and return None.
    assert result is None
    # And it should emit a clear, provider-aware message.
    assert "provider 'openrouter'" in captured.out
    assert "401 Unauthorized" in captured.out

