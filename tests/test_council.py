import importlib
import pytest


@pytest.mark.asyncio
async def test_stage2_uses_only_successful_stage1_models(monkeypatch):
    # Arrange: configure two council models
    monkeypatch.setenv("LLM_COUNCIL_MODELS", "model-a,model-b")
    monkeypatch.setenv("LLM_CHAIRMAN_MODEL", "model-a")
    # Treat bare model ids as OpenRouter-style; ensure it is considered configured.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    from backend import config as config_module

    importlib.reload(config_module)

    from backend import council as council_module

    importlib.reload(council_module)

    recorded_calls = {"stage1_models": None, "stage2_models": None}

    async def fake_query_models_parallel(models, messages):
        # Stage 1 prompt does not contain "FINAL RANKING"
        if "FINAL RANKING:" not in messages[0]["content"]:
            recorded_calls["stage1_models"] = list(models)
            # Simulate model-a success, model-b failure
            return {
                "model-a": {"content": "answer from a"},
                "model-b": None,
            }

        # Stage 2 ranking prompt
        recorded_calls["stage2_models"] = list(models)
        # Only model-a should be asked to rank
        return {
            "model-a": {
                "content": (
                    "Some analysis\n\nFINAL RANKING:\n1. Response A\n"
                )
            }
        }

    async def fake_query_model(model_spec, messages, timeout=120.0):
        return {"content": "final answer from chairman"}

    monkeypatch.setattr(council_module, "query_models_parallel", fake_query_models_parallel)
    monkeypatch.setattr(council_module, "query_model", fake_query_model)

    # Act
    stage1, stage2, stage3, metadata = await council_module.run_full_council("test question")

    # Assert: Stage1 saw both models
    assert recorded_calls["stage1_models"] == ["model-a", "model-b"]
    # Stage2 should only query the model that succeeded in Stage1
    assert recorded_calls["stage2_models"] == ["model-a"]
    # Pipeline still produces sensible outputs
    assert len(stage1) == 1
    assert stage3["response"] == "final answer from chairman"


@pytest.mark.asyncio
async def test_run_full_council_propagates_failures_when_no_valid_models(monkeypatch):
    # Configure a council model that requires OpenRouter but without a valid key.
    monkeypatch.setenv("LLM_COUNCIL_MODELS", "google/gemini-3-pro-preview")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    import dotenv
    from backend import config as config_module

    # Avoid external .env interference
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    importlib.reload(config_module)

    from backend import council as council_module

    importlib.reload(council_module)

    async def fake_query_models_parallel(models, messages):
        # All models will effectively fail at runtime as well.
        return {model: None for model in models}

    async def fake_query_model(model_spec, messages, timeout=120.0):
        return None

    monkeypatch.setattr(council_module, "query_models_parallel", fake_query_models_parallel)
    monkeypatch.setattr(council_module, "query_model", fake_query_model)

    stage1, stage2, stage3, metadata = await council_module.run_full_council("test question")

    assert stage1 == []
    assert stage2 == []
    assert stage3["model"] == "error"
    assert "failures" in metadata
    assert metadata["failures"]
