import importlib


def _reload_modules(monkeypatch):
    import dotenv
    from backend import config as config_module

    # Avoid picking up any real .env on disk; tests should control env fully.
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    importlib.reload(config_module)

    from backend import council_config as council_config_module

    importlib.reload(council_config_module)
    return config_module, council_config_module


def test_load_council_config_all_valid(monkeypatch):
    # Configure two valid models with complete provider settings.
    monkeypatch.setenv(
        "LLM_COUNCIL_MODELS",
        "openai:gpt-4.1-mini,gemini:gemini-1.5-pro",
    )
    monkeypatch.setenv("LLM_CHAIRMAN_MODEL", "gemini:gemini-1.5-pro")
    monkeypatch.setenv("TITLE_MODEL", "gemini:gemini-2.5-flash")

    # Provider env for openai and gemini
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    monkeypatch.setenv("GEMINI_API_KEY", "sk-gemini")
    monkeypatch.setenv("GEMINI_API_BASE", "https://example-gemini.com")

    config_module, council_config_module = _reload_modules(monkeypatch)

    cfg = council_config_module.load_council_config()

    assert cfg.council_models == [
        "openai:gpt-4.1-mini",
        "gemini:gemini-1.5-pro",
    ]
    assert cfg.chairman_model == "gemini:gemini-1.5-pro"
    assert cfg.title_model == "gemini:gemini-2.5-flash"
    assert cfg.failures == []


def test_load_council_config_filters_misconfigured_models(monkeypatch):
    # One OpenRouter-style model without OPENROUTER_API_KEY, one valid OpenAI model.
    monkeypatch.setenv(
        "LLM_COUNCIL_MODELS",
        "google/gemini-3-pro-preview,openai:gpt-4.1-mini",
    )
    # Only configure OpenAI
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    # Force OPENROUTER_API_KEY to be effectively missing for this test.
    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    config_module, council_config_module = _reload_modules(monkeypatch)

    cfg = council_config_module.load_council_config()

    # Misconfigured OpenRouter model should be filtered out.
    assert cfg.council_models == ["openai:gpt-4.1-mini"]
    assert cfg.chairman_model == "openai:gpt-4.1-mini"
    assert cfg.failures
    failure = cfg.failures[0]
    assert failure["model_spec"] == "google/gemini-3-pro-preview"
    assert failure["error_type"] == "config_error"
    assert "OPENROUTER_API_KEY" in failure["missing"]


def test_load_council_config_invalid_chairman_falls_back(monkeypatch):
    # Only one valid council model but chairman points elsewhere.
    monkeypatch.setenv("LLM_COUNCIL_MODELS", "openai:gpt-4.1-mini")
    monkeypatch.setenv("LLM_CHAIRMAN_MODEL", "nonexistent:model")

    # Configure OpenAI so it is considered valid.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai.com/v1")

    config_module, council_config_module = _reload_modules(monkeypatch)

    cfg = council_config_module.load_council_config()

    assert cfg.council_models == ["openai:gpt-4.1-mini"]
    # Chairman should fall back to first effective model.
    assert cfg.chairman_model == "openai:gpt-4.1-mini"
    assert any(f["error_type"] == "invalid_chairman" for f in cfg.failures)
