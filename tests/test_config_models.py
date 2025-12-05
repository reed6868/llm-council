import importlib
import os


def test_council_models_and_chairman_from_env(monkeypatch, tmp_path):
    # Ensure environment-driven configuration is used
    monkeypatch.setenv(
        "LLM_COUNCIL_MODELS",
        "openai:gpt-4.1-mini, qwen:qwen3-coder:free, grok:x-ai/grok-4.1-fast:free",
    )
    monkeypatch.setenv("LLM_CHAIRMAN_MODEL", "qwen:qwen3-coder:free")

    from backend import config as config_module
    # Avoid loading real .env values that could interfere with this
    # environment-driven configuration test.
    monkeypatch.setattr(config_module, "load_dotenv", lambda: None)
    importlib.reload(config_module)

    assert config_module.COUNCIL_MODELS == [
        "openai:gpt-4.1-mini",
        "qwen:qwen3-coder:free",
        "grok:x-ai/grok-4.1-fast:free",
    ]
    assert config_module.CHAIRMAN_MODEL == "qwen:qwen3-coder:free"


def test_council_models_without_explicit_chairman_uses_first(monkeypatch):
    monkeypatch.setenv(
        "LLM_COUNCIL_MODELS",
        "gemini:gemini-1.5-flash, anthropic:claude-3.5-sonnet",
    )
    monkeypatch.delenv("LLM_CHAIRMAN_MODEL", raising=False)

    from backend import config as config_module
    monkeypatch.setattr(config_module, "load_dotenv", lambda: None)
    importlib.reload(config_module)

    assert config_module.COUNCIL_MODELS == [
        "gemini:gemini-1.5-flash",
        "anthropic:claude-3.5-sonnet",
    ]
    # When not provided explicitly, chairman should default to the first model
    assert config_module.CHAIRMAN_MODEL == "gemini:gemini-1.5-flash"


def test_council_models_default_empty_when_not_configured(monkeypatch):
    # Ensure no council-related env is set
    monkeypatch.delenv("LLM_COUNCIL_MODELS", raising=False)
    monkeypatch.delenv("LLM_CHAIRMAN_MODEL", raising=False)

    from backend import config as config_module
    monkeypatch.setattr(config_module, "load_dotenv", lambda: None)
    importlib.reload(config_module)

    assert config_module.COUNCIL_MODELS == []
    assert config_module.CHAIRMAN_MODEL is None


def test_title_model_respects_env_override(monkeypatch):
    from backend import config as config_module

    monkeypatch.setattr(config_module, "load_dotenv", lambda: None)

    monkeypatch.setenv("LLM_COUNCIL_MODELS", "openai:gpt-4.1-mini")
    monkeypatch.setenv("LLM_CHAIRMAN_MODEL", "openai:gpt-4.1-mini")
    monkeypatch.setenv("TITLE_MODEL", "gemini:gemini-1.5-flash")

    importlib.reload(config_module)

    assert config_module.TITLE_MODEL == "gemini:gemini-1.5-flash"


def test_anthropic_api_version_env_only(monkeypatch):
    """ANTHROPIC_API_VERSION should be driven purely by env."""
    from backend import config as config_module

    # Clear any existing env and reload
    monkeypatch.delenv("ANTHROPIC_API_VERSION", raising=False)
    importlib.reload(config_module)
    assert config_module.ANTHROPIC_API_VERSION is None

    # When env is set, config should reflect it
    monkeypatch.setenv("ANTHROPIC_API_VERSION", "2025-01-01")
    importlib.reload(config_module)
    assert config_module.ANTHROPIC_API_VERSION == "2025-01-01"


def test_anthropic_slash_uses_openrouter_when_openrouter_configured(monkeypatch):
    """anthropic/... stays on OpenRouter when OPENROUTER_API_KEY is set."""
    from backend import config as config_module

    # Base config for this test
    monkeypatch.setattr(config_module, "load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_COUNCIL_MODELS", "anthropic/claude-3.5-sonnet")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    # Clear Anthropic direct config to ensure we don't accidentally route to it
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_VERSION", raising=False)

    importlib.reload(config_module)

    from backend import council_config

    importlib.reload(council_config)
    cfg = council_config.load_council_config()

    # Effective council models should keep the raw OpenRouter-style spec
    assert cfg.council_models == ["anthropic/claude-3.5-sonnet"]


def test_anthropic_slash_routes_to_direct_when_only_anthropic_config(monkeypatch):
    """anthropic/... requires OpenRouter; with only Anthropic configured it is filtered out."""
    from backend import config as config_module

    monkeypatch.setattr(config_module, "load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_COUNCIL_MODELS", "anthropic/claude-3.5-sonnet")

    # No OpenRouter, full Anthropic direct config
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak")
    monkeypatch.setenv("ANTHROPIC_API_BASE", "https://api.anthropic.com")
    monkeypatch.setenv("ANTHROPIC_API_VERSION", "2025-01-01")

    importlib.reload(config_module)

    from backend import council_config

    importlib.reload(council_config)
    cfg = council_config.load_council_config()

    # Without OpenRouter configured, an OpenRouter-style spec should be dropped.
    assert cfg.council_models == []
