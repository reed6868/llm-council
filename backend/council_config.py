"""Council configuration helpers.

This module derives the *effective* council configuration from environment-
driven settings in ``backend.config`` and basic provider-level validation.
It does **not** perform any network I/O; it only checks that the required
API keys / base URLs are present for each provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from . import config
from .llm_client import parse_model_spec


@dataclass
class CouncilEffectiveConfig:
    council_models: List[str]
    chairman_model: Optional[str]
    title_model: Optional[str]
    failures: List[Dict[str, Any]]


def _missing_env_for_provider(provider: str) -> List[str]:
    """Return a list of missing env-backed settings for a provider."""
    missing: List[str] = []

    if provider == "openrouter":
        if not config.OPENROUTER_API_KEY:
            missing.append("OPENROUTER_API_KEY")
        return missing

    if provider == "openai":
        if not config.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not config.OPENAI_API_BASE:
            missing.append("OPENAI_API_BASE")
        return missing

    if provider == "gemini":
        if not config.GEMINI_API_KEY:
            missing.append("GEMINI_API_KEY")
        if not config.GEMINI_API_BASE:
            missing.append("GEMINI_API_BASE")
        return missing

    if provider == "anthropic":
        if not config.ANTHROPIC_API_KEY:
            missing.append("ANTHROPIC_API_KEY")
        if not config.ANTHROPIC_API_BASE:
            missing.append("ANTHROPIC_API_BASE")
        if not config.ANTHROPIC_API_VERSION:
            missing.append("ANTHROPIC_API_VERSION")
        return missing

    if provider == "glm":
        if not config.GLM_API_KEY:
            missing.append("GLM_API_KEY")
        if not config.GLM_API_BASE:
            missing.append("GLM_API_BASE")
        return missing

    if provider == "qwen":
        if not config.QWEN_API_KEY:
            missing.append("QWEN_API_KEY")
        if not config.QWEN_API_BASE:
            missing.append("QWEN_API_BASE")
        return missing

    if provider == "grok":
        if not config.GROK_API_KEY:
            missing.append("GROK_API_KEY")
        if not config.GROK_API_BASE:
            missing.append("GROK_API_BASE")
        return missing

    # Unknown providers are treated as-is; we do not know the required env vars.
    return missing


def load_council_config() -> CouncilEffectiveConfig:
    """Compute the effective council configuration from env-driven settings."""
    failures: List[Dict[str, Any]] = []
    effective_models: List[str] = []

    # Filter council models by provider-level configuration.
    for spec in config.COUNCIL_MODELS:
        provider, _ = parse_model_spec(spec)
        missing = _missing_env_for_provider(provider)
        if missing:
            failures.append(
                {
                    "model_spec": spec,
                    "provider": provider,
                    "error_type": "config_error",
                    "missing": missing,
                }
            )
            continue
        effective_models.append(spec)

    # Determine chairman model based on effective models.
    chairman_model: Optional[str] = None
    explicit_chairman = config.CHAIRMAN_MODEL
    if effective_models:
        if explicit_chairman and explicit_chairman in effective_models:
            chairman_model = explicit_chairman
        elif explicit_chairman and explicit_chairman not in effective_models:
            # Env points to a model that is not in the effective set.
            failures.append(
                {
                    "model_spec": explicit_chairman,
                    "provider": parse_model_spec(explicit_chairman)[0],
                    "error_type": "invalid_chairman",
                    "missing": [],
                }
            )
            chairman_model = effective_models[0]
        else:
            chairman_model = effective_models[0]

    # Determine title model; if misconfigured, fall back to chairman.
    title_model: Optional[str] = config.TITLE_MODEL
    if title_model:
        provider, _ = parse_model_spec(title_model)
        missing = _missing_env_for_provider(provider)
        if missing:
            failures.append(
                {
                    "model_spec": title_model,
                    "provider": provider,
                    "error_type": "config_error",
                    "missing": missing,
                }
            )
            title_model = chairman_model
    else:
        title_model = chairman_model

    return CouncilEffectiveConfig(
        council_models=effective_models,
        chairman_model=chairman_model,
        title_model=title_model,
        failures=failures,
    )

