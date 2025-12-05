"""Unified LLM client supporting multiple providers.

This module provides a small abstraction layer over different LLM HTTP APIs.
It keeps a simple OpenAI-style messages interface and delegates to specific
providers based on a ``provider:model`` model spec string.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple, Awaitable, Callable

import httpx

from . import config


class ConfigError(Exception):
    """Configuration error for a specific provider/model."""


ProviderFn = Callable[[str, List[Dict[str, str]], float], Awaitable[Optional[Dict[str, Any]]]]


def parse_model_spec(spec: str) -> Tuple[str, str]:
    """Parse a model spec into (provider, model_name).

    Examples:
        "openai:gpt-4.1-mini" -> ("openai", "gpt-4.1-mini")
        "google/gemini-3-pro-preview" -> ("openrouter", "google/gemini-3-pro-preview")
    """
    spec = spec.strip()
    if ":" in spec:
        provider, model_name = spec.split(":", 1)
        provider = provider.strip().lower() or "openrouter"
        model_name = model_name.strip()
        return provider, model_name

    # No explicit provider: default to OpenRouter-style model identifier,
    # e.g. "google/gemini-3-pro-preview".
    provider = "openrouter"
    model_name = spec
    return provider, model_name
    
async def _query_openrouter(
    model_name: str,
    messages: List[Dict[str, str]],
    timeout: float,
) -> Optional[Dict[str, Any]]:
    """Query a model via OpenRouter using the unified client surface."""
    if not config.OPENROUTER_API_KEY or not config.OPENROUTER_API_URL:
        raise ConfigError("Missing OPENROUTER_API_KEY or OPENROUTER_API_URL")

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            config.OPENROUTER_API_URL,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        return {
            "content": message.get("content"),
            "reasoning_details": message.get("reasoning_details"),
        }


async def _query_openai_like(
    api_key: Optional[str],
    api_base: Optional[str],
    model_name: str,
    messages: List[Dict[str, str]],
    timeout: float,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Generic OpenAI-compatible chat completions client.

    Used for providers that expose a /chat/completions style API (OpenAI,
    Grok, GLM, Qwen, etc.) with minor header differences.
    """
    if not api_key or not api_base:
        raise ConfigError("Missing API key or base URL for OpenAI-compatible provider")

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    payload = {
        "model": model_name,
        "messages": messages,
    }

    url = api_base.rstrip("/") + "/chat/completions"

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        return {
            "content": message.get("content"),
            "reasoning_details": message.get("reasoning_details"),
        }


async def _query_openai(model_name: str, messages: List[Dict[str, str]], timeout: float) -> Optional[Dict[str, Any]]:
    return await _query_openai_like(
        config.OPENAI_API_KEY,
        config.OPENAI_API_BASE,
        model_name,
        messages,
        timeout,
    )


async def _query_grok(model_name: str, messages: List[Dict[str, str]], timeout: float) -> Optional[Dict[str, Any]]:
    # Grok currently exposes an OpenAI-compatible API surface.
    return await _query_openai_like(
        config.GROK_API_KEY,
        config.GROK_API_BASE,
        model_name,
        messages,
        timeout,
    )


async def _query_glm(model_name: str, messages: List[Dict[str, str]], timeout: float) -> Optional[Dict[str, Any]]:
    # Many GLM deployments expose OpenAI-compatible endpoints; the exact base URL
    # is configured via GLM_API_BASE.
    return await _query_openai_like(
        config.GLM_API_KEY,
        config.GLM_API_BASE,
        model_name,
        messages,
        timeout,
    )


async def _query_qwen(model_name: str, messages: List[Dict[str, str]], timeout: float) -> Optional[Dict[str, Any]]:
    # Qwen also provides an OpenAI-compatible mode; the exact base URL should be
    # configured via QWEN_API_BASE.
    return await _query_openai_like(
        config.QWEN_API_KEY,
        config.QWEN_API_BASE,
        model_name,
        messages,
        timeout,
    )


def _to_gemini_contents(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-style messages into Gemini contents.

    System messages are inlined into the first user turn, and
    consecutive turns with the same logical role are merged to avoid
    producing long runs of identical roles.
    """
    contents: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        text = msg.get("content", "")

        if role == "assistant":
            g_role = "model"
        else:
            # Inline any system messages into the user stream with a
            # clear prefix so that Gemini can see the distinction.
            if role == "system":
                text = "[SYSTEM]\n" + text
            g_role = "user"

        if contents and contents[-1]["role"] == g_role:
            # Merge consecutive turns for the same role.
            contents[-1]["parts"][0]["text"] += "\n\n" + text
        else:
            contents.append(
                {
                    "role": g_role,
                    "parts": [{"text": text}],
                }
            )

    return contents


async def _query_gemini(model_name: str, messages: List[Dict[str, str]], timeout: float) -> Optional[Dict[str, Any]]:
    if not config.GEMINI_API_KEY or not config.GEMINI_API_BASE:
        raise ConfigError("Missing GEMINI_API_KEY or GEMINI_API_BASE")

    url = f"{config.GEMINI_API_BASE.rstrip('/')}/models/{model_name}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": config.GEMINI_API_KEY,
    }
    payload: Dict[str, Any] = {
        "contents": _to_gemini_contents(messages),
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    # Gemini responses contain candidates; we concatenate all text parts.
    text_chunks: List[str] = []
    for candidate in data.get("candidates", []):
        content = candidate.get("content") or {}
        for part in content.get("parts", []):
            piece = part.get("text")
            if isinstance(piece, str):
                text_chunks.append(piece)

    content = "\n".join(text_chunks).strip()
    return {"content": content, "reasoning_details": None}


def _to_anthropic_messages(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-style messages into Anthropic messages.

    System messages are inlined into the user stream and consecutive
    turns with the same role are merged so that the sequence respects
    Anthropic's expectations (alternating user/assistant where
    possible) without adjacent duplicate roles.
    """
    converted: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        text = msg.get("content", "")

        if role == "system":
            # Anthropic does not have a separate system role; inline it.
            text = "[SYSTEM]\n" + text
            role = "user"

        if role not in ("user", "assistant"):
            role = "user"

        if converted and converted[-1]["role"] == role:
            converted[-1]["content"] += "\n\n" + text
        else:
            converted.append(
                {
                    "role": role,
                    "content": text,
                }
            )

    return converted


async def _query_anthropic(model_name: str, messages: List[Dict[str, str]], timeout: float) -> Optional[Dict[str, Any]]:
    if not config.ANTHROPIC_API_KEY or not config.ANTHROPIC_API_BASE or not config.ANTHROPIC_API_VERSION:
        raise ConfigError(
            "Missing Anthropic configuration (ANTHROPIC_API_KEY, ANTHROPIC_API_BASE, or ANTHROPIC_API_VERSION)"
        )

    url = f"{config.ANTHROPIC_API_BASE.rstrip('/')}/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": config.ANTHROPIC_API_KEY,
        # Version pinned for compatibility; can be overridden via ANTHROPIC_API_VERSION.
        "anthropic-version": config.ANTHROPIC_API_VERSION,
    }
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": _to_anthropic_messages(messages),
        "max_tokens": config.ANTHROPIC_DEFAULT_MAX_TOKENS,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    text_chunks: List[str] = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            value = block.get("text")
            if isinstance(value, str):
                text_chunks.append(value)

    content = "\n".join(text_chunks).strip()
    return {"content": content, "reasoning_details": None}


PROVIDERS: Dict[str, ProviderFn] = {
    "openrouter": _query_openrouter,
    "openai": _query_openai,
    "gemini": _query_gemini,
    "anthropic": _query_anthropic,
    "glm": _query_glm,
    "qwen": _query_qwen,
    "grok": _query_grok,
}


async def query_model(
    model_spec: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0,
) -> Optional[Dict[str, Any]]:
    """Query a model via the configured provider layer.

    Returns:
        A dict with at least a ``content`` field, or None on error.
    """
    provider, model_name = parse_model_spec(model_spec)
    provider_fn = PROVIDERS.get(provider)

    if provider_fn is None:
        # Treat unknown providers as configuration errors rather than
        # silently falling back to OpenRouter. This avoids confusing
        # logs like provider 'google' hitting the OpenRouter endpoint
        # and surfacing HTTP 400 errors at runtime.
        known_providers = ", ".join(sorted(PROVIDERS.keys()))
        print(
            f"Config error for model spec '{model_spec}': unknown provider "
            f"'{provider}'. Expected one of: {known_providers}."
        )
        return None

    try:
        return await provider_fn(model_name, messages, timeout)
    except ConfigError as e:
        # Configuration issues should be surfaced clearly but treated as a soft failure.
        print(f"Config error for provider '{provider}' model '{model_name}': {e}")
        return None
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status == 401:
            print(
                f"Error querying provider '{provider}' model '{model_name}': "
                f"401 Unauthorized – check API key or credentials"
            )
        else:
            print(f"HTTP error querying provider '{provider}' model '{model_name}': {e}")
        return None
    except Exception as e:
        # Fail soft – the council orchestration is robust to individual model failures.
        print(f"Error querying provider '{provider}' model '{model_name}': {e}")
        return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Query multiple models concurrently via asyncio.gather."""
    tasks = [query_model(model, messages) for model in models]
    responses = await asyncio.gather(*tasks)
    return {model: response for model, response in zip(models, responses)}
