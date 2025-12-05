"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key (optional)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Council members - configuration driven via environment.
#
# LLM_COUNCIL_MODELS: comma-separated list of model specs:
#   e.g. "openai:gpt-4.1-mini,qwen:qwen3-coder:free,grok:x-ai/grok-4.1-fast:free"
_raw_council_models = os.getenv("LLM_COUNCIL_MODELS", "")
if _raw_council_models:
    COUNCIL_MODELS = [m.strip() for m in _raw_council_models.split(",") if m.strip()]
else:
    # No hard-coded model IDs in code – if not configured, the council is empty
    # and the application will surface an error at runtime, prompting the user
    # to configure models explicitly in .env.
    COUNCIL_MODELS = []

# Chairman model - synthesizes final response
# LLM_CHAIRMAN_MODEL can override the default; otherwise we use the first
# council model as chairman.
_explicit_chairman = os.getenv("LLM_CHAIRMAN_MODEL")
if _explicit_chairman:
    CHAIRMAN_MODEL = _explicit_chairman
elif COUNCIL_MODELS:
    CHAIRMAN_MODEL = COUNCIL_MODELS[0]
else:
    CHAIRMAN_MODEL = None

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Root directory for llm-council data (default: ~/.llm-council)
LLM_COUNCIL_HOME = os.getenv(
    "LLM_COUNCIL_HOME",
    os.path.expanduser("~/.llm-council"),
)

# Data directory for conversation storage (default: <LLM_COUNCIL_HOME>/conversations)
DATA_DIR = os.getenv(
    "LLM_COUNCIL_DATA_DIR",
    os.path.join(LLM_COUNCIL_HOME, "conversations"),
)

# --------
# Direct provider configuration

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

# Google Gemini (direct REST API; used with /models/{model}:generateContent)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_BASE = os.getenv(
    "GEMINI_API_BASE",
    "https://generativelanguage.googleapis.com/v1beta",
)

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_API_BASE = os.getenv("ANTHROPIC_API_BASE", "https://openrouter.ai/api/v1")
# Anthropic API version is driven purely by environment configuration.
ANTHROPIC_API_VERSION = os.getenv("ANTHROPIC_API_VERSION")
ANTHROPIC_DEFAULT_MAX_TOKENS = int(os.getenv("ANTHROPIC_DEFAULT_MAX_TOKENS", "1024"))

# xAI Grok
GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_API_BASE = os.getenv("GROK_API_BASE", "https://openrouter.ai/api/v1")

# GLM (OpenAI-compatible endpoint; base URL must be configured by user)
GLM_API_KEY = os.getenv("GLM_API_KEY")
GLM_API_BASE = os.getenv("GLM_API_BASE")

# Qwen (OpenAI-compatible endpoint; defaults to OpenRouter; override for direct Qwen API)
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_API_BASE = os.getenv("QWEN_API_BASE", "https://openrouter.ai/api/v1")

# Model to use for generating conversation titles. If not explicitly set,
# reuse the chairman model to avoid hard-coding a specific provider/model
# combination in code.
TITLE_MODEL = os.getenv("TITLE_MODEL") or CHAIRMAN_MODEL
