"""Which provider and model the agent talks to, and how to change it."""

import os
import re
import time
from dataclasses import dataclass

import httpx

FALLBACK_MODEL = "openai:gpt-4.1-mini"
_LIST_CACHE_SECONDS = 3600
_LIST_TIMEOUT = 15
_OPENAI_CHAT = re.compile(r"^(gpt-|chatgpt-|chat-latest$|o\d)")
_OPENAI_NOT_CHAT = re.compile(
    r"image|audio|realtime|transcribe|tts|search|instruct|deep-research|live|whisper"
)
_DATED_SNAPSHOT = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{4})$")


@dataclass(frozen=True)
class Provider:
    key_env: str
    default_model: str
    models_url: str


PROVIDERS = {
    "openai": Provider(
        key_env="OPENAI_API_KEY",
        default_model="gpt-4.1-mini",
        models_url="https://platform.openai.com/docs/models",
    ),
    "openrouter": Provider(
        key_env="OPENROUTER_API_KEY",
        default_model="openai/gpt-4.1-mini",
        models_url="https://openrouter.ai/models",
    ),
}


@dataclass(frozen=True)
class ModelChoice:
    provider: str
    model: str

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


def default_choice() -> ModelChoice:
    """The model from AGENT_MODEL, used until someone picks another."""
    raw = os.environ.get("AGENT_MODEL", "").strip() or FALLBACK_MODEL
    provider, sep, model = raw.partition(":")
    if not sep:
        return ModelChoice("openai", raw)
    return ModelChoice(provider.lower(), model)


def provider_default(provider: str) -> ModelChoice:
    """The model to use when someone switches to a provider without naming one."""
    current = default_choice()
    if current.provider == provider:
        return current
    return ModelChoice(provider, PROVIDERS[provider].default_model)


def parse_choice(text: str, current: ModelChoice) -> ModelChoice:
    """Read `/model` arguments.

    Accepts `default`, a provider (`openrouter`), a model on the current
    provider (`gpt-4.1`), or both (`openrouter:anthropic/claude-sonnet-4.5`
    or `openrouter anthropic/claude-sonnet-4.5`).
    """
    parts = text.split()
    if not parts:
        raise ValueError("Name a model or a provider.")
    if len(parts) > 2:
        raise ValueError("Use /model <model> or /model <provider> <model>.")

    first = parts[0]
    if len(parts) == 1 and first.lower() == "default":
        return default_choice()
    if len(parts) == 2:
        provider = _known_provider(first)
        return ModelChoice(provider, parts[1])

    head, sep, tail = first.partition(":")
    if sep and head.lower() in PROVIDERS:
        if not tail:
            raise ValueError(f"Name a model after {head}:.")
        return ModelChoice(head.lower(), tail)
    if first.lower() in PROVIDERS:
        return provider_default(first.lower())
    return ModelChoice(current.provider, first)


def missing_key(choice: ModelChoice) -> str | None:
    """The environment variable this choice needs and doesn't have, if any."""
    provider = PROVIDERS.get(choice.provider)
    if provider is None or os.environ.get(provider.key_env, "").strip():
        return None
    return provider.key_env


def describe_providers() -> str:
    lines = []
    for name, provider in PROVIDERS.items():
        ready = "ready" if os.environ.get(provider.key_env, "").strip() else f"needs {provider.key_env}"
        lines.append(f"• {name} ({ready}), models: {provider.models_url}")
    return "\n".join(lines)


def _known_provider(name: str) -> str:
    provider = name.lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {name}. Choose one of: {', '.join(PROVIDERS)}.")
    return provider


def parse_listing(text: str, current: ModelChoice) -> tuple[str, str]:
    """Read `/models` arguments: an optional provider, then words to search for."""
    parts = text.split()
    if parts and parts[0].lower() in PROVIDERS:
        return parts[0].lower(), " ".join(parts[1:])
    provider = current.provider if current.provider in PROVIDERS else next(iter(PROVIDERS))
    return provider, " ".join(parts)


def filter_models(models: list[str], search: str) -> list[str]:
    """Models whose id contains every word of the search."""
    words = search.lower().split()
    return [model for model in models if all(word in model.lower() for word in words)]


def openai_chat_models(data: list[dict]) -> list[str]:
    """Chat models from OpenAI's model list, newest first, without dated snapshots."""
    newest_first = sorted(data, key=lambda entry: -entry.get("created", 0))
    return [
        entry["id"]
        for entry in newest_first
        if _OPENAI_CHAT.match(entry["id"])
        and not _OPENAI_NOT_CHAT.search(entry["id"])
        and not _DATED_SNAPSHOT.search(entry["id"])
    ]


def openrouter_tool_models(data: list[dict]) -> list[str]:
    """Models from OpenRouter's list that can call tools, in OpenRouter's order (newest first)."""
    return [
        entry["id"]
        for entry in data
        if "tools" in (entry.get("supported_parameters") or [])
    ]


_list_cache: dict[str, tuple[float, list[str]]] = {}


async def list_models(provider: str) -> list[str]:
    """The models a provider offers right now. Raises httpx.HTTPError when it can't be reached."""
    cached = _list_cache.get(provider)
    if cached and time.monotonic() - cached[0] < _LIST_CACHE_SECONDS:
        return cached[1]
    async with httpx.AsyncClient(timeout=_LIST_TIMEOUT) as client:
        if provider == "openrouter":
            response = await client.get("https://openrouter.ai/api/v1/models")
            response.raise_for_status()
            models = openrouter_tool_models(response.json()["data"])
        else:
            key = os.environ.get(PROVIDERS[provider].key_env, "").strip()
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {key}"},
            )
            response.raise_for_status()
            models = openai_chat_models(response.json()["data"])
    _list_cache[provider] = (time.monotonic(), models)
    return models
