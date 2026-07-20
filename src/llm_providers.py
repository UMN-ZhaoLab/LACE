"""LLM provider configuration resolver for multi-provider support.

Supported providers:
- OpenAI       (MODEL_NAME=openai:gpt-4.1, OPENAI_API_KEY)
- Moonshot     (MODEL_NAME=moonshot:kimi-k2, MOONSHOT_API_KEY)
- Azure OpenAI (MODEL_NAME=azure_openai:gpt-4, AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_KEY)
- Local / vLLM (MODEL_NAME=openai:Qwen2.5-Coder, OPENAI_API_BASE=http://localhost:8000/v1)
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Resolved LLM provider configuration."""

    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    api_version: str | None = None
    extra_kwargs: dict[str, str] | None = None


def _detect_provider(model_name: str) -> tuple[str, str]:
    """Detect provider prefix and extract model name.

    Supports formats like:
        "openai:gpt-4.1"      → ("openai", "gpt-4.1")
        "moonshot:kimi-k2"    → ("moonshot", "kimi-k2")
        "azure_openai:gpt-4"  → ("azure_openai", "gpt-4")
    """
    if ":" in model_name:
        provider, model = model_name.split(":", 1)
        return provider.strip().lower(), model.strip()
    return "openai", model_name.strip()


def resolve_provider_config(
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> ProviderConfig:
    """Resolve provider configuration from environment and arguments.

    Priority:
        1. Explicit arguments
        2. Environment variables
        3. Defaults
    """
    model_name = model_name or os.getenv("MODEL_NAME", "openai:gpt-4.1")
    provider, model = _detect_provider(model_name)

    if provider == "openai":
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        resolved_base = base_url or os.getenv("OPENAI_API_BASE")
        return ProviderConfig(
            provider="openai",
            model=model,
            api_key=resolved_key,
            base_url=resolved_base,
        )

    if provider == "moonshot":
        resolved_key = api_key or os.getenv("MOONSHOT_API_KEY") or os.getenv("OPENAI_API_KEY")
        resolved_base = base_url or os.getenv("MOONSHOT_API_BASE", "https://api.moonshot.cn/v1")
        return ProviderConfig(
            provider="openai",  # Moonshot is OpenAI-compatible
            model=model,
            api_key=resolved_key,
            base_url=resolved_base,
        )

    if provider == "azure_openai":
        resolved_key = api_key or os.getenv("AZURE_OPENAI_KEY")
        resolved_base = base_url or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
        return ProviderConfig(
            provider="azure_openai",
            model=model,
            api_key=resolved_key,
            base_url=resolved_base,
            api_version=api_version,
        )

    # Fallback: treat as OpenAI-compatible local provider
    resolved_key = api_key or os.getenv("OPENAI_API_KEY", "")
    resolved_base = base_url or os.getenv("OPENAI_API_BASE")
    return ProviderConfig(
        provider="openai",
        model=model,
        api_key=resolved_key or None,
        base_url=resolved_base,
    )
