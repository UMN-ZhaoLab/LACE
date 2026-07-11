"""LLM utilities with multi-provider support."""

from __future__ import annotations

import copy
import json
import re
from typing import Any, cast

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from src.config import LACEConfig
from src.llm_providers import ProviderConfig, resolve_provider_config


class _OpenAICompatibleStructured:
    """Fallback structured-output runnable for providers without response_format support."""

    def __init__(self, model: BaseChatModel, schema: Any) -> None:
        self._model = model
        self._schema = schema

    def invoke(self, messages: list[Any], **kwargs: Any) -> Any:
        """Inject JSON schema hint, invoke plain model, parse and validate response."""
        # First, try to get structured output directly (for mock LLMs)
        raw = self._model.invoke(messages, **kwargs)
        if isinstance(raw, self._schema):
            return raw

        # Inject JSON schema hint for providers that don't support response_format
        msgs = list(copy.deepcopy(messages))
        schema_dict: dict[str, Any] = (
            self._schema.model_json_schema()
            if hasattr(self._schema, "model_json_schema")
            else {}
        )
        json_hint = (
            "\n\nYou MUST respond with a single valid json object matching the required schema. "
            "Do NOT wrap the response in markdown code fences.\n"
            f"Schema: {json.dumps(schema_dict, ensure_ascii=False)}"
        )
        if msgs:
            last = msgs[-1]
            if hasattr(last, "content"):
                cls = type(last)
                msgs[-1] = cls(content=last.content + json_hint)

        raw = self._model.invoke(msgs, **kwargs)
        content = raw.content if hasattr(raw, "content") else str(raw)
        content = content.strip()

        # Strip markdown fences if the model disobeyed the instruction
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
            content = re.sub(r"\s*```$", "", content)
        content = content.strip()

        data = json.loads(content)
        return self._schema(**data)


def _needs_structured_fallback(cfg: ProviderConfig) -> bool:
    """Detect providers that do not support OpenAI's response_format / structured output.

    Native ``response_format`` is only guaranteed on api.openai.com. Most
    third-party OpenAI-compatible gateways (DeepSeek, DashScope, MIMO, iFlyTek
    Spark/Maas, local vLLM) reject it with HTTP 400. We use a blacklist of
    known-unsupported hosts; when in doubt for an unknown host, prefer the
    prompt-injection fallback rather than risk a hard 400 on every structured
    call.
    """
    if cfg.provider != "openai" or not cfg.base_url:
        return False
    known_unsupported = (
        "deepseek",
        "dashscope",
        "localhost",
        "mimo",
        "xf-yun.com",       # iFlyTek Spark Maas coding gateway
        "maas-coding",      # iFlyTek Maas (alternate host)
        "aliyuncs.com",     # Alibaba DashScope alternate
        "baidubce.com",     # Baidu Qianfan OpenAI-compat
        "127.0.0.1",
    )
    return any(host in cfg.base_url for host in known_unsupported)


def get_chat_model(timeout: float | None = None) -> BaseChatModel:
    """Initialize and return the chat model with resolved provider config.

    Provider resolution order:
        1. MODEL_NAME environment variable (e.g. "openai:gpt-4.1", "moonshot:kimi-k2")
        2. OPENAI_API_KEY / MOONSHOT_API_KEY / AZURE_OPENAI_KEY
        3. OPENAI_API_BASE for local/vLLM endpoints

    Examples:
        # OpenAI
        MODEL_NAME=openai:gpt-4.1 OPENAI_API_KEY=sk-...

        # Moonshot (OpenAI-compatible)
        MODEL_NAME=moonshot:kimi-k2 MOONSHOT_API_KEY=sk-...

        # Local vLLM / llama.cpp
        MODEL_NAME=openai:Qwen2.5-Coder OPENAI_API_BASE=http://localhost:8000/v1
    """
    cfg = resolve_provider_config()
    kwargs: dict[str, Any] = {}
    effective_timeout = timeout if timeout is not None else LACEConfig.LLM_TIMEOUT
    if effective_timeout is not None:
        kwargs["timeout"] = effective_timeout

    if cfg.provider == "azure_openai":
        kwargs["azure_endpoint"] = cfg.base_url
        kwargs["api_key"] = cfg.api_key
        kwargs["api_version"] = cfg.api_version
        model: BaseChatModel = cast(
            BaseChatModel,
            cast(object, init_chat_model(cfg.model, model_provider="azure_openai", **kwargs)),
        )
        return model

    # OpenAI-compatible providers (OpenAI, Moonshot, local)
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key

    model = cast(
        BaseChatModel,
        cast(object, init_chat_model(cfg.model, model_provider="openai", **kwargs)),
    )

    return model


def get_structured_runnable(model: BaseChatModel, schema: Any) -> Any:
    """Return a structured-output runnable, with fallback for unsupported providers.

    Some OpenAI-compatible endpoints (e.g. DeepSeek) do not support the
    ``response_format`` parameter. For those providers we inject the JSON schema
    into the prompt and parse the plain-text response manually.
    """
    # Detect mock mode: mock models have a side_effect set on with_structured_output
    if hasattr(model, "with_structured_output") and hasattr(model.with_structured_output, "side_effect"):
        return model.with_structured_output(schema)
    cfg = resolve_provider_config()
    if _needs_structured_fallback(cfg):
        return _OpenAICompatibleStructured(model, schema)
    return model.with_structured_output(schema)
