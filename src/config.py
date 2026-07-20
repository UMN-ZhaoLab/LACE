"""Configuration utilities."""

from __future__ import annotations

import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


def get_env(name: str, default: str | None = None, required: bool = False) -> str | None:
    """Get environment variable with optional default and required check."""
    value = os.getenv(name, default)
    if required and not value:
        raise ValueError(f"Missing environment variable: {name}")
    return value


class LACEConfig:
    """Configuration defaults for LACE."""

    MODEL_NAME: str = "openai:gpt-4.1"
    EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-4B"
    MEMORY_MAX_ITEMS: int = 5
    MEMORY_MAX_CHARS: int = 1500
    MEMORY_TTL_HOURS: int = 24
    MAX_TASK_RETRIES: int = 1
    MAX_VERILATOR_RETRIES: int = 2
    CONFIDENCE_THRESHOLD: float = 0.6
    MAX_STAGE_RETRIES: int = 2
    CAPTURE_MODE: str = "failure"
    ARTIFACT_DIR: str = "artifacts"
    CPU_CONFIG_PATH: str = "config/cpus.yaml"
    LLM_TIMEOUT: float = 300.0  # seconds; prevents indefinite hangs on slow API

    # riscv-formal configuration
    RISCV_FORMAL_DIR: str = "tools/riscv-formal"
    # Formal solving is allowed to run for a long time. These are deliberately
    # independent from LLM/Verilator budgets and may be overridden in .env.
    RISCV_FORMAL_TIMEOUT: int = int(
        get_env("RISCV_FORMAL_TIMEOUT", "3600") or "3600"
    )  # seconds per sby check
    RISCV_FORMAL_GENCHECKS_TIMEOUT: int = int(
        get_env("RISCV_FORMAL_GENCHECKS_TIMEOUT", "300") or "300"
    )
    RISCV_FORMAL_MAX_CHECKS: int = 10  # max checks to run per invocation
    # OSS CAD Suite ships Boolector and it is substantially faster than Z3
    # for the bounded bit-vector instruction checks used by riscv-formal.
    # Users can still select a different installed solver through the env.
    RISCV_FORMAL_SOLVER: str = get_env("RISCV_FORMAL_SOLVER", "boolector") or "boolector"
    MAX_FORMAL_RETRIES: int = int(
        get_env("MAX_FORMAL_RETRIES", "5") or "5"
    )
