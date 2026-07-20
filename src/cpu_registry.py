from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

from src.config import LACEConfig


@dataclass(frozen=True)
class CpuConfig:
    name: str
    cpu_dir: str
    top_file: str
    sv_include_dir: str
    verilator_std: str | None = None
    verilator_waive_flags: List[str] = field(default_factory=lambda: ["--Wno-MULTITOP"])


def load_cpu_registry(path: str | None = None) -> dict[str, dict[str, str]]:
    config_path = Path(path or LACEConfig.CPU_CONFIG_PATH)
    if not config_path.exists():
        raise FileNotFoundError(f"CPU config not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "cpus" not in data:
        raise ValueError(f"CPU config must contain a top-level 'cpus' key: {config_path}")
    registry = data["cpus"]
    if not isinstance(registry, dict):
        raise ValueError(f"CPU config 'cpus' must be a mapping: {config_path}")
    return registry


def list_cpu_choices(path: str | None = None) -> list[str]:
    registry = load_cpu_registry(path)
    return sorted(registry.keys())


def resolve_cpu(cpu_name: str, path: str | None = None) -> CpuConfig:
    registry = load_cpu_registry(path)
    if cpu_name not in registry:
        choices = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown cpu_name '{cpu_name}'. Choices: {choices}")
    cfg = registry[cpu_name]
    if not isinstance(cfg, dict):
        raise ValueError(f"CPU config for {cpu_name} must be a mapping")

    required = {"cpu_dir", "top_file", "sv_include_dir"}
    expected = required | {"verilator_std", "verilator_waive_flags"}
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f"CPU config for {cpu_name} missing keys: {', '.join(sorted(missing))}")

    extra = cfg.keys() - expected
    if extra:
        warnings.warn(
            f"CPU config for {cpu_name} contains unexpected keys: {', '.join(sorted(extra))}",
            stacklevel=2,
        )

    cpu_dir = cfg["cpu_dir"]
    top_file = cfg["top_file"]

    dir_path = Path(cpu_dir)
    if not dir_path.exists():
        raise ValueError(f"CPU directory does not exist: {cpu_dir}")
    file_path = dir_path / top_file
    if not file_path.exists():
        raise ValueError(f"CPU top file does not exist: {file_path}")

    return CpuConfig(
        name=cpu_name,
        cpu_dir=cpu_dir,
        top_file=top_file,
        sv_include_dir=cfg["sv_include_dir"],
        verilator_std=cfg.get("verilator_std"),
        verilator_waive_flags=cfg.get("verilator_waive_flags", ["--Wno-MULTITOP"]),
    )
