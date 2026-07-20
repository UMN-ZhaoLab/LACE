"""Checkpoint utilities for artifact-based persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.artifact_store import run_dir, thin_state
from src.config import LACEConfig
from src.file_utils import register_safe_zone, write_text
from src.state_types import CheckpointPayload, WorkflowState


@dataclass
class Checkpoint:
    stage: str
    timestamp: str
    payload: CheckpointPayload
    thin_state: dict[str, Any] | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def capture_checkpoint(state: Any, stage: str) -> Checkpoint:
    """Capture a lightweight checkpoint from state without writing to disk."""
    payload = CheckpointPayload(
        stage=stage,
        op_index=getattr(state, "op_index", 0),
        hdl_index=getattr(state, "hdl_index", 0),
        cpu_dir=getattr(state, "cpu_dir", ""),
        cpu_summary_path=getattr(state, "cpu_summary_path", ""),
        ops_count=len(getattr(state, "ops", []) or []),
        hdl_tasks_count=len(getattr(state, "hdl_tasks", []) or []),
        notes=list(getattr(state, "notes", []) or []),
        needs_review=getattr(state, "needs_review", False),
    )
    thin = thin_state(state) if hasattr(state, "model_dump") else dict(state)
    return Checkpoint(stage=stage, timestamp=_utc_now(), payload=payload, thin_state=thin)


def persist_failure_bundle(
    state: Any,
    stage: str,
    error: str,
    checkpoint: Checkpoint | None = None,
    evidence_dir: str | None = None,
) -> str:
    """Persist a failure bundle if capture mode is enabled."""
    if LACEConfig.CAPTURE_MODE.lower() == "off":
        return ""
    if evidence_dir is None:
        evidence_dir = LACEConfig.ARTIFACT_DIR
    register_safe_zone(evidence_dir)
    run_id = getattr(state, "run_id", "") or build_run_id()
    # Write into the run directory if it exists, otherwise fallback to evidence_dir
    dest_dir = run_dir(run_id) if run_dir(run_id).exists() else Path(evidence_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"failure_bundle_{run_id}.json"
    payload: dict[str, Any] = {
        "run_id": run_id,
        "stage": stage,
        "error": error,
        "timestamp": _utc_now(),
        "checkpoint": checkpoint.payload.model_dump() if checkpoint else None,
        "thin_state": checkpoint.thin_state if checkpoint else None,
    }
    trace_map = getattr(state, "trace_map", None)
    if trace_map and (trace_map.ops or trace_map.hdl_tasks):
        payload["trace_map_ref"] = "artifacts/trace_map.json"
    write_text(path, json.dumps(payload, ensure_ascii=True, indent=2), atomic=True)
    return str(path)


def load_checkpoint(path: str) -> WorkflowState:
    """Load a full WorkflowState from a failure bundle path."""
    from src.artifact_store import hydrate_state
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    thin = data.get("thin_state")
    if not thin:
        raise ValueError(f"No thin_state found in checkpoint: {path}")
    return WorkflowState(**hydrate_state(thin))
