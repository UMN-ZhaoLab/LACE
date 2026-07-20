"""CPU resolution node shared between graph and imperative workflow."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.artifact_store import workspace_dir
from src.checkpoint import build_run_id, capture_checkpoint, persist_failure_bundle
from src.config import LACEConfig
from src.cpu_registry import resolve_cpu
from src.file_utils import register_safe_zone
from src.state_types import WorkflowState


def resolve_cpu_state(state: WorkflowState) -> WorkflowState:
    """Resolve CPU configuration and update state."""
    if not state.run_id:
        state = state.model_copy(update={"run_id": build_run_id()})

    # Direct cpu_dir provided without cpu_name
    if not state.cpu_name:
        if state.cpu_dir:
            workspace_dir = _create_workspace(state.cpu_dir, state.run_id, "unknown")
            register_safe_zone(workspace_dir)
            return state.model_copy(
                update={
                    "workspace_dir": workspace_dir,
                    "cpu_summary_path": state.cpu_summary_path
                    or str(Path(LACEConfig.ARTIFACT_DIR) / "cpu_summary.json"),
                }
            )
        error = "cpu_name is required for CPU selection"
        checkpoint = capture_checkpoint(state, "cpu_select")
        persist_failure_bundle(state, "cpu_select", error, checkpoint)
        return state.model_copy(update={"needs_review": True, "last_error": error})

    try:
        cfg = resolve_cpu(state.cpu_name)
        register_safe_zone(cfg.cpu_dir)
        workspace_dir = _create_workspace(cfg.cpu_dir, state.run_id, state.cpu_name)
        register_safe_zone(workspace_dir)
        return state.model_copy(
            update={
                "cpu_dir": cfg.cpu_dir,
                "cpu_top_file": cfg.top_file,
                "sv_include_dir": cfg.sv_include_dir,
                "verilator_std": cfg.verilator_std,
                "verilator_waive_flags": cfg.verilator_waive_flags,
                "cpu_summary_path": state.cpu_summary_path
                or str(Path(LACEConfig.ARTIFACT_DIR) / "cpu_summary.json"),
                "workspace_dir": workspace_dir,
            }
        )
    except Exception as exc:
        checkpoint = capture_checkpoint(state, "cpu_select")
        persist_failure_bundle(state, "cpu_select", str(exc), checkpoint)
        return state.model_copy(update={"needs_review": True, "last_error": str(exc)})


def _create_workspace(cpu_dir: str, run_id: str, cpu_name: str) -> str:
    """Create an isolated workspace by copying the CPU source directory.

    The workspace lives under artifacts/runs/<run_id>/workspace/.
    A base.json file records the original cpu_dir for diff generation.
    """
    ws = workspace_dir(run_id)
    if ws.exists():
        shutil.rmtree(str(ws))
    shutil.copytree(cpu_dir, str(ws))
    # Record base info for future overlay/diff support
    base_info = {
        "cpu_dir": str(Path(cpu_dir).resolve()),
        "cpu_name": cpu_name,
        "run_id": run_id,
    }
    (ws / "base.json").write_text(
        json.dumps(base_info, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    return str(ws)
