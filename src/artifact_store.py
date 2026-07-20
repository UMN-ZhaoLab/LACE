"""Artifact storage with transparent state thinning and hydration.

Large fields are extracted from WorkflowState into external artifact files.
The thin state only keeps references (relative paths) to these artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import LACEConfig

# Fields that get extracted into external artifact files
ARTIFACT_FIELDS = {
    "cpu_summary",
    "cpu_module_index",
    "ops",
    "arithmetic_ops",
    "hdl_tasks",
    "interface_code",
    "arithmetic_code",
    "candidate_modules",
    "candidate_notes",
    "notes",
    "trace_map",
    "last_error",
}


def run_dir(run_id: str) -> Path:
    return Path(LACEConfig.ARTIFACT_DIR) / "runs" / run_id


def artifacts_dir(run_id: str) -> Path:
    d = run_dir(run_id) / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_dir(run_id: str) -> Path:
    d = run_dir(run_id) / "state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoints_dir(run_id: str) -> Path:
    d = state_dir(run_id) / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def steps_dir(run_id: str) -> Path:
    d = run_dir(run_id) / "steps"
    d.mkdir(parents=True, exist_ok=True)
    return d


def reports_dir(run_id: str) -> Path:
    d = run_dir(run_id) / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def workspace_dir(run_id: str) -> Path:
    d = run_dir(run_id) / "workspace"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_artifact(run_id: str, kind: str, data: Any) -> str:
    """Save data as artifact, return relative path from run_dir."""
    d = artifacts_dir(run_id)
    if isinstance(data, str):
        path = d / f"{kind}.md"
        path.write_text(data, encoding="utf-8")
    else:
        path = d / f"{kind}.json"
        path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
    return str(path.relative_to(run_dir(run_id)))


def load_artifact(run_id: str, ref: str) -> Any:
    """Load artifact by relative reference path."""
    if not ref:
        return None
    path = run_dir(run_id) / ref
    if not path.exists():
        return None
    if path.suffix == ".md":
        return path.read_text(encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def thin_state(state: Any) -> dict[str, Any]:
    """Extract large fields into artifacts, return thin state dict with _artifact_refs."""
    data = state.model_dump() if hasattr(state, "model_dump") else dict(state)
    refs: dict[str, str] = {}
    run_id = data.get("run_id", "")
    if not run_id:
        return data
    for field in ARTIFACT_FIELDS:
        if field in data and data[field]:
            refs[field] = save_artifact(run_id, field, data[field])
            # Replace with empty placeholder of correct type
            val = data[field]
            if isinstance(val, list):
                data[field] = []
            elif isinstance(val, str):
                data[field] = ""
            elif isinstance(val, dict):
                data[field] = {}
    data["_artifact_refs"] = refs
    return data


def hydrate_state(thin_data: dict[str, Any]) -> dict[str, Any]:
    """Restore large fields from artifacts into thin state dict."""
    run_id = thin_data.get("run_id", "")
    refs = thin_data.pop("_artifact_refs", {})
    for field, ref in refs.items():
        if ref:
            val = load_artifact(run_id, ref)
            if val is not None:
                thin_data[field] = val
    return thin_data


def save_thin_state(run_id: str, name: str, state: Any) -> str:
    """Save thin state to state/<name>.json."""
    d = state_dir(run_id)
    thin = thin_state(state)
    path = d / f"{name}.json"
    path.write_text(json.dumps(thin, ensure_ascii=True, indent=2), encoding="utf-8")
    return str(path)


def load_thin_state(run_id: str, name: str) -> dict[str, Any] | None:
    """Load thin state from state/<name>.json."""
    path = state_dir(run_id) / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(run_id: str, step_index: int, step_name: str, state: Any) -> str:
    """Save thin checkpoint to state/checkpoints/<idx>_<name>.json."""
    d = checkpoints_dir(run_id)
    thin = thin_state(state)
    # If interface_code was thinned to empty but the workspace file has content,
    # save it as an artifact so hydration can restore it later.
    if not thin.get("interface_code") and thin.get("workspace_dir") and thin.get("cpu_top_file"):
        try:
            from pathlib import Path
            ws_code = (Path(thin["workspace_dir"]) / thin["cpu_top_file"]).read_text(encoding="utf-8")
            if ws_code:
                thin["_artifact_refs"] = thin.get("_artifact_refs", {})
                thin["_artifact_refs"]["interface_code"] = save_artifact(run_id, "interface_code", ws_code)
        except Exception:
            pass
    path = d / f"{step_index:03d}_{step_name}.json"
    path.write_text(json.dumps(thin, ensure_ascii=True, indent=2), encoding="utf-8")
    return str(path)


def load_checkpoint(run_id: str, step_index: int, step_name: str) -> dict[str, Any] | None:
    """Load thin checkpoint from state/checkpoints/."""
    path = checkpoints_dir(run_id) / f"{step_index:03d}_{step_name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_step_output(
    run_id: str,
    step_index: int,
    step_name: str,
    status: str,
    error: str = "",
) -> None:
    """Save step execution metadata to steps/<idx>_<name>/output.json."""
    d = steps_dir(run_id) / f"{step_index:03d}_{step_name}"
    d.mkdir(parents=True, exist_ok=True)
    output = {
        "step_index": step_index,
        "step_name": step_name,
        "status": status,
        "error": error,
    }
    (d / "output.json").write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")


def save_llm_artifacts(
    run_id: str,
    step_index: int,
    step_name: str,
    prompt: dict[str, str],
    raw_response: Any,
    metrics: dict[str, Any] | None = None,
) -> None:
    """Save LLM prompt, raw response, and metrics for reproducibility."""
    d = steps_dir(run_id) / f"{step_index:03d}_{step_name}"
    d.mkdir(parents=True, exist_ok=True)

    # Prompt
    prompt_md = f"# System\n\n{prompt.get('system', '')}\n\n# Human\n\n{prompt.get('human', '')}"
    if prompt.get("memory"):
        prompt_md += f"\n\n# Memory\n\n{prompt['memory']}"
    (d / "prompt.md").write_text(prompt_md, encoding="utf-8")

    # Raw response
    raw = raw_response if isinstance(raw_response, dict) else {"raw": str(raw_response)}
    (d / "llm_raw.json").write_text(json.dumps(raw, ensure_ascii=True, indent=2), encoding="utf-8")

    # Metrics
    if metrics:
        (d / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=True, indent=2), encoding="utf-8")
