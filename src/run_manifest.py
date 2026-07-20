"""Run manifest management.

Each run has a manifest.json that records metadata, parent relationship,
resumption point, and any manual overrides.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.artifact_store import run_dir


def create_manifest(
    run_id: str,
    cpu_name: str,
    spec: str,
    parent_run_id: str | None = None,
    resume_from: str | None = None,
    overrides: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create and persist a run manifest."""
    manifest = {
        "run_id": run_id,
        "cpu_name": cpu_name,
        "spec": spec,
        "parent_run_id": parent_run_id,
        "resume_from": resume_from,
        "overrides": overrides or [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    d = run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    return manifest


def load_manifest(run_id: str) -> dict[str, Any] | None:
    """Load a run manifest if it exists."""
    path = run_dir(run_id) / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
