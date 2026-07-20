"""Create isolated, lightweight riscv-formal workspaces for pipeline runs."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from src.config import LACEConfig


SANDBOX_MARKER = ".lace-formal-sandbox.json"


def formal_sandbox_path(run_id: str, workspace_dir: str = "") -> Path:
    """Return the per-run formal workspace path.

    A CPU workspace normally lives at ``artifacts/runs/<run>/workspace``. Keeping
    formal beside it makes the isolation independent of the process cwd.
    """
    if workspace_dir:
        return Path(workspace_dir).resolve().parent / "formal"
    safe_run_id = run_id.strip() or "adhoc"
    return Path(LACEConfig.ARTIFACT_DIR).resolve() / "runs" / safe_run_id / "formal"


def is_formal_sandbox(path: str | Path) -> bool:
    """Return whether *path* is a sandbox created by this module."""
    return (Path(path) / SANDBOX_MARKER).is_file()


def _looks_generated_directory(path: Path, sby_output_names: set[str]) -> bool:
    if path.name in {".git", "checks", "__pycache__"}:
        return True
    if path.name.startswith("cexdata-") or path.name in sby_output_names:
        return True
    return (path / "status").is_file() or any(path.glob("engine_*"))


def _link_core_overlay(source_core: Path, target_core: Path) -> None:
    """Mirror a core with symlinked inputs and a private checks.cfg."""
    sby_output_names = {path.stem for path in source_core.glob("*.sby")}
    for current, directory_names, file_names in os.walk(source_core):
        current_path = Path(current)
        relative = current_path.relative_to(source_core)
        target_directory = target_core / relative
        target_directory.mkdir(parents=True, exist_ok=True)

        directory_names[:] = [
            name
            for name in directory_names
            if not _looks_generated_directory(current_path / name, sby_output_names)
        ]
        for name in file_names:
            source_file = current_path / name
            target_file = target_directory / name
            if relative == Path(".") and name == "checks.cfg":
                shutil.copy2(source_file, target_file)
            else:
                target_file.symlink_to(source_file.resolve())


def prepare_riscv_formal_sandbox(
    *,
    run_id: str,
    cpu_name: str,
    workspace_dir: str = "",
    source_dir: str | Path | None = None,
) -> Path:
    """Prepare and return an isolated riscv-formal tree for one pipeline run.

    Immutable framework files are symlinked, while ``insns`` and the selected
    core's ``checks.cfg`` are copied because LACE modifies them. Generated
    ``checks`` directories are deliberately excluded.
    """
    source = Path(source_dir or LACEConfig.RISCV_FORMAL_DIR).resolve()
    source_checks = source / "checks"
    source_insns = source / "insns"
    source_core = source / "cores" / cpu_name
    required = [source_checks / "genchecks.py", source_insns, source_core / "checks.cfg"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "riscv-formal source is incomplete: " + ", ".join(missing)
        )

    destination = formal_sandbox_path(run_id, workspace_dir)
    marker_data = {"source": str(source), "cpu_name": cpu_name}
    marker = destination / SANDBOX_MARKER
    if marker.is_file():
        try:
            if json.loads(marker.read_text(encoding="utf-8")) == marker_data:
                return destination
        except (OSError, json.JSONDecodeError):
            pass

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        (temporary / "checks").symlink_to(source_checks)
        shutil.copytree(source_insns, temporary / "insns")
        _link_core_overlay(source_core, temporary / "cores" / cpu_name)
        (temporary / SANDBOX_MARKER).write_text(
            json.dumps(marker_data, sort_keys=True), encoding="utf-8"
        )

        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() or destination.is_file():
                destination.unlink()
            else:
                shutil.rmtree(destination)
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination
