"""Sequential node-by-node pipeline execution with snapshot persistence.

This CLI script is a thin wrapper around src.pipeline_runner.

Usage:
    # Full pipeline from scratch (mock LLM)
    python scripts/test_nodes_sequential.py --mock --spec "Add ROL" --cpu picorv32

    # Resume from a specific node (use previous snapshot as input)
    python scripts/test_nodes_sequential.py --mock --from op2hdl_planner --run-id 20250420-120000

    # Run with real LLM
    python scripts/test_nodes_sequential.py --spec "Add ROL" --cpu picorv32

    # List available nodes
    python scripts/test_nodes_sequential.py --list-nodes

    # Run from YAML config
    python scripts/test_nodes_sequential.py -c config/batch_test.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.artifact_store import load_thin_state, hydrate_state
from src.pipeline_runner import (
    NODE_NAMES,
    list_snapshots,
    run_graph_segment,
)
from src.prompts import spec2op


def _resolve_spec_from_prompts(spec_ref: str) -> str:
    """Resolve a spec reference to its content from src.prompts.spec2op.

    Raises:
        ValueError: if the reference does not exist or is not a string.
    """
    if not hasattr(spec2op, spec_ref):
        available = [
            k for k in dir(spec2op)
            if not k.startswith("_") and isinstance(getattr(spec2op, k, None), str)
        ]
        raise ValueError(
            f"YAML/CLI 'spec' must reference a string variable in src.prompts.spec2op. "
            f"Got {spec_ref!r}. Available: {available}"
        )

    value = getattr(spec2op, spec_ref)
    if not isinstance(value, str):
        raise ValueError(
            f"src.prompts.spec2op.{spec_ref} exists but is not a string "
            f"(type: {type(value).__name__})."
        )
    return value.strip()


def _load_specs_file(path: str) -> list[str]:
    """Load one spec per line from a text file."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def _load_config(path: str) -> dict[str, Any]:
    """Load pipeline configuration from a YAML file."""
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping")
    return data


def _print_summary(results: list[dict[str, Any]]) -> None:
    """Print a summary table of batch execution results."""
    total = len(results)
    passed = sum(1 for r in results if r["exit_code"] == 0 and not r.get("needs_review"))
    failed = total - passed

    print("\n" + "=" * 60)
    print(f"BATCH SUMMARY: {passed}/{total} passed, {failed} failed")
    print("=" * 60)
    for r in results:
        status = "PASS" if r["exit_code"] == 0 and not r.get("needs_review") else "FAIL"
        print(f"  [{status}] {r['run_id']} | spec={r['spec']!r} | cpu={r['cpu']}")
        if r.get("error"):
            print(f"         error: {r['error']}")
    print("=" * 60)


def _restore_latest_snapshot(run_id: str) -> dict[str, Any] | None:
    """Find the most recent persisted snapshot for *run_id* by node name.

    Graph-mode checkpoints are named by node (not by a stable positional
    index), so we walk NODE_NAMES newest-first and return the first hit.
    """
    from src.pipeline_runner import load_snapshot
    for node_name in reversed(NODE_NAMES):
        thin = load_snapshot(run_id, node_name)
        if thin is not None:
            return thin
    return None


def _run_single(
    spec: str,
    cpu_name: str,
    mock: bool = False,
    start_from: str | None = None,
    stop_at: str | None = None,
    run_id: str | None = None,
    parent_run_id: str | None = None,
    max_task_retries: int = 1,
    _runner: Any = run_graph_segment,
) -> tuple[int, str, WorkflowState | None]:
    """Run a single pipeline segment and print progress."""
    # When resuming from a parent run, reuse its run_id so snapshots stay
    # in the same directory. Otherwise generate a fresh timestamp.
    rid = run_id or parent_run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    print(f"Run ID: {rid}")
    print(f"Mode: {'mock' if mock else 'real LLM'}")

    try:
        final_state, log, rid = _runner(
            spec=spec,
            cpu_name=cpu_name,
            start_from=start_from,
            stop_at=stop_at,
            mock=mock,
            run_id=rid,
            parent_run_id=parent_run_id,
            max_task_retries=max_task_retries,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1, rid, None

    for entry in log:
        status = entry["status"]
        node = entry["step_name"]
        if status == "skipped":
            print(f"  SKIP  {node}")
        elif status == "ok":
            print(f"  OK    {node}")
        elif status == "error":
            print(f"  ERROR {node}: {entry['error']}")
        elif status == "halt":
            print(f"  HALT  {node}: {entry['error']}")
        elif status == "stopped":
            print(f"  STOP  {node}")

    if final_state.needs_review:
        print(f"  HALT: needs_review=True, last_error={final_state.last_error!r}")

    print(f"\nPipeline completed. Run directory: {Path('artifacts/runs') / rid}")
    return 0, rid, final_state


def run_batch(
    specs: list[str],
    cpu_name: str,
    mock: bool = False,
    start_from: str | None = None,
    stop_at: str | None = None,
    output_csv: str | None = None,
    parent_run_id: str | None = None,
) -> int:
    """Run pipeline for multiple specs sequentially."""
    results: list[dict[str, Any]] = []

    for i, spec in enumerate(specs, 1):
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + f"-{i:03d}"
        print(f"\n{'=' * 60}")
        print(f"BATCH [{i}/{len(specs)}] run_id={run_id}")
        print(f"  spec={spec!r}")
        print(f"  cpu={cpu_name}")
        if parent_run_id:
            print(f"  parent={parent_run_id}")
        print("=" * 60)

        exit_code, _, final_state = _run_single(
            spec=spec,
            cpu_name=cpu_name,
            mock=mock,
            start_from=start_from,
            stop_at=stop_at,
            run_id=run_id,
            parent_run_id=parent_run_id,
        )

        needs_review = final_state.needs_review if final_state else False
        last_error = final_state.last_error if final_state else ""

        result = {
            "run_id": run_id,
            "spec": spec,
            "cpu": cpu_name,
            "exit_code": exit_code,
            "needs_review": needs_review,
            "error": last_error,
        }
        results.append(result)

    _print_summary(results)

    if output_csv:
        import csv
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["run_id", "spec", "cpu", "exit_code", "needs_review", "error"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV report saved: {output_csv}")

    return 0 if all(r["exit_code"] == 0 and not r["needs_review"] for r in results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sequential node-by-node pipeline execution with snapshots."
    )
    parser.add_argument("-c", "--config", help="Path to YAML config file")
    parser.add_argument("--list-nodes", action="store_true", help="List available node names")
    parser.add_argument("--spec", help="Instruction specification")
    parser.add_argument("--cpu", default="picorv32", help="Target CPU name")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM")
    parser.add_argument("--from", dest="start_from", help="Start from this node name")
    parser.add_argument("--to", dest="stop_at", help="Stop at this node name")
    parser.add_argument("--run-id", help="Explicit run ID")
    parser.add_argument("--parent-run-id", help="Resume from parent run snapshots")
    parser.add_argument("--output-csv", help="Write batch results to CSV")
    args = parser.parse_args()

    if args.list_nodes:
        print("Available pipeline nodes:")
        for i, name in enumerate(NODE_NAMES, 1):
            print(f"  {i}. {name}")
        return 0

    # Load from YAML config if provided
    cfg: dict[str, Any] = {}
    if args.config:
        cfg = _load_config(args.config)

    cpu_name = cfg.get("cpu", args.cpu)
    mock = cfg.get("mock", args.mock)
    start_from = cfg.get("from", args.start_from)
    stop_at = cfg.get("to", args.stop_at)
    run_id = cfg.get("run_id", args.run_id)
    output_csv = cfg.get("output_csv", args.output_csv)
    parent_run_id = cfg.get("parent_run_id", args.parent_run_id)
    spec = cfg.get("spec", args.spec)
    # The compiled LangGraph is the only execution path now.
    runner = run_graph_segment

    # Resolve spec reference from prompts module.
    # Empty/None spec is allowed (falls back to snapshot restore or default).
    if spec:
        spec = _resolve_spec_from_prompts(spec)

    count = cfg.get("count")
    runs = cfg.get("runs")

    # Handle snapshot restoration logic
    if parent_run_id and not spec:
        thin = _restore_latest_snapshot(parent_run_id)
        if thin is None:
            print(f"ERROR: No checkpoint found in parent run {parent_run_id}")
            return 1
        from src.state_types import WorkflowState
        state = WorkflowState(**hydrate_state(thin))
        spec = state.spec
        cpu_name = state.cpu_name or cpu_name
        print(f"Restored from parent run {parent_run_id}: spec={spec!r}, cpu={cpu_name}")
    elif run_id and not spec:
        thin = _restore_latest_snapshot(run_id)
        if thin is None:
            print(f"ERROR: No checkpoint found for run_id={run_id}")
            return 1
        from src.state_types import WorkflowState
        state = WorkflowState(**hydrate_state(thin))
        spec = state.spec
        cpu_name = state.cpu_name or cpu_name
        print(f"Restored from run {run_id}: spec={spec!r}, cpu={cpu_name}")
    elif not spec:
        spec = "Add a rotate left instruction"

    # Mode 1: count-based batch
    if count and not runs:
        if not isinstance(count, int) or count < 1:
            print("ERROR: 'count' must be a positive integer")
            return 1
        results: list[dict[str, Any]] = []
        for i in range(1, count + 1):
            rid = run_id or (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + f"-{i:03d}")
            print(f"\n{'=' * 60}")
            print(f"COUNT [{i}/{count}] run_id={rid}")
            print(f"  spec={spec!r}")
            print(f"  cpu={cpu_name}")
            print("=" * 60)
            exit_code, _, final_state = _run_single(
                spec=spec,
                cpu_name=cpu_name,
                mock=mock,
                start_from=start_from,
                stop_at=stop_at,
                run_id=rid,
                parent_run_id=parent_run_id,
                _runner=runner,
            )
            results.append({
                "run_id": rid,
                "spec": spec,
                "cpu": cpu_name,
                "exit_code": exit_code,
                "needs_review": final_state.needs_review if final_state else False,
                "error": final_state.last_error if final_state else "",
            })
        _print_summary(results)
        if output_csv:
            import csv
            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["run_id", "spec", "cpu", "exit_code", "needs_review", "error"])
                writer.writeheader()
                writer.writerows(results)
            print(f"\nCSV report saved: {output_csv}")
        return 0 if all(r["exit_code"] == 0 and not r["needs_review"] for r in results) else 1

    # Mode 2: independent runs list
    if runs:
        if not isinstance(runs, list):
            print("ERROR: 'runs' must be a list")
            return 1
        results: list[dict[str, Any]] = []
        for i, run_cfg in enumerate(runs, 1):
            if not isinstance(run_cfg, dict):
                print(f"ERROR: run {i} is not a mapping")
                return 1
            run_spec = run_cfg.get("spec")
            # Resolve spec reference from prompts module for each run.
            if run_spec:
                run_spec = _resolve_spec_from_prompts(run_spec)
            run_cpu = run_cfg.get("cpu")
            run_mock = run_cfg.get("mock", mock)
            run_from = run_cfg.get("from", start_from)
            run_stop = run_cfg.get("to", stop_at)
            run_parent = run_cfg.get("parent_run_id", parent_run_id)
            run_id_override = run_cfg.get("run_id")
            rid = run_id_override or (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + f"-{i:03d}")

            if run_parent and not run_spec:
                thin = _restore_latest_snapshot(run_parent)
                if thin is None:
                    print(f"ERROR: No checkpoint found in parent run {run_parent}")
                    return 1
                from src.state_types import WorkflowState
                state = WorkflowState(**hydrate_state(thin))
                run_spec = state.spec
                run_cpu = state.cpu_name or run_cpu or cpu_name
            elif not run_spec:
                print(f"ERROR: run {i} missing 'spec'")
                return 1

            if not run_cpu:
                run_cpu = cpu_name

            print(f"\n{'=' * 60}")
            print(f"RUN [{i}/{len(runs)}] run_id={rid}")
            print(f"  spec={run_spec!r}")
            print(f"  cpu={run_cpu}")
            print("=" * 60)
            exit_code, _, final_state = _run_single(
                spec=run_spec,
                cpu_name=run_cpu,
                mock=run_mock,
                start_from=run_from,
                stop_at=run_stop,
                run_id=rid,
                parent_run_id=run_parent,
                _runner=runner,
            )
            results.append({
                "run_id": rid,
                "spec": run_spec,
                "cpu": run_cpu,
                "exit_code": exit_code,
                "needs_review": final_state.needs_review if final_state else False,
                "error": final_state.last_error if final_state else "",
            })
        _print_summary(results)
        if output_csv:
            import csv
            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["run_id", "spec", "cpu", "exit_code", "needs_review", "error"])
                writer.writeheader()
                writer.writerows(results)
            print(f"\nCSV report saved: {output_csv}")
        return 0 if all(r["exit_code"] == 0 and not r["needs_review"] for r in results) else 1

    # Single-run mode
    exit_code, _, _ = _run_single(
        spec=spec,
        cpu_name=cpu_name,
        mock=mock,
        start_from=start_from,
        stop_at=stop_at,
        run_id=run_id,
        parent_run_id=parent_run_id,
        _runner=runner,
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
