"""Graph-mode pipeline runner with artifact-based snapshot persistence.

The legacy imperative path (workflow.py + run_pipeline_segment) has been
removed; the compiled LangGraph in src.main_graph is the single execution
path. This module wraps it with checkpointing, run-DB bookkeeping, mock-LLM
support, and summary reporting, and exposes snapshot listing/loading for the
MCP server and CLI.
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock, patch

from src.artifact_store import (
    hydrate_state,
    load_checkpoint,
    load_thin_state,
    reports_dir,
    save_checkpoint,
    save_step_output,
    workspace_dir,
)
from src.config import LACEConfig
from src.main_graph import builder
from src.run_db import insert_run, insert_step, update_run_status, get_run_status
from src.run_manifest import create_manifest, load_manifest
from src.state_types import WorkflowState


# ---------------------------------------------------------------------------
# Node names (derived from the compiled graph, used by MCP/CLI for listing)
# ---------------------------------------------------------------------------


def _graph_node_names() -> list[str]:
    """User-facing node names from the compiled graph (excluding __start__/__end__)."""
    try:
        return [
            n
            for n in builder.compile().get_graph().nodes
            if not n.startswith("__")
        ]
    except Exception:
        # Fallback static list kept in sync with main_graph.py
        return [
            "cpu_resolver",
            "spec2op_agent",
            "spec2op_gate",
            "cpu_structure_analyzer",
            "candidate_module_selector",
            "candidate_gate",
            "op2hdl_planner",
            "op2hdl_gate",
            "dispatch",
            "rag_retriever",
            "interface_writer",
            "interface_syntax_check",
            "arithmetic_writer",
            "check_arithmetic_syntax",
            "arithmetic_integrator",
            "semantic_port_check",
            "original_function_checker",
            "insn_model_writer",
            "final_function_checker",
            "formal_gate",
        ]


NODE_NAMES: list[str] = _graph_node_names()


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


def _make_mock_llm() -> MagicMock:
    """Create a mock chat model that returns plausible structured outputs."""
    from langchain_core.messages import AIMessage

    from src.state_types import (
        CandidateModulesOut,
        CpuStructureOut,
        HdlTasksOut,
        OpsOut,
    )

    model = MagicMock()

    def _with_structured_output(schema: Any) -> MagicMock:
        runnable = MagicMock()

        def _invoke(messages: Any) -> Any:
            if schema is OpsOut:
                return OpsOut(
                    ops=["RdInstr()", "RdRS1()", "RdRS2()", "WrRD()"],
                    arithmetic_ops="SLICE(imm, 4, 0)",
                    confidence="high",
                )
            if schema is CpuStructureOut:
                return CpuStructureOut(
                    summary="pipelined 4-stage CPU",
                    module_index=["Fetch: ifu.v", "Execute: alu.v"],
                )
            if schema is CandidateModulesOut:
                return CandidateModulesOut(
                    candidates=[
                        {"module": "alu.v", "reason": "add rotate logic", "related_ops": ["ROL"]}
                    ],
                    confidence="high",
                )
            if schema is HdlTasksOut:
                return HdlTasksOut(
                    hdl_tasks=["add rotate port to alu"],
                    confidence="high",
                )
            return schema()

        runnable.invoke.side_effect = _invoke
        return runnable

    model.with_structured_output.side_effect = _with_structured_output

    def _direct_invoke(messages: Any) -> AIMessage:
        return AIMessage(content="module top; // mock generated code\nendmodule")

    model.invoke.side_effect = _direct_invoke

    # bind_tools / bind return the same mock so that interface_writer's
    # extra_body binding and ReAct loop fall through to a plain invoke.
    def _bind_tools(tools):
        return model

    def _bind(**kwargs):
        return model

    model.bind_tools = _bind_tools
    model.bind = _bind
    return model


def run_graph_segment(
    spec: str = "",
    cpu_name: str = "",
    start_from: str | None = None,
    stop_at: str | None = None,
    mock: bool = False,
    run_id: str | None = None,
    parent_run_id: str | None = None,
    max_task_retries: int = 1,
) -> tuple[WorkflowState, list[dict[str, Any]], str]:
    """Execute pipeline using the compiled LangGraph graph with checkpointing.

    Reuses src.main_graph.builder so that conditional edges, retry gates,
    and parallel forks are honoured.  A SqliteSaver checkpointer provides
    true resume semantics (stream(None, config) continues from the last
    saved superstep).
    """
    import sqlite3
    from contextlib import ExitStack, nullcontext
    from hashlib import sha256
    from pathlib import Path

    from langgraph.checkpoint.sqlite import SqliteSaver
    from src.state_types import ensure_state

    rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    thread_id = parent_run_id or rid

    # Persistent checkpointer shared across all graph-mode runs
    checkpoint_db = Path(LACEConfig.ARTIFACT_DIR) / "checkpoints.sqlite"
    checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(checkpoint_db), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

    # Resume when parent_run_id is given or when both run_id + start_from are given
    is_resume = bool(parent_run_id or (run_id and start_from))

    if is_resume:
        stream_input = None  # Ask LangGraph to resume from checkpoint
    else:
        stream_input = WorkflowState(spec=spec, cpu_name=cpu_name, run_id=rid)

    state = WorkflowState(spec=spec, cpu_name=cpu_name, run_id=rid)

    create_manifest(
        run_id=rid,
        cpu_name=cpu_name or state.cpu_name or "unknown",
        spec=spec or state.spec,
        parent_run_id=parent_run_id,
        resume_from=start_from,
    )

    insert_run(
        run_id=rid,
        parent_run_id=parent_run_id,
        cpu_name=cpu_name or state.cpu_name or "unknown",
        spec_hash=sha256((spec or state.spec).encode()).hexdigest()[:16],
        status="running",
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    log: list[dict[str, Any]] = []

    if mock:
        ctx = ExitStack()
        ctx.enter_context(patch("src.agents.get_chat_model", _make_mock_llm))
        ctx.enter_context(patch("src.writers.get_chat_model", _make_mock_llm))
        ctx.enter_context(patch("src.arithmetic_integrator.get_chat_model", _make_mock_llm))
    else:
        ctx = nullcontext()

    _seen_nodes: set[str] = set()

    with ctx:
        try:
            stream_iterator = graph.stream(
                stream_input, config, stream_mode="updates"
            )
        except Exception:
            if is_resume:
                # Checkpoint missing or corrupt — fall back to fresh run
                stream_input = WorkflowState(spec=spec, cpu_name=cpu_name, run_id=rid)
                stream_iterator = graph.stream(
                    stream_input, config, stream_mode="updates"
                )
            else:
                raise

        try:
            for step in stream_iterator:
                for node_name, node_update in step.items():
                    # Skip LangGraph internal metadata nodes
                    if node_name.startswith("__"):
                        continue

                    _seen_nodes.add(node_name)

                    # Merge partial update into current state
                    if node_update is None:
                        # LangGraph normalises empty diffs to None; nothing to merge
                        continue
                    if isinstance(node_update, dict):
                        state = state.model_copy(update=node_update)
                    else:
                        state = ensure_state(node_update)

                    status = "ok"
                    error = ""
                    if state.needs_review:
                        status = "halt"
                        error = state.last_error or "needs_review=True"

                    log.append({
                        "step_index": len(log),
                        "step_name": node_name,
                        "status": status,
                        "error": error,
                    })

                    save_checkpoint(rid, len(log) - 1, node_name, state)
                    save_step_output(rid, len(log) - 1, node_name, status, error)
                    insert_step(rid, len(log) - 1, node_name, status, error)

                # stop_at handling — evaluated after the full superstep so parallel
                # nodes (e.g. cpu_structure_analyzer) are always logged.
                if stop_at:
                    if stop_at == "spec_and_cpu":
                        # Wait until both parallel branches have run and the retry
                        # gate has had a chance to fire (or pass) before stopping.
                        if (
                            "spec2op_agent" in _seen_nodes
                            and "cpu_structure_analyzer" in _seen_nodes
                            and "spec2op_gate" in _seen_nodes
                            and state.retry_stage == ""
                        ):
                            if log:
                                log[-1]["status"] = "stopped"
                            update_run_status(
                                rid,
                                "stopped",
                                completed_at=datetime.now(timezone.utc).isoformat(),
                            )
                            _write_summary_report(rid, state, log)
                            return state, log, rid
                    elif log and log[-1]["step_name"] == stop_at:
                        log[-1]["status"] = "stopped"
                        update_run_status(
                            rid,
                            "stopped",
                            completed_at=datetime.now(timezone.utc).isoformat(),
                        )
                        _write_summary_report(rid, state, log)
                        return state, log, rid

                # NOTE: We intentionally do NOT halt inside the stream loop.
                # LangGraph gates handle retries automatically; halting here
                # would abort the graph before the retry gate has a chance to
                # route back to the agent.  We let the graph run until it
                # reaches a terminal state (END) or stop_at triggers.
        finally:
            conn.close()

    final_status = "success"
    for entry in log:
        if entry["status"] in ("error", "halt"):
            final_status = "failed"
            break
    if state.needs_review:
        final_status = "needs_review"

    update_run_status(
        rid,
        final_status,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    _write_summary_report(rid, state, log)
    return state, log, rid


def _write_summary_report(
    run_id: str, state: WorkflowState, log: list[dict[str, Any]]
) -> None:
    """Write reports/summary.json and reports/report.md."""
    d = reports_dir(run_id)

    failed_step = None
    for entry in log:
        if entry["status"] in ("error", "halt"):
            failed_step = entry["step_name"]
            break

    retry_counts: dict[str, int] = {}
    for entry in log:
        if entry["status"] == "error":
            retry_counts[entry["step_name"]] = retry_counts.get(entry["step_name"], 0) + 1

    summary = {
        "run_id": run_id,
        "cpu": state.cpu_name,
        "spec": state.spec[:200] if state.spec else "",
        "status": "failed" if failed_step else "success",
        "failed_step": failed_step,
        "ops_count": len(state.ops),
        "hdl_tasks_count": len(state.hdl_tasks),
        "candidate_modules": [c.module for c in state.candidate_modules],
        "modified_files": [],
        "syntax_pass": state.interface_syntax_ok,
        "function_pass": state.function_ok,
        "retry_count": retry_counts,
        "needs_review": state.needs_review,
        "last_error": state.last_error,
    }
    (d / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    md_lines = [
        f"# Run {run_id}",
        "",
        "## Spec",
        f"```\n{state.spec[:500]}\n```" if state.spec else "_No spec_",
        "",
        "## Result",
        f"**Status:** {summary['status']}",
        f"**Failed at:** {failed_step}" if failed_step else "**All steps completed**",
        "",
        "## Steps",
        "| Step | Status | Error |",
        "|------|--------|-------|",
    ]
    for entry in log:
        status = entry["status"]
        error = entry.get("error", "")[:60]
        md_lines.append(f"| {entry['step_name']} | {status} | {error} |")

    md_lines.extend([
        "",
        "## Generated Ops",
    ])
    for op in state.ops:
        md_lines.append(f"- {op}")

    if state.hdl_tasks:
        md_lines.extend(["", "## HDL Tasks"])
        for task in state.hdl_tasks:
            md_lines.append(f"- {task}")

    if state.last_error:
        md_lines.extend(["", "## Error", f"```\n{state.last_error}\n```"])

    (d / "report.md").write_text("\n".join(md_lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Snapshot listing / loading helpers for CLI
# ---------------------------------------------------------------------------


def list_snapshots() -> list[dict[str, Any]]:
    """Return a list of all snapshot runs with their checkpoint files."""
    from src.config import LACEConfig
    runs_dir = Path(LACEConfig.ARTIFACT_DIR) / "runs"
    results: list[dict[str, Any]] = []
    if not runs_dir.exists():
        return results
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        checkpoints: list[str] = []
        cp_dir = run_dir / "state" / "checkpoints"
        if cp_dir.exists():
            for f in sorted(cp_dir.glob("*.json")):
                checkpoints.append(f.name)
        results.append({"run_id": run_dir.name, "checkpoints": checkpoints})
    return results


def load_snapshot(run_id: str, node_name: str) -> dict[str, Any] | None:
    """Load the most recent checkpoint for *node_name* in *run_id*.

    Tries the named thin-state file first, then any per-node checkpoint
    ("<idx>_<node>.json"). Returns None if neither exists.
    """
    thin = load_thin_state(run_id, node_name)
    if thin is not None:
        return hydrate_state(thin)
    from src.artifact_store import checkpoints_dir as _cpd

    cp_dir = _cpd(run_id)
    matches = sorted(cp_dir.glob(f"*_{node_name}.json"), reverse=True)
    for f in matches:
        try:
            thin = json.loads(f.read_text(encoding="utf-8"))
            return hydrate_state(thin)
        except Exception:
            continue
    return None
