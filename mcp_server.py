#!/usr/bin/env python3
"""MCP server for the LACE pipeline.

Exposes LACE workflow, graph execution, and CPU registry as MCP tools
so that kimi-cli (or any MCP client) can invoke them.

Usage (stdio transport):
    python mcp_server.py

The server logs to stderr only; stdout is reserved for JSON-RPC messages.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so `src.*` imports work
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env before importing LACE modules
try:
    from dotenv import load_dotenv

    _env_path = _PROJECT_ROOT / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=False)
except Exception:
    pass

from mcp.server.fastmcp import FastMCP

from src.cpu_registry import list_cpu_choices, load_cpu_registry, resolve_cpu
from src.main_graph import graph as compiled_graph
from src.pipeline_runner import run_graph_segment
from src.state_types import WorkflowState, ensure_state


# ---------------------------------------------------------------------------
# Logging helpers — MUST write to stderr so stdout stays clean for JSON-RPC
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(f"[LACE-MCP] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------

mcp = FastMCP("lace")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_to_dict(state: WorkflowState) -> dict[str, Any]:
    """Serialize a WorkflowState to a plain dict for JSON transport."""
    return state.model_dump(mode="json")


def _ensure_project_root() -> None:
    """Make sure CWD is the project root so relative paths in config work."""
    os.chdir(_PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def lace_list_cpus() -> str:
    """List all supported CPU targets configured in config/cpus.yaml.

    Returns a JSON array of CPU names.
    """
    _ensure_project_root()
    try:
        cpus = list_cpu_choices()
        return json.dumps({"cpus": cpus}, indent=2, ensure_ascii=False)
    except Exception as exc:
        _log(f"lace_list_cpus error: {exc}")
        return json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False)


@mcp.tool()
async def lace_get_cpu_info(cpu_name: str) -> str:
    """Get detailed configuration for a specific CPU target.

    Args:
        cpu_name: One of the supported CPU names (e.g. "picorv32", "ibex").

    Returns a JSON object with cpu_dir, top_file, sv_include_dir, etc.
    """
    _ensure_project_root()
    try:
        cfg = resolve_cpu(cpu_name)
        payload = {
            "name": cfg.name,
            "cpu_dir": cfg.cpu_dir,
            "top_file": cfg.top_file,
            "sv_include_dir": cfg.sv_include_dir,
            "verilator_std": cfg.verilator_std,
            "verilator_waive_flags": cfg.verilator_waive_flags,
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)
    except Exception as exc:
        _log(f"lace_get_cpu_info error: {exc}")
        return json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False)


@mcp.tool()
async def lace_run_workflow(
    spec: str,
    cpu_name: str = "",
    cpu_dir: str = "",
    max_task_retries: int = 1,
) -> str:
    """Run the full LACE pipeline for a given instruction specification.

    This is the primary high-level tool. It takes a textual RISC-V extension
    spec, decomposes it into micro-operations, analyses the target CPU,
    plans HDL modifications, generates Verilog/SystemVerilog code, and runs
    syntax checks.

    Args:
        spec: Textual description of the new instruction (encoding + behaviour).
        cpu_name: Target CPU name from config/cpus.yaml (e.g. "picorv32").
                  If provided, cpu_dir is resolved automatically.
        cpu_dir: Optional explicit CPU prototype directory. Overrides cpu_name.
        max_task_retries: How many times a single HDL task may retry (default 1).

    Returns:
        A JSON object containing the final WorkflowState fields:
        - ops: list of decomposed operations
        - hdl_tasks: list of planned HDL tasks
        - interface_code: generated interface Verilog/SV
        - arithmetic_code: generated arithmetic Verilog/SV
        - candidate_modules: selected candidate modules
        - cpu_summary: CPU structure summary
        - needs_review: whether the pipeline halted for human review
        - last_error: error message if any
        - run_id: unique run identifier
    """
    _ensure_project_root()
    _log(f"lace_run_workflow start | cpu={cpu_name or cpu_dir}")
    try:
        loop = asyncio.get_running_loop()
        final_state, _log_entries, rid = await loop.run_in_executor(
            None,
            lambda: run_graph_segment(
                spec=spec,
                cpu_name=cpu_name,
                max_task_retries=max_task_retries,
            ),
        )
        payload = _state_to_dict(final_state)
        _log(f"lace_run_workflow done | run_id={rid} | needs_review={final_state.needs_review}")
        return json.dumps(payload, indent=2, ensure_ascii=False)
    except Exception as exc:
        _log(f"lace_run_workflow exception: {exc}\n{traceback.format_exc()}")
        return json.dumps(
            {"error": str(exc), "traceback": traceback.format_exc()},
            indent=2,
            ensure_ascii=False,
        )


@mcp.tool()
async def lace_run_graph(
    spec: str,
    cpu_name: str = "",
    cpu_dir: str = "",
) -> str:
    """Run the compiled LangGraph (main_graph.py:graph) for a specification.

    This is an alternative entry point that uses the declarative StateGraph
    rather than the imperative workflow runner. It is useful when you want
    the exact retry-gate behaviour defined in main_graph.py.

    Args:
        spec: Textual description of the new instruction.
        cpu_name: Target CPU name from config/cpus.yaml.
        cpu_dir: Optional explicit CPU prototype directory.

    Returns:
        A JSON object with the final WorkflowState after graph completion.
    """
    _ensure_project_root()
    _log(f"lace_run_graph start | cpu={cpu_name or cpu_dir}")
    try:
        initial_state = WorkflowState(
            spec=spec,
            cpu_name=cpu_name,
            cpu_dir=cpu_dir,
        )
        loop = asyncio.get_running_loop()
        final_state = await loop.run_in_executor(
            None,
            lambda: compiled_graph.invoke(initial_state),
        )
        payload = _state_to_dict(final_state)
        _log(f"lace_run_graph done | run_id={final_state.run_id}")
        return json.dumps(payload, indent=2, ensure_ascii=False)
    except Exception as exc:
        _log(f"lace_run_graph exception: {exc}\n{traceback.format_exc()}")
        return json.dumps(
            {"error": str(exc), "traceback": traceback.format_exc()},
            indent=2,
            ensure_ascii=False,
        )


@mcp.tool()
async def lace_resume_workflow(state_json: str) -> str:
    """Resume a previously-saved LACE workflow from its JSON state.

    Args:
        state_json: A JSON string representing a WorkflowState (the same format
                    returned by lace_run_workflow or lace_run_graph).

    Returns:
        A JSON object with the updated WorkflowState after resuming.
    """
    _ensure_project_root()
    _log("lace_resume_workflow start")
    try:
        state_dict = json.loads(state_json)
        state = ensure_state(state_dict)
        loop = asyncio.get_running_loop()
        final_state, _log_entries, rid = await loop.run_in_executor(
            None,
            lambda: run_graph_segment(
                spec=state.spec,
                cpu_name=state.cpu_name,
                run_id=state.run_id,
                parent_run_id=state.run_id,
                max_task_retries=1,
            ),
        )
        payload = _state_to_dict(final_state)
        _log(f"lace_resume_workflow done | run_id={rid}")
        return json.dumps(payload, indent=2, ensure_ascii=False)
    except Exception as exc:
        _log(f"lace_resume_workflow exception: {exc}\n{traceback.format_exc()}")
        return json.dumps(
            {"error": str(exc), "traceback": traceback.format_exc()},
            indent=2,
            ensure_ascii=False,
        )


@mcp.tool()
async def lace_spec_to_ops(spec: str, cpu_name: str = "") -> str:
    """Run only the spec→ops decomposition stage.

    Args:
        spec: Textual instruction specification.
        cpu_name: Optional CPU name for memory context.

    Returns:
        JSON with ops, confidence, and any validation errors.
    """
    _ensure_project_root()
    from src.agents import spec_to_ops
    from src.nodes.cpu_resolver import resolve_cpu_state

    try:
        state = WorkflowState(spec=spec, cpu_name=cpu_name)
        state = resolve_cpu_state(state)
        state = spec_to_ops(state)
        return json.dumps(
            {
                "ops": state.ops,
                "spec_confidence": state.spec_confidence,
                "needs_review": state.needs_review,
                "last_error": state.last_error,
                "run_id": state.run_id,
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception as exc:
        _log(f"lace_spec_to_ops exception: {exc}")
        return json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False)


@mcp.tool()
async def lace_analyze_cpu(cpu_name: str = "", cpu_dir: str = "") -> str:
    """Run only the CPU structure analysis stage.

    Args:
        cpu_name: Target CPU name.
        cpu_dir: Optional explicit CPU directory.

    Returns:
        JSON with cpu_summary, module_index, and analysis notes.
    """
    _ensure_project_root()
    from src.agents import analyze_cpu_structure_agent
    from src.nodes.cpu_resolver import resolve_cpu_state

    try:
        state = WorkflowState(cpu_name=cpu_name, cpu_dir=cpu_dir)
        state = resolve_cpu_state(state)
        state = analyze_cpu_structure_agent(state)
        return json.dumps(
            {
                "cpu_summary": state.cpu_summary,
                "cpu_module_index": state.cpu_module_index,
                "cpu_analysis_skipped": state.cpu_analysis_skipped,
                "needs_review": state.needs_review,
                "last_error": state.last_error,
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception as exc:
        _log(f"lace_analyze_cpu exception: {exc}")
        return json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False)


@mcp.tool()
async def lace_plan_hdl(
    spec: str,
    ops: list[str],
    cpu_name: str = "",
    cpu_summary: str = "",
    cpu_module_index: list[str] | None = None,
    candidate_modules: list[dict[str, Any]] | None = None,
) -> str:
    """Run the HDL task planner (op2hdl) for a given operation list.

    This is useful when you already have decomposed ops and a CPU summary
    and want to generate the HDL modification plan without running the
    full pipeline.

    Args:
        spec: Original instruction specification.
        ops: List of decomposed operations.
        cpu_name: Target CPU name.
        cpu_summary: CPU structure summary text.
        cpu_module_index: List of module names/index entries.
        candidate_modules: List of {module, reason, related_ops} dicts.

    Returns:
        JSON with hdl_tasks, hdl_confidence, and validation status.
    """
    _ensure_project_root()
    from src.agents import op_to_hdl_tasks, select_candidate_modules
    from src.nodes.cpu_resolver import resolve_cpu_state
    from src.state_types import CandidateModule

    try:
        mods = [
            CandidateModule(**c) for c in (candidate_modules or [])
        ]
        state = WorkflowState(
            spec=spec,
            ops=ops,
            cpu_name=cpu_name,
            cpu_summary=cpu_summary,
            cpu_module_index=cpu_module_index or [],
            candidate_modules=mods,
        )
        state = resolve_cpu_state(state)

        # If no candidates provided, run selector first
        if not state.candidate_modules and state.cpu_summary:
            state = select_candidate_modules(state)

        state = op_to_hdl_tasks(state)
        return json.dumps(
            {
                "hdl_tasks": state.hdl_tasks,
                "hdl_confidence": state.hdl_confidence,
                "candidate_modules": [
                    {"module": c.module, "reason": c.reason, "related_ops": c.related_ops}
                    for c in state.candidate_modules
                ],
                "needs_review": state.needs_review,
                "last_error": state.last_error,
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception as exc:
        _log(f"lace_plan_hdl exception: {exc}")
        return json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False)


@mcp.tool()
async def lace_generate_code(
    spec: str,
    ops: list[str],
    hdl_tasks: list[str],
    cpu_name: str = "",
    cpu_dir: str = "",
    cpu_summary: str = "",
    candidate_modules: list[dict[str, Any]] | None = None,
) -> str:
    """Generate Verilog/SystemVerilog interface and arithmetic code.

    This runs only the code-generation and syntax-check stages, skipping
    spec decomposition and HDL planning. It is useful when you already
    have a complete ops + hdl_tasks plan.

    Args:
        spec: Original instruction specification.
        ops: List of decomposed operations.
        hdl_tasks: List of HDL modification tasks.
        cpu_name: Target CPU name.
        cpu_dir: Optional explicit CPU directory.
        cpu_summary: CPU structure summary.
        candidate_modules: List of {module, reason, related_ops} dicts.

    Returns:
        JSON with interface_code, arithmetic_code, syntax check flags,
        and any errors.
    """
    _ensure_project_root()
    from src.checks import (
        check_arithmetic_syntax,
        check_interface_syntax,
        check_semantic_ports,
        function_check,
    )
    from src.nodes.cpu_resolver import resolve_cpu_state
    from src.state_types import CandidateModule
    from src.writers import arithmetic_writer, interface_writer

    try:
        mods = [CandidateModule(**c) for c in (candidate_modules or [])]
        state = WorkflowState(
            spec=spec,
            ops=ops,
            hdl_tasks=hdl_tasks,
            cpu_name=cpu_name,
            cpu_dir=cpu_dir,
            cpu_summary=cpu_summary,
            candidate_modules=mods,
        )
        state = resolve_cpu_state(state)

        loop = asyncio.get_running_loop()

        def _gen():
            st = state
            # Interface loop
            for hdl_index in range(len(st.hdl_tasks)):
                st = st.model_copy(update={"hdl_index": hdl_index})
                st = interface_writer(st)
                st = check_interface_syntax(st)
            st = check_semantic_ports(st)
            # Arithmetic
            st = arithmetic_writer(st)
            st = check_arithmetic_syntax(st)
            st = function_check(st)
            return st

        final_state = await loop.run_in_executor(None, _gen)
        return json.dumps(
            {
                "interface_code": final_state.interface_code,
                "arithmetic_code": final_state.arithmetic_code,
                "interface_syntax_ok": final_state.interface_syntax_ok,
                "arithmetic_syntax_ok": final_state.arithmetic_syntax_ok,
                "function_ok": final_state.function_ok,
                "needs_review": final_state.needs_review,
                "last_error": final_state.last_error,
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception as exc:
        _log(f"lace_generate_code exception: {exc}")
        return json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Pipeline segment / snapshot tools (from scripts/test_nodes_sequential.py)
# ---------------------------------------------------------------------------


@mcp.tool()
async def lace_list_pipeline_nodes() -> str:
    """List all pipeline node names available for segmented execution.

    Use these names with lace_run_pipeline_segment's from_node / to_node.

    Returns:
        JSON array of node names in execution order.
    """
    _ensure_project_root()
    try:
        from src.pipeline_runner import NODE_NAMES

        return json.dumps({"nodes": NODE_NAMES}, indent=2, ensure_ascii=False)
    except Exception as exc:
        _log(f"lace_list_pipeline_nodes error: {exc}")
        return json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False)


@mcp.tool()
async def lace_run_pipeline_segment(
    spec: str = "",
    cpu_name: str = "",
    from_node: str = "",
    to_node: str = "",
    mock: bool = False,
    run_id: str = "",
    parent_run_id: str = "",
    max_task_retries: int = 1,
) -> str:
    """Run a segment of the LACE pipeline node-by-node with snapshots.

    This is the MCP equivalent of scripts/test_nodes_sequential.py.
    It allows you to start from any node, stop at any node, use mock LLM
    for fast testing, and resume from a previous run's snapshots.

    Args:
        spec: Instruction specification (required for fresh runs).
        cpu_name: Target CPU name (required for fresh runs).
        from_node: Node name to start from (e.g. "op2hdl_planner").
                   If omitted, starts from the beginning.
        to_node: Node name to stop at (inclusive). If omitted, runs to end.
        mock: If True, use a mock LLM that returns canned responses.
        run_id: Explicit run ID. If omitted, auto-generated as timestamp.
        parent_run_id: Load initial state from a previous run's snapshots.
        max_task_retries: Per-HDL-task retry budget (default 1).

    Returns:
        JSON with:
        - run_id: the run identifier
        - final_state: serialized WorkflowState
        - execution_log: list of {node_name, status, snapshot_path, error}
    """
    _ensure_project_root()
    _log(
        f"lace_run_pipeline_segment | from={from_node or 'start'} "
        f"to={to_node or 'end'} mock={mock} parent={parent_run_id}"
    )
    try:
        loop = asyncio.get_running_loop()
        final_state, log, rid = await loop.run_in_executor(
            None,
            lambda: run_graph_segment(
                spec=spec,
                cpu_name=cpu_name,
                start_from=from_node or None,
                stop_at=to_node or None,
                mock=mock,
                run_id=run_id or None,
                parent_run_id=parent_run_id or None,
                max_task_retries=max_task_retries,
            ),
        )
        payload = {
            "run_id": rid,
            "final_state": _state_to_dict(final_state),
            "execution_log": log,
        }
        _log(f"lace_run_pipeline_segment done | run_id={rid} | needs_review={final_state.needs_review}")
        return json.dumps(payload, indent=2, ensure_ascii=False)
    except Exception as exc:
        _log(f"lace_run_pipeline_segment exception: {exc}\n{traceback.format_exc()}")
        return json.dumps(
            {"error": str(exc), "traceback": traceback.format_exc()},
            indent=2,
            ensure_ascii=False,
        )


@mcp.tool()
async def lace_list_snapshots() -> str:
    """List all persisted snapshot runs.

    Returns:
        JSON array of {run_id, files} where files is a list of snapshot filenames.
    """
    _ensure_project_root()
    try:
        from src.pipeline_runner import list_snapshots

        snapshots = list_snapshots()
        return json.dumps({"snapshots": snapshots}, indent=2, ensure_ascii=False)
    except Exception as exc:
        _log(f"lace_list_snapshots error: {exc}")
        return json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False)


@mcp.tool()
async def lace_get_snapshot(run_id: str, node_name: str) -> str:
    """Load a specific snapshot as JSON.

    Args:
        run_id: The run identifier.
        node_name: The pipeline node name (e.g. "op2hdl_planner").

    Returns:
        JSON object representing the WorkflowState at that snapshot,
        or an error if not found.
    """
    _ensure_project_root()
    try:
        from src.pipeline_runner import load_snapshot

        state = load_snapshot(run_id, node_name)
        if state is None:
            return json.dumps(
                {"error": f"Snapshot not found: {run_id}/{node_name}"},
                indent=2,
                ensure_ascii=False,
            )
        return json.dumps(_state_to_dict(state), indent=2, ensure_ascii=False)
    except Exception as exc:
        _log(f"lace_get_snapshot error: {exc}")
        return json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Interactive-mode tools (orchestration engine exposed to client LLM)
# ---------------------------------------------------------------------------


@mcp.tool()
async def lace_get_current_step(state_json: str) -> str:
    """Determine the next logical pipeline step based on current state.

    Args:
        state_json: JSON string representing a WorkflowState.

    Returns:
        JSON with {current_step, needs_review, last_error}.
        current_step is null if all steps appear complete.
    """
    _ensure_project_root()
    try:
        from src.interactive_engine import get_current_step, get_workflow_status

        state = ensure_state(json.loads(state_json))
        status = get_workflow_status(state)
        return json.dumps(status, indent=2, ensure_ascii=False)
    except Exception as exc:
        _log(f"lace_get_current_step error: {exc}")
        return json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False)


@mcp.tool()
async def lace_get_step_prompt(step_name: str, state_json: str) -> str:
    """Get the full LLM prompt for a specific pipeline step.

    This is the core of Interactive Mode: the server assembles the prompt
    (system instruction, human payload, memory context, format requirements),
    and the client LLM (kimi) generates the response.

    Args:
        step_name: One of the registered step names (e.g. "spec_to_ops",
                   "cpu_analysis", "candidate_modules", "op2hdl_tasks").
        state_json: JSON string representing the current WorkflowState.

    Returns:
        JSON with {system, human, memory, expected_schema, notes, error}.
    """
    _ensure_project_root()
    try:
        from src.interactive_engine import STEP_REGISTRY

        state = ensure_state(json.loads(state_json))
        handler = STEP_REGISTRY.get(step_name)
        if handler is None:
            available = ", ".join(STEP_REGISTRY.keys())
            return json.dumps(
                {"error": f"Unknown step '{step_name}'. Available: {available}"},
                indent=2,
                ensure_ascii=False,
            )

        prompt = handler.build_prompt(state)
        payload = {
            "step": step_name,
            "description": handler.description,
            **prompt,
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)
    except Exception as exc:
        _log(f"lace_get_step_prompt error: {exc}")
        return json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False)


@mcp.tool()
async def lace_validate_step(step_name: str, state_json: str, raw_output: str) -> str:
    """Validate a raw LLM output for a specific pipeline step.

    Args:
        step_name: The step name.
        state_json: Current WorkflowState (for context, some validators may use it).
        raw_output: The raw JSON or text output from the LLM.

    Returns:
        JSON with {valid, error, parsed}.
    """
    _ensure_project_root()
    try:
        from src.interactive_engine import STEP_REGISTRY

        state = ensure_state(json.loads(state_json))
        handler = STEP_REGISTRY.get(step_name)
        if handler is None:
            return json.dumps(
                {"error": f"Unknown step: {step_name}"},
                indent=2,
                ensure_ascii=False,
            )

        # Try to parse raw_output as JSON if it looks like JSON
        candidate: Any = raw_output
        try:
            if isinstance(raw_output, str) and raw_output.strip().startswith(("{", "[")):
                candidate = json.loads(raw_output)
        except Exception:
            pass

        valid, error, parsed = handler.validate_output(candidate)
        return json.dumps(
            {
                "valid": valid,
                "error": error,
                "parsed": parsed.model_dump() if hasattr(parsed, "model_dump") else parsed,
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception as exc:
        _log(f"lace_validate_step error: {exc}")
        return json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False)


@mcp.tool()
async def lace_advance_state(step_name: str, state_json: str, raw_output: str) -> str:
    """Validate and merge an LLM output into the WorkflowState.

    This advances the workflow by one step. If validation fails,
    needs_review is set and last_error is populated.

    Args:
        step_name: The step name.
        state_json: Current WorkflowState.
        raw_output: Raw LLM output (JSON or text).

    Returns:
        JSON with {state, log} where state is the updated WorkflowState
        and log contains {step, valid, error, confidence}.
    """
    _ensure_project_root()
    try:
        from src.interactive_engine import advance_step

        state = ensure_state(json.loads(state_json))

        # Try JSON parse
        candidate: Any = raw_output
        try:
            if isinstance(raw_output, str) and raw_output.strip().startswith(("{", "[")):
                candidate = json.loads(raw_output)
        except Exception:
            pass

        updated, log = advance_step(state, step_name, candidate)
        return json.dumps(
            {
                "state": _state_to_dict(updated),
                "log": log,
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception as exc:
        _log(f"lace_advance_state error: {exc}")
        return json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False)


@mcp.tool()
async def lace_get_workflow_status(state_json: str) -> str:
    """Get a complete checklist of the workflow progress.

    Args:
        state_json: WorkflowState JSON.

    Returns:
        JSON with {current_step, needs_review, last_error, checklist}.
    """
    _ensure_project_root()
    try:
        from src.interactive_engine import get_workflow_status

        state = ensure_state(json.loads(state_json))
        status = get_workflow_status(state)
        return json.dumps(status, indent=2, ensure_ascii=False)
    except Exception as exc:
        _log(f"lace_get_workflow_status error: {exc}")
        return json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Pure-local tools (zero LLM cost)
# ---------------------------------------------------------------------------


@mcp.tool()
async def lace_read_cpu_module(cpu_name: str, relative_path: str) -> str:
    """Read the contents of a CPU source module.

    Args:
        cpu_name: Target CPU name (e.g. "picorv32").
        relative_path: Path relative to the CPU directory (e.g. "picorv32.v").

    Returns:
        JSON with {cpu_name, path, relative_path, content, lines}.
    """
    _ensure_project_root()
    try:
        from src.lace_tools import read_cpu_module

        result = read_cpu_module(cpu_name, relative_path)
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as exc:
        _log(f"lace_read_cpu_module error: {exc}")
        return json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False)


@mcp.tool()
async def lace_search_code(cpu_name: str, pattern: str, case_sensitive: bool = False) -> str:
    """Search for a regex pattern across all CPU source files.

    Args:
        cpu_name: Target CPU name.
        pattern: Regular expression to search for.
        case_sensitive: Whether the search is case-sensitive.

    Returns:
        JSON with {cpu_name, pattern, match_count, matches}.
    """
    _ensure_project_root()
    try:
        from src.lace_tools import search_code

        result = search_code(cpu_name, pattern, case_sensitive)
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as exc:
        _log(f"lace_search_code error: {exc}")
        return json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False)


@mcp.tool()
async def lace_list_cpu_modules(cpu_name: str) -> str:
    """List all modules in a CPU with keyword-based classification.

    Args:
        cpu_name: Target CPU name.

    Returns:
        JSON with {cpu_name, module_count, modules}.
    """
    _ensure_project_root()
    try:
        from src.lace_tools import list_cpu_modules

        result = list_cpu_modules(cpu_name)
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as exc:
        _log(f"lace_list_cpu_modules error: {exc}")
        return json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False)


@mcp.tool()
async def lace_apply_patch(file_path: str, patch_text: str) -> str:
    """Apply a SEARCH/REPLACE patch to a file.

    Args:
        file_path: Absolute path to the file.
        patch_text: Patch in SEARCH/REPLACE format.

    Returns:
        JSON with {success, path, original_lines, new_lines}.
    """
    _ensure_project_root()
    try:
        from src.lace_tools import apply_verilog_patch

        result = apply_verilog_patch(file_path, patch_text)
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as exc:
        _log(f"lace_apply_patch error: {exc}")
        return json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False)


@mcp.tool()
async def lace_run_verilator(
    code_snippet: str, cpu_name: str = "", include_dir: str = ""
) -> str:
    """Run Verilator lint-only check on a code snippet.

    Args:
        code_snippet: Verilog/SystemVerilog code.
        cpu_name: Optional CPU name to resolve include dir and flags.
        include_dir: Optional explicit include directory.

    Returns:
        JSON with {ok, output}.
    """
    _ensure_project_root()
    try:
        from src.lace_tools import run_verilator

        result = run_verilator(code_snippet, cpu_name, include_dir)
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as exc:
        _log(f"lace_run_verilator error: {exc}")
        return json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False)


@mcp.tool()
async def lace_get_predefined_ops() -> str:
    """Return the reference table of predefined ISA operations.

    Returns:
        JSON with {interface_ops, arithmetic_ops}.
    """
    _ensure_project_root()
    try:
        from src.lace_tools import get_predefined_ops_table

        result = get_predefined_ops_table()
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as exc:
        _log(f"lace_get_predefined_ops error: {exc}")
        return json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False)


@mcp.tool()
async def lace_get_cpu_summary(cpu_name: str) -> str:
    """Return the auto-generated CPU structure summary.

    Args:
        cpu_name: Target CPU name.

    Returns:
        JSON with {cpu_name, summary, module_index}.
    """
    _ensure_project_root()
    try:
        from src.lace_tools import get_cpu_summary_text

        result = get_cpu_summary_text(cpu_name)
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as exc:
        _log(f"lace_get_cpu_summary error: {exc}")
        return json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    _log("Starting LACE MCP server (stdio transport)")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
