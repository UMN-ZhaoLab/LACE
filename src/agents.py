"""Agent implementations for the LACE pipeline.

These functions run in **Auto Mode**: they assemble prompts via
`src.interactive_engine`, invoke the LLM, and merge results back into state.

For **Interactive Mode** (MCP), the same build/validate/merge functions are
exposed directly through `mcp_server.py` so that the client LLM handles
generation while the server maintains orchestration and state.
"""

from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import LACEConfig
from src.cpu_analyzer import analyze_cpu_structure
from src.interactive_engine import (
    STEP_REGISTRY,
    build_candidate_prompt,
    build_cpu_analysis_prompt,
    build_op2hdl_prompt,
    build_spec2op_prompt,
    merge_candidate_result,
    merge_cpu_analysis_result,
    merge_op2hdl_result,
    merge_spec2op_result,
)
from src.llm import get_chat_model, get_structured_runnable
from src.nodes.agent_runner import invoke_with_backoff
from src.state_types import (
    CandidateModulesOut,
    HdlTasksOut,
    OpsOut,
    WorkflowState,
    ensure_state,
)


def spec_to_ops(state: WorkflowState | dict[str, Any]) -> WorkflowState:
    """Convert specification to operations using LLM (Auto Mode)."""
    state = ensure_state(state)
    prompt = build_spec2op_prompt(state)

    model = get_chat_model()
    messages = [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=prompt["human"]),
    ]
    if prompt.get("memory"):
        messages.insert(1, SystemMessage(content=prompt["memory"]))

    structured = get_structured_runnable(model, OpsOut)
    try:
        raw = invoke_with_backoff(structured, messages, LACEConfig.MAX_STAGE_RETRIES + 1)
        response = cast(OpsOut, raw)
    except RuntimeError as exc:
        return state.model_copy(update={"needs_review": True, "last_error": str(exc)})

    return merge_spec2op_result(state, response)


def op_to_hdl_tasks(state: WorkflowState | dict[str, Any]) -> WorkflowState:
    """Convert operations to HDL tasks using LLM (Auto Mode)."""
    state = ensure_state(state)
    if not state.ops:
        return state

    # Guard: skip only when the current op already has a plan. Plans for prior
    # ops must not prevent the planner from processing the remaining ops.
    current_op_is_planned = state.op_index in state.hdl_task_op_index_map
    if current_op_is_planned and state.retry_stage != "op_to_hdl_tasks":
        return state

    prompt = build_op2hdl_prompt(state)

    model = get_chat_model()
    messages = [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=prompt["human"]),
    ]
    if prompt.get("memory"):
        messages.insert(1, SystemMessage(content=prompt["memory"]))

    structured = get_structured_runnable(model, HdlTasksOut)
    try:
        raw = invoke_with_backoff(structured, messages, LACEConfig.MAX_STAGE_RETRIES + 1)
        response = cast(HdlTasksOut, raw)
    except RuntimeError as exc:
        return state.model_copy(update={"needs_review": True, "last_error": str(exc)})

    return merge_op2hdl_result(state, response)


def analyze_cpu_structure_agent(state: WorkflowState | dict[str, Any]) -> WorkflowState:
    """Analyze CPU architecture by feeding RTL excerpts to an LLM (Auto Mode)."""
    state = ensure_state(state)
    if not state.cpu_dir:
        return state.model_copy(update={"cpu_analysis_skipped": True})

    prompt = build_cpu_analysis_prompt(state)
    module_index = prompt.get("module_index", [])

    model = get_chat_model()
    messages = [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=prompt["human"]),
    ]
    try:
        raw = invoke_with_backoff(model, messages, LACEConfig.MAX_STAGE_RETRIES + 1)
        response_content = raw.content if hasattr(raw, "content") else str(raw)
    except RuntimeError as exc:
        return state.model_copy(update={"needs_review": True, "last_error": str(exc)})

    return merge_cpu_analysis_result(state, response_content, module_index)


def select_candidate_modules(state: WorkflowState | dict[str, Any]) -> WorkflowState:
    """Select candidate CPU modules for the current operation (Auto Mode)."""
    state = ensure_state(state)
    if not state.hdl_tasks or not state.cpu_summary:
        return state

    prompt = build_candidate_prompt(state)

    model = get_chat_model()
    messages = [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=prompt["human"]),
    ]
    if prompt.get("memory"):
        messages.insert(1, SystemMessage(content=prompt["memory"]))

    structured = get_structured_runnable(model, CandidateModulesOut)
    try:
        raw = invoke_with_backoff(structured, messages, LACEConfig.MAX_STAGE_RETRIES + 1)
        response = cast(CandidateModulesOut, raw)
    except RuntimeError as exc:
        return state.model_copy(update={"needs_review": True, "last_error": str(exc)})

    return merge_candidate_result(state, response)
