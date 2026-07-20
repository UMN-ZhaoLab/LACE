"""Main graph definition for the LACE pipeline."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agents import (
    analyze_cpu_structure_agent,
    op_to_hdl_tasks,
    select_candidate_modules,
    spec_to_ops,
)
from src.checks import (
    check_arithmetic_syntax,
    check_interface_syntax,
    check_semantic_ports,
    final_function_check,
    function_check,
)
from src.config import LACEConfig
from src.nodes.cpu_resolver import resolve_cpu_state
from src.nodes.gates import retry_gate, route_gate
from src.nodes.rag_retriever import rag_retriever
from src.state_types import WorkflowState, ensure_state
from src.arithmetic_integrator import arithmetic_integrator
from src.writers import arithmetic_writer, interface_writer, insn_model_writer


def finished_interface_syntax_check(state: WorkflowState) -> str:
    """Determine next step after interface syntax check.

    Returns:
        "retry"  – current task failed, retry interface_writer.
        "next"   – current task passed and more tasks remain, go back to rag_retriever.
        "halt"   – a task exhausted its retry budget.
        "done"   – all tasks passed, proceed to arithmetic generation.
    """
    if state.needs_review:
        return "halt"
    if not state.interface_syntax_ok:
        return "retry"
    # Syntax passed – check whether more HDL tasks remain.
    # check_interface_syntax already incremented hdl_index on success,
    # so hdl_index < len(hdl_tasks) means there are still tasks to process.
    if state.hdl_index < len(state.hdl_tasks):
        return "next"
    return "done"


def finished_original_function(state: WorkflowState) -> str:
    """Determine next step after original function check."""
    if state.needs_review or state.formal_skipped:
        return "stop"
    if not state.function_ok:
        return "redo"
    return "done"


def spec2op_gate(state: WorkflowState) -> WorkflowState:
    return retry_gate(state, "spec_to_ops", "spec_retry_count")


def candidate_gate(state: WorkflowState) -> WorkflowState:
    return retry_gate(state, "candidate_modules", "candidate_retry_count")


def op2hdl_gate(state: WorkflowState) -> WorkflowState:
    return retry_gate(state, "op_to_hdl_tasks", "hdl_retry_count")


def route_spec2op_gate(state: WorkflowState) -> str:
    return route_gate(state, "spec_to_ops")


def route_candidate_gate(state: WorkflowState) -> str:
    return route_gate(state, "candidate_modules")


def route_op2hdl_gate(state: WorkflowState) -> str:
    outcome = route_gate(state, "op_to_hdl_tasks")
    if outcome != "continue":
        return outcome
    if state.op_index + 1 < len(state.ops):
        return "next"
    return "continue"


def advance_op(state: WorkflowState) -> WorkflowState:
    """Advance the planner cursor to the next micro-operation."""
    return state.model_copy(
        update={
            "op_index": state.op_index + 1,
            "hdl_retry_count": 0,
            "retry_stage": "",
            "last_error": "",
        }
    )


def route_arithmetic_syntax(state: WorkflowState) -> str:
    """Do not integrate arithmetic RTL that failed its syntax gate."""
    if state.needs_review:
        return "halt"
    if not state.arithmetic_syntax_ok:
        return "retry"
    return "continue"


def route_arithmetic_integration(state: WorkflowState) -> str:
    """Stop before semantic/formal checks when integrated RTL failed lint."""
    return "halt" if state.needs_review or not state.interface_syntax_ok else "continue"


def formal_gate(state: WorkflowState) -> WorkflowState:
    return retry_gate(
        state,
        "formal",
        "formal_retry_count",
        max_retries=LACEConfig.MAX_FORMAL_RETRIES,
    )


def route_formal_gate(state: WorkflowState) -> str:
    return route_gate(state, "formal")


def _node_wrapper(fn):
    """Wrap a node function so it only returns modified fields (dict).

    This keeps checkpoints compact when node functions return full
    WorkflowState instances.
    """
    # These stage markers are persisted separately by the runner.
    _EXCLUDED = {"last_stage", "last_checkpoint"}

    def wrapped(state):
        old = ensure_state(state)
        new = fn(old)
        new = ensure_state(new)
        diff = {}
        for field in type(old).model_fields:
            if field in _EXCLUDED:
                continue
            old_val = getattr(old, field)
            new_val = getattr(new, field)
            if old_val != new_val:
                diff[field] = new_val
        return diff
    return wrapped


builder = StateGraph(WorkflowState)

builder.add_node("cpu_resolver", _node_wrapper(resolve_cpu_state))
builder.add_node("spec2op_agent", _node_wrapper(spec_to_ops))
builder.add_node("spec2op_gate", _node_wrapper(spec2op_gate))
builder.add_node("cpu_structure_analyzer", _node_wrapper(analyze_cpu_structure_agent))
builder.add_node("candidate_module_selector", _node_wrapper(select_candidate_modules))
builder.add_node("candidate_gate", _node_wrapper(candidate_gate))
builder.add_node("op2hdl_planner", _node_wrapper(op_to_hdl_tasks))
builder.add_node("op2hdl_gate", _node_wrapper(op2hdl_gate))
builder.add_node("advance_op", _node_wrapper(advance_op))
builder.add_node("rag_retriever", _node_wrapper(rag_retriever))
builder.add_node("interface_writer", _node_wrapper(interface_writer))
builder.add_node("interface_syntax_check", _node_wrapper(check_interface_syntax))
builder.add_node("arithmetic_writer", _node_wrapper(arithmetic_writer))
builder.add_node("check_arithmetic_syntax", _node_wrapper(check_arithmetic_syntax))
builder.add_node("arithmetic_integrator", _node_wrapper(arithmetic_integrator))
builder.add_node("semantic_port_check", _node_wrapper(check_semantic_ports))
builder.add_node("original_function_checker", _node_wrapper(function_check))
builder.add_node("insn_model_writer", _node_wrapper(insn_model_writer))
builder.add_node("final_function_checker", _node_wrapper(final_function_check))
builder.add_node("formal_gate", _node_wrapper(formal_gate))

builder.add_edge(START, "cpu_resolver")
builder.add_edge("cpu_resolver", "spec2op_agent")
builder.add_edge("spec2op_agent", "spec2op_gate")
builder.add_conditional_edges(
    "spec2op_gate",
    route_spec2op_gate,
    {"retry": "spec2op_agent", "continue": "cpu_structure_analyzer", "stop": END},
)
builder.add_edge("cpu_structure_analyzer", "op2hdl_planner")
builder.add_edge("op2hdl_planner", "op2hdl_gate")
builder.add_conditional_edges(
    "op2hdl_gate",
    route_op2hdl_gate,
    {
        "retry": "op2hdl_planner",
        "next": "advance_op",
        "continue": "candidate_module_selector",
        "stop": END,
    },
)
builder.add_edge("advance_op", "op2hdl_planner")
builder.add_edge("candidate_module_selector", "candidate_gate")
builder.add_conditional_edges(
    "candidate_gate",
    route_candidate_gate,
    {"retry": "candidate_module_selector", "continue": "rag_retriever", "stop": END},
)

# Code generation is intentionally serial. This gives every gate a single
# predecessor and prevents duplicate join execution or sibling-state races.
builder.add_edge("rag_retriever", "interface_writer")
builder.add_edge("interface_writer", "interface_syntax_check")

# Interface retry / next-task loop
builder.add_conditional_edges(
    "interface_syntax_check",
    finished_interface_syntax_check,
    {
        "done": "arithmetic_writer",
        "halt": END,
        "retry": "interface_writer",
        "next": "rag_retriever",
    },
)

# Arithmetic: write → syntax-check → integrate.
builder.add_edge("arithmetic_writer", "check_arithmetic_syntax")
builder.add_conditional_edges(
    "check_arithmetic_syntax",
    route_arithmetic_syntax,
    {"continue": "arithmetic_integrator", "retry": "arithmetic_writer", "halt": END},
)
builder.add_conditional_edges(
    "arithmetic_integrator",
    route_arithmetic_integration,
    {"continue": "semantic_port_check", "halt": END},
)

builder.add_edge("semantic_port_check", "original_function_checker")

builder.add_conditional_edges(
    "original_function_checker",
    finished_original_function,
    {
        "done": "insn_model_writer",
        "redo": "interface_writer",
        "stop": END,
    },
)

builder.add_edge("insn_model_writer", "final_function_checker")

builder.add_edge("final_function_checker", "formal_gate")

builder.add_conditional_edges(
    "formal_gate",
    route_formal_gate,
    {"retry": "final_function_checker", "done": END, "stop": END, "continue": END},
)

graph = builder.compile()
