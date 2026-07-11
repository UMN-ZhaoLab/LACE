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
        "done"   – all tasks done or needs_review is set, proceed to semantic_port_check.
    """
    if state.needs_review:
        return "done"
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
    if state.needs_review:
        return "done"
    # A formal skip is not a code defect — redoing the interface writer will
    # not conjure an sby toolchain into existence. Proceed to the next stage
    # and let the final checker escalate the skip into a review.
    if state.formal_skipped:
        return "done"
    if not state.function_ok:
        return "redo"
    if state.advance_op:
        return "continue"
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
    return route_gate(state, "op_to_hdl_tasks")


def formal_gate(state: WorkflowState) -> WorkflowState:
    return retry_gate(state, "formal", "formal_retry_count")


def route_formal_gate(state: WorkflowState) -> str:
    return route_gate(state, "formal")


def dispatch(state: WorkflowState) -> WorkflowState:
    """No-op dispatch node to fork parallel interface + arithmetic execution."""
    return state


def _node_wrapper(fn):
    """Wrap a node function so it only returns modified fields (dict).

    This prevents LangGraph checkpointer from seeing concurrent updates
    to unchanged fields (e.g. 'spec') when parallel nodes both return
    full WorkflowState instances.
    """
    # These fields are "stage markers" written by every node and cause
    # collisions when parallel branches both touch them.
    _EXCLUDED = {"last_stage", "last_checkpoint"}

    def wrapped(state):
        old = ensure_state(state)
        new = fn(old)
        new = ensure_state(new)
        diff = {}
        for field in old.model_fields:
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
builder.add_node("dispatch", _node_wrapper(dispatch))
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
builder.add_edge("cpu_resolver", "cpu_structure_analyzer")
builder.add_edge("spec2op_agent", "spec2op_gate")
builder.add_edge("cpu_structure_analyzer", "candidate_module_selector")
builder.add_conditional_edges(
    "spec2op_gate",
    route_spec2op_gate,
    {"retry": "spec2op_agent", "continue": "candidate_module_selector", "stop": END},
)
builder.add_edge("candidate_module_selector", "candidate_gate")
builder.add_conditional_edges(
    "candidate_gate",
    route_candidate_gate,
    {"retry": "candidate_module_selector", "continue": "op2hdl_planner", "stop": END},
)
builder.add_edge("op2hdl_planner", "op2hdl_gate")
builder.add_conditional_edges(
    "op2hdl_gate",
    route_op2hdl_gate,
    {"retry": "op2hdl_planner", "continue": "dispatch", "stop": END},
)

# Parallel fork: rag_retriever (for interface) + arithmetic_writer
builder.add_edge("dispatch", "rag_retriever")
builder.add_edge("dispatch", "arithmetic_writer")
builder.add_edge("rag_retriever", "interface_writer")
builder.add_edge("interface_writer", "interface_syntax_check")
# Arithmetic: write → syntax-check → integrate (no retry edge — the parallel
# fork with the interface branch makes a back-edge unsafe; failures escalate
# to needs_review and halt).
builder.add_edge("arithmetic_writer", "check_arithmetic_syntax")
builder.add_edge("check_arithmetic_syntax", "arithmetic_integrator")

# Interface retry / next-task loop
builder.add_conditional_edges(
    "interface_syntax_check",
    finished_interface_syntax_check,
    {
        "done": "semantic_port_check",
        "retry": "interface_writer",
        "next": "rag_retriever",
    },
)

# Arithmetic joins at semantic_port_check via integrator
builder.add_edge("arithmetic_integrator", "semantic_port_check")

builder.add_edge("semantic_port_check", "original_function_checker")

builder.add_conditional_edges(
    "original_function_checker",
    finished_original_function,
    {
        "done": "insn_model_writer",
        "continue": "candidate_module_selector",
        "redo": "interface_writer",
    },
)

builder.add_edge("insn_model_writer", "final_function_checker")

builder.add_edge("final_function_checker", "formal_gate")

builder.add_conditional_edges(
    "formal_gate",
    route_formal_gate,
    {"retry": "interface_writer", "done": END, "stop": END, "continue": END},
)

graph = builder.compile()
