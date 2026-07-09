"""Writer agents for HDL code generation.

These functions run in **Auto Mode**: they assemble prompts via
`src.interactive_engine`, invoke the LLM, parse the response, and write
generated code back to the workspace.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from typing import Any

from src.interactive_engine import (
    _parse_model_response as parse_model_response,
    build_arithmetic_prompt,
    build_insn_model_prompt,
    build_interface_prompt,
    merge_arithmetic_result,
    merge_insn_model_result,
    merge_interface_result,
)
from src.llm import get_chat_model, get_structured_runnable
from src.nodes.agent_runner import invoke_with_backoff
from src.config import LACEConfig
from src.state_types import ArithmeticExprsOut, WorkflowState, ensure_state


def _coerce_model_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if isinstance(item, dict) and "text" in item and isinstance(item["text"], str):
            parts.append(item["text"])
            continue
        parts.append(str(item))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# rg_tools wrappers for LLM tool-calling
# ---------------------------------------------------------------------------

def _make_rg_tools(cpu_dir: str):
    """Create LangChain tool callables bound to a specific CPU directory."""
    try:
        from langchain_core.tools import tool
    except Exception:
        # Fallback: if langchain tools unavailable, return empty list
        return []

    @tool
    def rg_search(query: str, top_k: int = 5) -> str:
        """Search CPU RTL source code for code blocks matching a natural language query."""
        from src import rg_tools as rg

        blocks = rg.get_similar_block(query, cpu_dir, top_k=top_k)
        if not blocks:
            return f"No results for query: {query}"
        lines = [f"Results for: {query}"]
        for b in blocks:
            header = f"--- {Path(b['filename']).name} (lines {b['begin']}-{b['end']}) ---"
            lines.append(f"{header}\n{b['text']}")
        return "\n".join(lines)

    @tool
    def rg_get_signal(signal_name: str) -> str:
        """Find declarations and assignments of a specific signal in the CPU RTL."""
        from src import rg_tools as rg

        blocks = rg.get_signal_by_name(signal_name, cpu_dir)
        if not blocks:
            return f"No occurrences of '{signal_name}' found."
        lines = [f"Signal occurrences: {signal_name}"]
        for b in blocks[:10]:
            lines.append(f"  {b['filename']}:{b['line']} | {b['text']}")
        return "\n".join(lines)

    return [rg_search, rg_get_signal]


def _execute_tool_call(tc: dict[str, Any], tools: list[Any]) -> str:
    """Execute a single tool call and return its string result."""
    tool_name = tc.get("name")
    tool_args = tc.get("args", {})
    if not tool_name:
        return "Error: tool call missing name"
    for t in tools:
        if getattr(t, "name", None) == tool_name:
            try:
                return str(t.invoke(tool_args))
            except Exception as exc:
                return f"Error executing {tool_name}: {exc}"
    return f"Tool '{tool_name}' not found"


def interface_writer(state: WorkflowState | dict[str, Any]) -> WorkflowState:
    """Write interface code for the current op (Auto Mode).

    The LLM receives the full source file and returns SEARCH/REPLACE blocks.
    All tasks for the current op are presented in a single prompt.
    """
    state = ensure_state(state)
    ops = state.ops
    if not ops:
        return state

    hdl_tasks = state.hdl_tasks
    if state.hdl_index >= len(hdl_tasks):
        return state

    prompt = build_interface_prompt(state)

    # Include the full original file so the LLM can generate accurate
    # SEARCH/REPLACE blocks. The response (only the changed blocks) stays
    # small even though the input is large.
    human = prompt["human"]
    original_code = prompt.get("original_code")

    # Build the list of extension interface wires expected by the arithmetic
    # module, based on ALL operations used by this instruction.  Even when the
    # current interface_writer step is handling a single op, the LLM must know
    # every wire that the final `lace_arithmetic` instance will connect to, so
    # it does not remove connections that belong to other ops.
    wire_lines: list[str] = []
    op_names = {op.split("(")[0].strip() for op in (ops or [])}
    if "RdInstr" in op_names:
        wire_lines.append("wire [31:0] RdInstr_0_o  // driven by fetched instruction word")
    if "RdRS1" in op_names:
        wire_lines.append("wire [31:0] RdRS1_1_o  // driven by rs1 register read value")
    if "RdRS2" in op_names:
        wire_lines.append("wire [31:0] RdRS2_1_o  // driven by rs2 register read value")
    if "WrRD" in op_names:
        wire_lines.append("wire [31:0] WrRD_2_i  // used in register writeback mux")
        wire_lines.append("wire        WrRD_validReq_2_i  // used to select extension result")
    if "WrPC" in op_names:
        wire_lines.append("wire [31:0] WrPC_3_i")
        wire_lines.append("wire        WrPC_validReq_3_i")
    if "RdPC" in op_names:
        wire_lines.append("wire [31:0] RdPC_0_o")
    if "RdMem" in op_names:
        wire_lines.append("wire [31:0] RdMem_2_o")
        wire_lines.append("wire [31:0] RdMem_addr_2_i")
        wire_lines.append("wire        RdMem_validReq_2_i")
        wire_lines.append("wire        RdMem_addr_valid_2_i")
    if "WrMem" in op_names:
        wire_lines.append("wire [31:0] WrMem_2_i")
        wire_lines.append("wire        WrMem_validReq_2_i")

    if original_code:
        human += (
            "\n\n## Required Extension Interface Wires\n"
            "Create these as INTERNAL WIRES inside the CPU module body. "
            "Do NOT add them to the module port list under any circumstances. The CPU's external interface must remain exactly the same. "
            "The `lace_arithmetic` module will be instantiated later and connected to these wires.\n"
            "Examples:\n"
            "  wire [31:0] RdInstr_0_o = <cpu_internal_instruction_signal>;\n"
            "  wire [31:0] RdRS1_1_o = <cpu_internal_rs1_value>;\n"
            "  wire [31:0] WrRD_2_i;\n"
            "  wire        WrRD_validReq_2_i;\n"
        )
        for line in wire_lines:
            human += f"- {line}\n"
        human += (
            "\n## Important\n"
            "If any wire listed above is missing from the code, ADD it. "
            "Do NOT remove existing connections from the `lace_arithmetic` instance, "
            "even if the corresponding wire is not part of the current task. "
            "All wires above must eventually exist for the arithmetic module.\n"
            "\n## Original Code to Modify\n"
            "```verilog\n"
            + original_code
            + "\n```\n"
            "\nGenerate SEARCH/REPLACE blocks that match the original code EXACTLY."
        )

    messages: list[Any] = [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=human),
    ]

    # Some OpenAI-compatible endpoints (e.g. mimo) sporadically reject
    # requests with a reasoning_content error.  We retry with a fresh
    # model instance when that happens.
    def _safe_invoke(msgs: list[Any]) -> Any:
        for attempt in range(3):
            m = get_chat_model()
            try:
                return m.invoke(msgs)
            except Exception as exc:
                if "reasoning_content" in str(exc) and attempt < 2:
                    continue
                raise
        return m.invoke(msgs)

    response = _safe_invoke(messages)
    content = _coerce_model_content(response.content)

    # Debug: log LLM raw response
    debug_path = Path(f"/tmp/lace_llm_response_{state.hdl_index}.log")
    with open(debug_path, "a", encoding="utf-8") as f:
        f.write(f"=== hdl_index={state.hdl_index} ===\n")
        f.write(content)
        f.write("\n\n")

    return merge_interface_result(state, content)


def arithmetic_writer(state: WorkflowState | dict[str, Any]) -> WorkflowState:
    """Write arithmetic implementation code (Auto Mode).

    The LLM generates the complete lace_arithmetic Verilog module.
    No regex injection — the LLM outputs the full module directly.
    """
    state = ensure_state(state)
    ops = state.ops
    if not ops:
        return state

    # Clear any retry signal from the preceding syntax check so the graph
    # doesn't loop forever after a successful rewrite.
    if state.retry_stage == "arithmetic_writer":
        state = state.model_copy(update={"retry_stage": ""})

    prompt = build_arithmetic_prompt(state)

    # Include the skeleton in the human prompt as a reference for port names.
    human = prompt["human"]
    skeleton = prompt.get("skeleton")
    if skeleton:
        human += (
            "\n\n## Module Skeleton (reference for port names and structure)\n"
            "```verilog\n"
            + skeleton
            + "\n```\n"
            "\nGenerate the COMPLETE module above, filling in the decode and computation logic."
        )

    model = get_chat_model()
    messages = [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=human),
    ]

    response = model.invoke(messages)
    content = _coerce_model_content(response.content)

    # Debug: log generated arithmetic code
    debug_path = Path("/tmp/lace_arithmetic_debug.log")
    with open(debug_path, "a", encoding="utf-8") as f:
        f.write("=== arithmetic ===\n")
        f.write(f"generated_code:\n{content}\n\n")

    return merge_arithmetic_result(state, content)


def insn_model_writer(state: WorkflowState | dict[str, Any]) -> WorkflowState:
    """Generate riscv-formal instruction model Verilog (Auto Mode).

    The LLM generates a ``rvfi_insn_*`` module based on the instruction spec.
    The instruction name is extracted from the generated module declaration.
    """
    state = ensure_state(state)
    if not state.spec:
        return state

    prompt = build_insn_model_prompt(state)

    model = get_chat_model()
    messages = [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=prompt["human"]),
    ]

    response = model.invoke(messages)
    content = _coerce_model_content(response.content)

    # Debug: log generated insn model
    debug_path = Path("/tmp/lace_insn_model_debug.log")
    with open(debug_path, "a", encoding="utf-8") as f:
        f.write("=== insn_model ===\n")
        f.write(f"spec: {state.spec[:200]}\n")
        f.write(f"generated_code:\n{content}\n\n")

    riscv_formal_dir = Path(LACEConfig.RISCV_FORMAL_DIR)
    if not riscv_formal_dir.is_absolute():
        riscv_formal_dir = Path.cwd() / riscv_formal_dir

    return merge_insn_model_result(state, content, riscv_formal_dir)
