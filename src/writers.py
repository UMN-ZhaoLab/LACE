"""Writer agents for HDL code generation.

These functions run in **Auto Mode**: they assemble prompts via
`src.interactive_engine`, invoke the LLM, parse the response, and write
generated code back to the workspace.
"""

from __future__ import annotations

import logging
import json
import re
import subprocess
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
from src.formal.sandbox import prepare_riscv_formal_sandbox
from src.state_types import ArithmeticExprsOut, WorkflowState, ensure_state

logger = logging.getLogger("lace.writer")


_DISCOVERY_SYSTEM_PROMPT = """
You are an RTL integration analyst.  Before any HDL is edited, select the
actual CPU-local sites that implement a custom RISC-V instruction.  Do not
invent identifiers, ports, stages, or files.  Use only identifiers visible in
the supplied source excerpts.

Return JSON only, with these objects: decode, writeback, timing, and (when the
operation requires them) rs1 and rs2.  Every object must contain:
  file: workspace-relative .v/.sv path
  lines: [first_line, last_line]
  signals: non-empty list of identifiers copied from the excerpt
  excerpt: exact contiguous source text copied from that file

`decode` must identify instruction decode and illegal-instruction handling.
`rs1`/`rs2` must identify operands belonging to the same executing instruction,
not merely a generic register-file port.  `writeback` must identify the normal
rd data, rd address, and write-enable/valid path.  `timing` must identify clock,
reset, and any stall/flush/exception condition relevant to that path.

If the excerpts do not prove a required item, return {"status":"insufficient",
"missing":[...]}.  Never guess.
"""


def _required_evidence_keys(ops: list[str]) -> set[str]:
    names = {op.split("(", 1)[0].split("=", 1)[-1].strip() for op in ops}
    required = {"decode", "writeback", "timing"}
    if "RdRS1" in names:
        required.add("rs1")
    if "RdRS2" in names:
        required.add("rs2")
    return required


def _parse_discovery_evidence(
    content: str, workspace: Path, required: set[str]
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate that every CPU-local identifier is backed by source text."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL)
    try:
        evidence = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, f"RTL discovery did not return JSON: {exc}"
    if not isinstance(evidence, dict):
        return None, "RTL discovery must return a JSON object"
    if evidence.get("status") == "insufficient":
        missing = evidence.get("missing", [])
        return None, f"RTL discovery lacks required source evidence: {missing}"
    missing = sorted(required - evidence.keys())
    if missing:
        return None, f"RTL discovery omitted required evidence: {missing}"

    for key in sorted(required):
        item = evidence.get(key)
        if not isinstance(item, dict):
            return None, f"RTL discovery field '{key}' must be an object"
        file_name = item.get("file")
        excerpt = item.get("excerpt")
        signals = item.get("signals")
        lines = item.get("lines")
        if not isinstance(file_name, str) or not isinstance(excerpt, str) or not excerpt.strip():
            return None, f"RTL discovery field '{key}' lacks file/excerpt"
        if not isinstance(signals, list) or not signals or not all(isinstance(s, str) for s in signals):
            return None, f"RTL discovery field '{key}' lacks signals"
        if not (isinstance(lines, list) and len(lines) == 2 and all(isinstance(n, int) for n in lines)):
            return None, f"RTL discovery field '{key}' lacks line range"
        path = (workspace / file_name).resolve()
        if workspace.resolve() not in path.parents or path.suffix not in {".v", ".sv"} or not path.exists():
            return None, f"RTL discovery field '{key}' references invalid file '{file_name}'"
        source = path.read_text(encoding="utf-8")
        if excerpt not in source:
            return None, f"RTL discovery excerpt for '{key}' is not present in '{file_name}'"
        for signal in signals:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", signal):
                return None, f"RTL discovery signal '{signal}' is not a Verilog identifier"
            if not re.search(rf"\b{re.escape(signal)}\b", excerpt):
                return None, f"RTL discovery signal '{signal}' is not present in its excerpt"
    return evidence, None


def _discover_integration_evidence(state: WorkflowState, prompt: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    target_dir = state.workspace_dir or state.cpu_dir
    if not target_dir:
        return None, "RTL discovery has no CPU workspace"
    required = _required_evidence_keys(state.ops)
    human = (
        "## Required integration evidence\n"
        + ", ".join(sorted(required))
        + "\n\n## Instruction specification\n"
        + (state.spec or "")
        + "\n\n## Current HDL tasks\n"
        + "\n".join(prompt.get("op_tasks", []))
        + "\n\n## Source discovery excerpts\n"
        + (state.relevant_code or "(no excerpts available)")
    )
    messages = [
        SystemMessage(content=_DISCOVERY_SYSTEM_PROMPT),
        HumanMessage(content=human),
    ]
    response = get_chat_model().invoke(messages)
    content = _coerce_model_content(response.content)
    evidence, error = _parse_discovery_evidence(content, Path(target_dir), required)
    if evidence is not None:
        return evidence, None

    # An honest "insufficient" response should trigger one wider, generic
    # source search rather than forcing the model to guess.  The search terms
    # describe RTL roles only; they do not encode any CPU's signal mapping.
    try:
        parsed = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.DOTALL))
    except json.JSONDecodeError:
        parsed = {}
    if not isinstance(parsed, dict) or parsed.get("status") != "insufficient":
        return None, error

    expanded = _expand_discovery_context(Path(target_dir))
    if not expanded:
        return None, error
    retry_messages = [
        SystemMessage(content=_DISCOVERY_SYSTEM_PROMPT),
        HumanMessage(content=human + "\n\n## Expanded RTL search results\n" + expanded),
    ]
    retry_response = get_chat_model().invoke(retry_messages)
    retry_content = _coerce_model_content(retry_response.content)
    return _parse_discovery_evidence(retry_content, Path(target_dir), required)


def _expand_discovery_context(workspace: Path) -> str:
    """Fetch additional generic RTL-role contexts after a proven shortfall."""
    rtl_root = workspace / "rtl"
    root = rtl_root if rtl_root.exists() else workspace
    role_patterns = (
        r"opcode|funct|unique\s+case|\bcase\b|illegal_insn",
        r"rf_wdata|rf_waddr|rf_we|writeback",
        r"operand_[ab]|rs1|rs2|rdata",
        r"stall|flush|exception|\brst|\bclk",
    )
    files = [*root.rglob("*.v"), *root.rglob("*.sv")]
    snippets: list[str] = []
    for pattern in role_patterns:
        emitted = 0
        for path in sorted(files):
            try:
                proc = subprocess.run(
                    ["rg", "--line-number", "--color=never", "-i", pattern, str(path)],
                    capture_output=True, text=True, check=False,
                )
            except FileNotFoundError:
                return ""
            if proc.returncode not in (0, 1):
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            for match in proc.stdout.splitlines():
                number = match.split(":", 1)[0]
                if not number.isdigit():
                    continue
                index = int(number) - 1
                start, end = max(0, index - 5), min(len(lines), index + 9)
                snippets.append(
                    f"### {path.relative_to(workspace)}:{start + 1}-{end}\n"
                    + "\n".join(lines[start:end])
                )
                emitted += 1
                if emitted >= 5:
                    break
            if emitted >= 5:
                break
    return "\n\n".join(snippets)


def _evidence_source_files(state: WorkflowState, evidence: dict[str, Any]) -> str:
    """Return complete, selected source files for source-grounded patching."""
    root = Path(state.workspace_dir or state.cpu_dir)
    files = sorted({item["file"] for item in evidence.values() if isinstance(item, dict) and "file" in item})
    parts: list[str] = []
    for file_name in files:
        path = root / file_name
        source = path.read_text(encoding="utf-8")
        parts.append(f"### FILE: {file_name}\n```verilog\n{source}\n```")
    return "\n\n".join(parts)


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

    evidence, discovery_error = _discover_integration_evidence(state, prompt)
    if discovery_error or evidence is None:
        notes = list(state.notes)
        notes.append(discovery_error or "RTL discovery failed")
        return state.model_copy(update={
            "needs_review": True,
            "last_error": discovery_error or "RTL discovery failed",
            "notes": notes,
        })

    # The patch writer receives only source files selected by the discovery
    # evidence.  This keeps it grounded in CPU-local identifiers and allows it
    # to modify a decoder child module instead of guessing ports on the top.
    human = prompt["human"]
    original_code = prompt.get("original_code")
    human += (
        "\n\n## Verified CPU Integration Evidence\n```json\n"
        + json.dumps(evidence, indent=2)
        + "\n```\n\n## Complete Selected Source Files\n"
        + _evidence_source_files(state, evidence)
        + "\n\nUse only CPU identifiers proven in the evidence above.  You may introduce "
        "only the fixed LACE interface identifiers.  Do not add a connection to "
        "an existing module instance unless that destination port is visible in "
        "the selected module declaration."
    )

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
            "\n## Patch Format\n"
            "For each modified selected source file, prefix its SEARCH/REPLACE blocks with "
            "`FILE: <workspace-relative path>`.  The path must be one of the selected "
            "source files.  Generate SEARCH text that matches that file exactly."
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

    logger.debug("interface_writer hdl_index=%s response_len=%d", state.hdl_index, len(content))

    return merge_interface_result(state, content, evidence=evidence)


def arithmetic_writer(state: WorkflowState | dict[str, Any]) -> WorkflowState:
    """Write arithmetic implementation code (Auto Mode).

    The LLM generates the complete lace_arithmetic Verilog module.
    No regex injection — the LLM outputs the full module directly.
    """
    state = ensure_state(state)
    ops = state.ops
    if not ops:
        return state

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

    logger.debug("arithmetic_writer response_len=%d", len(content))

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

    logger.debug("insn_model_writer spec_len=%d response_len=%d", len(state.spec), len(content))

    try:
        riscv_formal_dir = prepare_riscv_formal_sandbox(
            run_id=state.run_id,
            cpu_name=state.cpu_name,
            workspace_dir=state.workspace_dir,
        )
    except Exception as exc:
        notes = list(state.notes)
        notes.append(f"Instruction-model sandbox setup failed: {exc}")
        return state.model_copy(
            update={
                "needs_review": True,
                "formal_terminal": True,
                "last_error": f"Instruction-model sandbox setup failed: {exc}",
                "notes": notes,
            }
        )

    return merge_insn_model_result(state, content, riscv_formal_dir)
