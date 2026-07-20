import re
from src.ops_registry import PREDEFINED_OPS_SET

system_prompt = """
You are a RISC-V architecture expert specializing in instruction set design and CPU microarchitecture.
Your task is to analyze a CPU's microarchitecture and generate concrete HDL modification tasks to integrate a custom ISA extension.

You will be given:
- The complete list of interface-level micro-operations for the instruction
- The CPU's microarchitecture summary (pipeline stages, key modules, signal names)
- The instruction's encoding (fixed bits to match in decode)

For each operation, generate 1-3 concrete HDL tasks. Each task must:
1. Specify the EXACT LACE extension interface port/signal to create or connect
2. Use the CPU summary to choose the CPU-internal signal name (e.g., `instr`, `mem_rdata_q`, `reg_rdata1`)
3. Describe the modification location in the CPU

The LACE extension interface uses these fixed port names:
- RdInstr_0_o [31:0] : fetched instruction (output from CPU)
- RdRS1_1_o   [31:0] : rs1 register read value (output from CPU)
- RdRS2_1_o   [31:0] : rs2 register read value (output from CPU)
- RdPC_0_o    [31:0] : current PC (output from CPU)
- RdMem_2_o   [31:0] : memory read data (output from CPU)
- WrRD_2_i    [31:0] : extension result to write to rd (input to CPU)
- WrRD_validReq_2_i  : extension result valid (input to CPU)
- WrPC_3_i    [31:0] : extension next PC (input to CPU)
- WrPC_validReq_3_i  : extension next PC valid (input to CPU)
- WrMem_2_i   [31:0] : extension memory write data (input to CPU)
- WrMem_validReq_2_i : extension memory write valid (input to CPU)

For decode logic, create a signal named `ISAX_isisax` (or `instr_<name>` following the CPU's naming convention) that asserts when the fixed encoding bits match.

GENERAL INTEGRATION METHODOLOGY (do NOT hard-code CPU-specific signal names; derive them from the CPU summary and source code):
1. Locate existing instruction decode: Search the CPU source for all always block(s) or assignments that set instruction decode registers (signals typically named `instr_<mnemonic>`). Add the custom instruction decode in every source-proven path that can execute the instruction.
2. Locate register read data paths: Search for where the CPU reads source registers (rs1/rs2). Reuse those existing data paths; do NOT create parallel read ports.
3. Locate result/writeback path: Search for the signal or register that feeds the register-file write data. Route extension results through that existing path and reuse the existing register-file write enable.
4. Check for encoding collisions: If the CPU already has custom instructions in the same opcode/funct7 space, add exclusion conditions to the existing decode logic so the new instruction does not overlap.
5. Check RVFI overlap: Search the CPU source for any RVFI patterns that match existing instructions. If the new instruction would overlap those patterns, update them with the same exclusion conditions used for decode. Otherwise, do NOT modify RVFI signals.

Return a confidence level (high/medium/low).
"""

from src.ops_registry import format_predefined_ops
predefined_ops = format_predefined_ops(include_arithmetic=False)

# Mapping from operation name to concrete HDL task templates.
# Each template is a function that takes (encoding_hint, cpu_summary_hint, decode_expr) -> list of task strings.
OP_TASK_TEMPLATES = {
    "RdInstr": lambda enc, cpu, dec: [
        f"Create an internal wire `RdInstr_0_o [31:0]` inside the CPU module and assign it to the source-proven instruction word that is stable during execution/retirement.",
        f"Add decode logic that creates an internal wire or register asserting when the fetched instruction matches the fixed encoding bits: {enc}. Use this exact decode expression template: `{dec}`. Compare only opcode/funct3/funct7/imm fields; do NOT compare rs1, rs2, or rd. Locate every source-proven decode site that can execute the instruction and add the custom decode there.",
        f"Check whether the target CPU already uses the same opcode/funct7 space for its own custom instructions. Search the CPU source for existing `instr_*` decode signals that match the same opcode. If any exist, add exclusion conditions (e.g., on funct3) to those existing decode expressions so the new instruction does not collide with them.",
        f"Check RVFI overlap: search the CPU source for any RVFI `casez` patterns that match existing custom instructions. If the new instruction would overlap those patterns, add the same exclusion conditions to the RVFI patterns.",
        f"Ensure the custom instruction does not trigger an illegal-instruction exception. Modify the trap/exception logic to suppress the illegal-instruction trap when the custom decode signal is asserted.",
    ],
    "RdRS1": lambda enc, cpu, dec: [
        f"Create an internal wire `RdRS1_1_o [31:0]` inside the CPU module and assign it to the source-proven rs1 value for the executing instruction. Reuse the CPU's existing rs1 read path; do NOT create a separate read port.",
    ],
    "RdRS2": lambda enc, cpu, dec: [
        f"Create an internal wire `RdRS2_1_o [31:0]` inside the CPU module and assign it to the source-proven rs2 value for the executing instruction. Reuse the CPU's existing rs2 read path; do NOT create a separate read port.",
    ],
    "RdPC": lambda enc, cpu, dec: [
        f"Create an internal wire `RdPC_0_o [31:0]` inside the CPU module and assign it to the current program counter value. Based on the CPU summary, connect it to the PC register or fetch-stage PC signal.",
    ],
    "RdMem": lambda enc, cpu, dec: [
        f"Create an internal wire `RdMem_2_o [31:0]` inside the CPU module and assign it to the memory read data. Based on the CPU summary, connect it to the load data path or memory response register.",
    ],
    "RdCustReg": lambda enc, cpu, dec: [
        f"Create an internal wire `RdCustReg_1_o [31:0]` inside the CPU module and assign it to the custom register file read value. Based on the CPU summary, connect it to the custom register file read port.",
    ],
    "WrRD": lambda enc, cpu, dec: [
        f"Create internal wires `WrRD_2_i [31:0]` and `WrRD_validReq_2_i` inside the CPU module. Route `WrRD_2_i` through the source-proven normal rd writeback path. Do NOT create a separate write-enable or bypass logic; reuse the existing register-file write logic.",
    ],
    "WrPC": lambda enc, cpu, dec: [
        f"Create internal wires `WrPC_3_i [31:0]` and `WrPC_validReq_3_i` inside the CPU module. Modify the program counter update logic so that when `WrPC_validReq_3_i` is asserted, the next PC is taken from `WrPC_3_i`. Based on the CPU summary, locate the PC update mux or branch resolution logic.",
    ],
    "WrMem": lambda enc, cpu, dec: [
        f"Create internal wires `WrMem_2_i [31:0]` and `WrMem_validReq_2_i` inside the CPU module. Modify the store data path so that when `WrMem_validReq_2_i` is asserted, the memory write data comes from `WrMem_2_i`. Based on the CPU summary, locate the store data mux or memory interface.",
    ],
    "WrCustReg": lambda enc, cpu, dec: [
        f"Create internal wires `WrCustReg_2_i [31:0]` and `WrCustReg_validReq_2_i` inside the CPU module. Modify the custom register file write path so that when `WrCustReg_validReq_2_i` is asserted, the selected custom register is written with `WrRD_2_i`.",
    ],
}


def _extract_encoding_hint(spec: str) -> str:
    """Extract fixed encoding bits from spec for decode logic.

    Returns a concise string like "opcode=0110011, funct3=001, funct7=0110000"
    or the full spec if no explicit encoding fields are found.
    """
    import re as _re
    hints = []
    # Match patterns like "opcode=0110011" or "funct3 [14:12] = 001"
    for match in _re.finditer(
        r"(opcode|funct3|funct7|funct2|funct6|bs)\s*(?:\[\d+\s*:\s*\d+\])?\s*[=:]\s*([01x]+)",
        spec, _re.IGNORECASE,
    ):
        field = match.group(1).lower()
        value = match.group(2)
        hints.append(f"{field}={value}")
    return ", ".join(hints) if hints else spec[:200]


def _build_decode_expression(spec: str, instr_signal: str = "RdInstr_0_i") -> str:
    """Build a Verilog decode expression template from the spec encoding fields.

    Args:
        spec: Instruction specification text.
        instr_signal: Name of the instruction signal to use in the expression.
            Use "RdInstr_0_i" for the arithmetic module, "RdInstr_0_o" for CPU decode.

    Returns a string like:
        (RdInstr_0_i[6:0] == 7'b0110011) && (RdInstr_0_i[14:12] == 3'b001) && (RdInstr_0_i[31:25] == 7'b0110000)
    or an empty string if no encoding fields are found.
    """
    import re as _re

    # Map field names to their bit ranges.  Ranges are (high, low) inclusive.
    field_ranges = {
        "opcode": (6, 0),
        "funct3": (14, 12),
        "funct7": (31, 25),
        "funct2": (26, 25),
        "funct6": (31, 26),
        "bs": (31, 30),
    }

    clauses: list[str] = []
    for match in _re.finditer(
        r"(opcode|funct3|funct7|funct2|funct6|bs)\s*(?:\[\d+\s*:\s*\d+\])?\s*[=:]\s*([01x]+)",
        spec, _re.IGNORECASE,
    ):
        field = match.group(1).lower()
        value = match.group(2)
        if field not in field_ranges:
            continue
        high, low = field_ranges[field]
        width = high - low + 1
        # Replace 'x' with Verilog '?' for don't-care bits.
        verilog_value = value.replace("x", "?")
        clauses.append(
            f"({instr_signal}[{high}:{low}] == {width}'b{verilog_value})"
        )

    return " && ".join(clauses) if clauses else ""


def get_prompt_for_op(ops: list[str], op_index: int, cpu_summary: str = "", spec: str = "") -> str:
    """Build a flexible prompt for op→HDL task planning.

    Args:
        ops: Full list of interface operations
        op_index: Current operation index
        cpu_summary: CPU microarchitecture summary
        spec: Instruction specification (for encoding extraction)
    """
    if not ops or op_index < 0 or op_index >= len(ops):
        return ""

    op_content = ops[op_index]
    encoding_hint = _extract_encoding_hint(spec)
    decode_expr = _build_decode_expression(spec, instr_signal="RdInstr_0_o")

    # Find which predefined op matches
    matched_op = None
    for op_name in sorted(PREDEFINED_OPS_SET, key=len, reverse=True):
        if re.search(rf"\b{re.escape(op_name)}\b", op_content):
            matched_op = op_name
            break

    parts: list[str] = []

    # CPU context
    parts.append("## CPU Microarchitecture")
    parts.append(cpu_summary if cpu_summary else "(No detailed CPU summary provided. Infer from standard RISC-V CPU structure.)")
    parts.append("")

    # Instruction encoding
    parts.append("## Instruction Specification (full text)")
    parts.append(spec if spec else "(No spec provided)")
    parts.append("")
    parts.append("## Instruction Encoding (fixed bits for decode)")
    parts.append(encoding_hint)
    if decode_expr:
        parts.append("")
        parts.append("## Decode Expression Template (use EXACTLY these bit ranges and values)")
        parts.append(decode_expr)
    parts.append("")

    # All operations context
    parts.append("## All Interface Operations for This Instruction")
    for i, op in enumerate(ops):
        marker = " <-- CURRENT" if i == op_index else ""
        parts.append(f"{i+1}. {op}{marker}")
    parts.append("")

    # Generate flexible tasks for the current op
    parts.append("## Generate HDL Tasks for This Operation")
    parts.append("Based on the CPU structure above, generate EXACTLY the following HDL modification tasks for this operation.")
    parts.append("Each task must specify: (1) WHAT to change, (2) WHERE in the CPU to make the change.")
    parts.append("Use the exact LACE port names from the system prompt (e.g., RdInstr_0_o, RdRS1_1_o, WrRD_2_i).")
    parts.append("Use the CPU summary to choose the correct CPU-internal signal names.")
    parts.append("")

    # CRITICAL: Tell LLM which ops are handled by OTHER operations so it doesn't duplicate work,
    # but emphasize that each operation MUST still generate its own mandatory tasks.
    parts.append("## Important Rules")
    parts.append("- ONLY generate tasks for the CURRENT operation (marked with <-- CURRENT above).")
    parts.append("- ALWAYS generate ALL mandatory tasks listed below for the CURRENT operation. Do NOT skip an operation just because another operation handles a related concern.")
    if matched_op != "RdInstr":
        parts.append("- Decode logic (instruction detection, encoding comparison) is ALREADY handled by the RdInstr operation. Do NOT generate decode tasks here.")
    if matched_op not in ("RdRS1", "RdInstr"):
        parts.append("- Register read operations (rs1, rs2) are handled by their respective RdRS1/RdRS2 operations. Do NOT generate register read tasks here.")
    if matched_op != "WrRD":
        parts.append("- Register write result routing is handled by the WrRD operation. Do NOT generate writeback tasks here.")
    parts.append("")

    if matched_op and matched_op in OP_TASK_TEMPLATES:
        templates = OP_TASK_TEMPLATES[matched_op](encoding_hint, cpu_summary, decode_expr)
        parts.append("MANDATORY tasks for THIS operation (adapt the CPU-internal signal names to the CPU structure, but use the EXACT LACE wire names). Generate ALL of them:")
        for i, t in enumerate(templates, 1):
            parts.append(f"{i}. {t}")
        parts.append("")

    parts.append(f"Current operation: {op_content}")
    parts.append("")
    parts.append("REMEMBER: Generate ALL mandatory tasks listed above for THIS operation. Do not skip any. Do not duplicate work from other operations.")

    return "\n".join(parts)
