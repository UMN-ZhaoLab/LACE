"""Generate arithmetic module skeleton from ops.

The skeleton includes SCAL-style interface wiring (ports, decode, pipeline,
pass-through) with TODO placeholders for the actual computation logic.
LLM only needs to fill in the computation / decode parts.
"""

from __future__ import annotations

import re

# Interface ops and their pipeline stage numbers (derived from port naming convention)
_INTERFACE_OP_STAGES: dict[str, int] = {
    "RdInstr": 0,
    "RdRS1": 1,
    "RdRS2": 1,
    "RdPC": 0,
    "RdMem": 2,
    "CreateCustReg": 0,
    "RdCustReg": 1,
    "WrRD": 2,
    "WrPC": 3,
    "WrMem": 2,
    "WrCustReg": 2,
}


def _parse_op_call(op_str: str) -> tuple[str, list[str]]:
    """Parse 'MUL(rs1, rs2)' -> ('MUL', ['rs1', 'rs2']).

    Also handles assignment syntax like 'var = Op(args)' by extracting
    the right-hand side before parsing the call.
    """
    stripped = op_str.strip()
    # Handle assignment syntax: 'var = Op(args)'
    if "=" in stripped:
        stripped = stripped.split("=", 1)[1].strip()
    match = re.match(r"(\w+)\s*\((.*)\)", stripped)
    if not match:
        return stripped, []
    name = match.group(1)
    args = [a.strip() for a in match.group(2).split(",") if a.strip()]
    return name, args


def _get_isax_op_name(ops: list[str]) -> str:
    """Derive a name for the ISAX operation from the first arithmetic op."""
    for op in ops:
        name, _ = _parse_op_call(op)
        if name not in _INTERFACE_OP_STAGES:
            return name.lower()
    return "lace"


def _indent(lines: list[str], spaces: int = 4) -> list[str]:
    """Indent a list of lines."""
    prefix = " " * spaces
    return [prefix + line if line.strip() else line for line in lines]


def generate_arithmetic_skeleton(ops: list[str]) -> str:
    """Generate a SCAL-style arithmetic module skeleton.

    The skeleton includes:
    - Port declarations derived from ops
    - Internal signal declarations
    - Decode / pipeline logic (RdIValid, RdFlush, RdStall)
    - Pass-through wiring
    - TODO placeholders for decode expression and computation logic

    LLM should fill in the TODO sections to complete the module.
    """
    # ------------------------------------------------------------------
    # 1. Parse ops
    # ------------------------------------------------------------------
    interface_ops: list[tuple[str, list[str]]] = []
    arithmetic_ops: list[tuple[str, list[str]]] = []
    for op in ops:
        name, args = _parse_op_call(op)
        if name in _INTERFACE_OP_STAGES:
            interface_ops.append((name, args))
        else:
            arithmetic_ops.append((name, args))

    # ------------------------------------------------------------------
    # 2. Determine features
    # ------------------------------------------------------------------
    interface_names = {n for n, _ in interface_ops}
    has_rdinstr = "RdInstr" in interface_names
    has_rdrs1 = "RdRS1" in interface_names
    has_rdrs2 = "RdRS2" in interface_names
    has_wrrd = "WrRD" in interface_names
    has_rdpc = "RdPC" in interface_names
    has_wrpc = "WrPC" in interface_names
    has_rdmem = "RdMem" in interface_names
    has_wrmem = "WrMem" in interface_names
    has_rdcustreg = "RdCustReg" in interface_names
    has_wrcustreg = "WrCustReg" in interface_names

    stages = [_INTERFACE_OP_STAGES.get(n, 0) for n in interface_names]
    max_stage = max(stages) if stages else 2

    isax_name = _get_isax_op_name(ops)

    # ------------------------------------------------------------------
    # 3. Build ports
    # ------------------------------------------------------------------
    port_lines: list[str] = []

    # Source operands are consumed by lace_arithmetic through the core-side
    # ``*_i`` ports. Do not mirror them back out: those pass-through outputs
    # have no consumer and create PINMISSING warnings (or a second driver when
    # accidentally connected to the CPU observation wires).
    if arithmetic_ops:
        port_lines.append(f"output   RdIValid_{isax_name}_1_o,")

    # ISAX-side inputs (from sub-arithmetic module, if any)
    if has_wrrd and arithmetic_ops:
        port_lines.append(f"input  [32 -1 : 0] WrRD_{isax_name}_2_i,")
        port_lines.append(f"input   WrRD_validReq_{isax_name}_2_i,")

    # Core-side outputs
    if has_wrrd:
        port_lines.append("output reg [32 -1 : 0] WrRD_2_o,")
        port_lines.append("output reg  WrRD_validReq_2_o,")
    if has_wrpc:
        port_lines.append("output reg [32 -1 : 0] WrPC_3_o,")
        port_lines.append("output reg  WrPC_validReq_3_o,")
    if has_wrmem:
        port_lines.append("output reg [32 -1 : 0] WrMem_2_o,")
        port_lines.append("output reg  WrMem_validReq_2_o,")
    if has_wrcustreg:
        port_lines.append("output reg [32 -1 : 0] WrCustReg_2_o,")
        port_lines.append("output reg  WrCustReg_validReq_2_o,")

    # Core-side inputs
    if has_rdinstr:
        port_lines.append("input  [32 -1 : 0] RdInstr_0_i,")
    if has_rdrs1:
        port_lines.append("input  [32 -1 : 0] RdRS1_1_i,")
    if has_rdrs2:
        port_lines.append("input  [32 -1 : 0] RdRS2_1_i,")
    if has_rdpc:
        port_lines.append("input  [32 -1 : 0] RdPC_0_i,")
    if has_rdmem:
        port_lines.append("input   RdMem_addr_valid_2_i,")
        port_lines.append("output  [32 -1 : 0] RdMem_2_o,")
        port_lines.append("input  [32 -1 : 0] RdMem_addr_2_i,")
        port_lines.append("input   RdMem_validReq_2_i,")
    if has_rdcustreg:
        port_lines.append("input  [32 -1 : 0] RdCustReg_1_i,")

    # Flush / Stall (always present for multi-stage pipelines)
    for s in range(max_stage + 1):
        port_lines.append(f"input   RdFlush_{s}_i,")
    for s in range(max_stage):
        port_lines.append(f"input   RdStall_{s}_i,")

    port_lines.append("input clk_i,")
    port_lines.append("input rst_i")

    # ------------------------------------------------------------------
    # 4. Build internal signals
    # ------------------------------------------------------------------
    signal_lines: list[str] = []
    if arithmetic_ops:
        signal_lines.append(f"wire  RdIValid_{isax_name}_0_s;")
        for s in range(1, max_stage + 1):
            signal_lines.append(f"wire  RdIValid_{isax_name}_{s}_s;")
            signal_lines.append(f"reg   RdIValid_{isax_name}_{s}_reg;")

    for s in range(max_stage + 1):
        signal_lines.append(f"wire  RdFlush_{s}_s;")
    for s in range(max_stage):
        signal_lines.append(f"wire  RdStall_{s}_s;")

    # ------------------------------------------------------------------
    # 5. Build pass-through logic
    # ------------------------------------------------------------------
    passthrough_lines: list[str] = []
    for s in range(max_stage + 1):
        passthrough_lines.append(f"assign RdFlush_{s}_s = RdFlush_{s}_i;")
    for s in range(max_stage):
        passthrough_lines.append(f"assign RdStall_{s}_s = RdStall_{s}_i;")

    # ------------------------------------------------------------------
    # 6. Build decode logic (TODO for LLM)
    # ------------------------------------------------------------------
    decode_lines: list[str] = []
    if arithmetic_ops:
        decode_lines.append(
            f"// TODO LLM: Replace with actual instruction encoding comparison for {isax_name}"
        )
        if has_rdinstr:
            decode_lines.append(
                f"assign RdIValid_{isax_name}_0_s = ( /* DECODE: compare RdInstr_0_i with instruction encoding */ ) && !RdFlush_0_s;"
            )
        else:
            decode_lines.append(
                f"assign RdIValid_{isax_name}_0_s = 1'b1 && !RdFlush_0_s;"
            )

    # ------------------------------------------------------------------
    # 7. Build pipeline registers
    # ------------------------------------------------------------------
    pipeline_lines: list[str] = []
    if arithmetic_ops:
        for s in range(1, max_stage + 1):
            prev = s - 1
            pipeline_lines.append(
                f"assign RdIValid_{isax_name}_{s}_s = RdIValid_{isax_name}_{s}_reg && !RdFlush_{s}_s;"
            )
            pipeline_lines.append(f"always@(posedge clk_i) begin")
            pipeline_lines.append(f"    if (rst_i)")
            pipeline_lines.append(f"        RdIValid_{isax_name}_{s}_reg <= 0;")
            pipeline_lines.append(f"    else if (!(RdStall_{prev}_s))")
            pipeline_lines.append(
                f"        RdIValid_{isax_name}_{s}_reg <= RdIValid_{isax_name}_{prev}_s;"
            )
            pipeline_lines.append(f"end")
            pipeline_lines.append("")

    # ------------------------------------------------------------------
    # 8. Build output wiring (with TODO for computation logic)
    # ------------------------------------------------------------------
    output_lines: list[str] = []
    if has_wrrd and arithmetic_ops:
        output_lines.append(f"// TODO LLM: Implement arithmetic computation for {isax_name}")
        output_lines.append(f"// Operations to implement:")
        for name, args in arithmetic_ops:
            output_lines.append(f"//   - {name}({', '.join(args)})")
        output_lines.append(f"// Available inputs:")
        if has_rdrs1:
            output_lines.append(f"//   - RdRS1_1_i [31:0]")
        if has_rdrs2:
            output_lines.append(f"//   - RdRS2_1_i [31:0]")
        if has_rdinstr:
            output_lines.append(f"//   - RdInstr_0_i [31:0] (for immediate extraction)")
        if has_rdpc:
            output_lines.append(f"//   - RdPC_0_i [31:0]")
        output_lines.append(f"// Output should drive WrRD_2_o")
        output_lines.append("")
        output_lines.append(
            f"always @(*)  WrRD_2_o = WrRD_{isax_name}_2_i;  // TODO: replace with internal computation if no sub-module"
        )
        output_lines.append("always @(*) begin")
        output_lines.append("    case(1'b1)")
        output_lines.append(
            f"        RdIValid_{isax_name}_2_s : WrRD_validReq_2_o = WrRD_validReq_{isax_name}_2_i;"
        )
        output_lines.append("        default : WrRD_validReq_2_o = ~1;")
        output_lines.append("    endcase")
        output_lines.append("end")
    elif has_wrrd:
        output_lines.append("always @(*)  WrRD_2_o = 32'd0;")
        output_lines.append("always @(*)  WrRD_validReq_2_o = ~1;")

    # ------------------------------------------------------------------
    # 9. Assemble module
    # ------------------------------------------------------------------
    lines: list[str] = []
    lines.append("// ============================================================")
    lines.append("// AUTO-GENERATED ARITHMETIC MODULE SKELETON")
    lines.append("// This skeleton contains all interface wiring.")
    lines.append("// LLM should ONLY fill in the TODO sections.")
    lines.append("// ============================================================")
    lines.append("")
    lines.append("module lace_arithmetic (")
    lines.extend(_indent(port_lines))
    lines.append(");")
    lines.append("")

    if signal_lines:
        lines.append("// Declare local signals")
        lines.extend(signal_lines)
        lines.append("")

    if passthrough_lines:
        lines.append("// Pass-through wiring")
        lines.extend(passthrough_lines)
        lines.append("")

    if decode_lines:
        lines.append("// Decode logic")
        lines.extend(decode_lines)
        lines.append("")

    if pipeline_lines:
        lines.append("// Pipeline registers")
        lines.extend(pipeline_lines)

    if output_lines:
        lines.append("// Output / computation logic")
        lines.extend(output_lines)
        lines.append("")

    lines.append("endmodule")
    lines.append("")

    return "\n".join(lines)
