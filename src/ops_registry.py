"""Central registry for predefined ISA operations used across prompts and validators."""

from __future__ import annotations

_INTERFACE_OPS: dict[str, str] = {
    "RdInstr": "read the instruction register. Returns the 32-bit instruction word. Usage: `insn = RdInstr()`",
    "RdRS1": "read the GPR specified by the rs1 field of the current instruction. Returns the register value. Usage: `rs1 = RdRS1()`",
    "RdRS2": "read the GPR specified by the rs2 field of the current instruction. Returns the register value. Usage: `rs2 = RdRS2()`",
    "CreateCustReg": "create `num` entries of custom register file. Usage: `CreateCustReg(num)`",
    "RdCustReg": "read the custom register at given address. Usage: `val = RdCustReg(addr)`",
    "RdPC": "read the current program counter. Returns PC value. Usage: `pc = RdPC()`",
    "RdMem": "read a 32-bit word from memory at the given address. Usage: `val = RdMem(addr)`",
    "WrRD": "write a value to the GPR specified by the rd field. Usage: `WrRD(value)`",
    "WrCustReg": "write a value to the custom register at given address. Usage: `WrCustReg(addr, value)`",
    "WrPC": "write a new value to the program counter. Usage: `WrPC(new_pc)`",
    "WrMem": "write a value to memory at the given address. Usage: `WrMem(addr, value)`",
}

_ARITHMETIC_OPS: dict[str, str] = {
    "AND": "Bitwise AND of variable X and Y",
    "OR": "Bitwise OR of variable X and Y",
    "XOR": "Bitwise XOR of variable X and Y",
    "NOT": "Bitwise NOT of variable X",
    "ADD": "Add variable X and Y",
    "SUB": "Subtraction of variable X and Y",
    "MUL": "Multiply variable X and Y",
    "DIV": "Divide variable X and Y",
    "SLL": "Logical left shift variable X by Y bits",
    "SRL": "Logical right shift variable X by Y bits",
    "SRA": "Arithmetic right shift variable X by Y bits",
    "SLICE": "Get bits Y to Z [Y:Z] of the variable X",
    "CONCAT": "Concatenate variable X and Y",
    "SIGN_EXTEND": "Sign extend variable X to Y bits",
    "UNSIGN_EXTEND": "Zero extend variable X to Y bits",
    "CMP_GE_S": "under signed condition, if X >= Y, return 1, else 0",
    "CMP_GE_U": "under unsigned condition, if X >= Y, return 1, else 0",
    "CMP_GT_S": "under signed condition, if X > Y, return 1, else 0",
    "CMP_GT_U": "under unsigned condition, if X > Y, return 1, else 0",
    "CMP_LT_S": "under signed condition, if X < Y, return 1, else 0",
    "CMP_LT_U": "under unsigned condition, if X < Y, return 1, else 0",
    "CMP_LE_S": "under signed condition, if X <= Y, return 1, else 0",
    "CMP_LE_U": "under unsigned condition, if X <= Y, return 1, else 0",
    "CMP_NE": "if X != Y, return 1, else 0",
    "CMP_EQ": "if X == Y, return 1, else 0",
    "COND": "if C is 1, return X, else Y",
    "CustomLogic": "Any custom logic not listed above",
}

INTERFACE_OPS_SET: set[str] = set(_INTERFACE_OPS.keys())
ARITHMETIC_OPS_SET: set[str] = set(_ARITHMETIC_OPS.keys())
PREDEFINED_OPS_SET: set[str] = INTERFACE_OPS_SET | ARITHMETIC_OPS_SET


def format_predefined_ops(include_arithmetic: bool = False) -> str:
    """Generate the reference-table text used in LLM prompts.

    By default only interface ops are returned, since spec2op now splits
    interface and arithmetic into two separate outputs.
    """
    lines = ["The following is the reference table for the predefined INTERFACE operations:", ""]
    lines.append("interface part:")
    for name, desc in _INTERFACE_OPS.items():
        lines.append(f"- {name}(): {desc}")
    if include_arithmetic:
        lines.append("")
        lines.append("arithmetic part:")
        for name, desc in _ARITHMETIC_OPS.items():
            lines.append(f"- {name}(X,Y): {desc}")
    return "\n".join(lines)
