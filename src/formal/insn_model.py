"""RISC-V Formal instruction model management.

Handles locating upstream insn model files and writing LLM-generated models.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def normalize_rd_wdata_x0(verilog_code: str) -> str:
    """Force instruction models to report zero write data for ``rd=x0``.

    RVFI cores suppress writes to x0 and consequently report ``rd_wdata=0``.
    A model that emits the arithmetic result unconditionally creates a false
    counterexample whenever the solver chooses rd address zero.
    """
    pattern = re.compile(
        r"(assign\s+spec_rd_wdata\s*=\s*)(.*?)(;)",
        re.DOTALL,
    )
    match = pattern.search(verilog_code)
    if not match or "spec_rd_addr" in match.group(2):
        return verilog_code

    expression = match.group(2).strip()
    replacement = (
        f"{match.group(1)}(spec_rd_addr != 0) ? ({expression}) : "
        "{`RISCV_FORMAL_XLEN{1'b0}};"
    )
    return verilog_code[: match.start()] + replacement + verilog_code[match.end() :]


def register_custom_instruction(
    insn_name: str,
    riscv_formal_dir: str | Path,
) -> Path | None:
    """Check if an insn model file exists in riscv-formal.

    If the file ``insns/insn_{insn_name}.v`` already exists (e.g. provided by
    the upstream riscv-formal submodule), return its path.  Otherwise return
    ``None`` — the caller is responsible for generating the file (e.g. via
    ``write_insn_model``).

    Args:
        insn_name: Instruction name (e.g. ``"rol"``).
        riscv_formal_dir: Root of the riscv-formal repository.

    Returns:
        Path to the model file, or ``None`` if it does not exist.
    """
    riscv_formal_dir = Path(riscv_formal_dir)
    model_path = riscv_formal_dir / "insns" / f"insn_{insn_name}.v"
    if model_path.exists():
        return model_path
    return None


def write_insn_model(
    insn_name: str,
    verilog_code: str,
    riscv_formal_dir: str | Path,
) -> Path:
    """Write an instruction model file into the riscv-formal insns directory.

    Args:
        insn_name: Instruction name (e.g. ``"rol"``).
        verilog_code: Complete Verilog module source.
        riscv_formal_dir: Root of the riscv-formal repository.

    Returns:
        Path to the written file.
    """
    riscv_formal_dir = Path(riscv_formal_dir)
    insns_dir = riscv_formal_dir / "insns"
    insns_dir.mkdir(parents=True, exist_ok=True)

    path = insns_dir / f"insn_{insn_name}.v"
    path.write_text(normalize_rd_wdata_x0(verilog_code), encoding="utf-8")
    logger.info("Wrote insn model: %s", path)
    return path
