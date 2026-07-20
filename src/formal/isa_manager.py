"""Manage riscv-formal ISA instruction lists and checks.cfg updates.

When adding custom instructions (e.g. ROL from Zbb), riscv-formal needs:
1. An `isa_<name>.txt` file listing all instructions to verify
2. `checks.cfg` pointing to that ISA string
3. An `insn_<name>.v` model file (usually provided by riscv-formal upstream)

This module automates creation of the ISA list and updates checks.cfg.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any


# Map custom instruction names to their standard RISC-V extension
CUSTOM_INSN_EXTENSIONS: dict[str, str] = {
    "rol": "Zbb",
    "ror": "Zbb",
    "rori": "Zbb",
    "andn": "Zbb",
    "orn": "Zbb",
    "xnor": "Zbb",
    "clz": "Zbb",
    "ctz": "Zbb",
    "cpop": "Zbb",
    "max": "Zbb",
    "maxu": "Zbb",
    "min": "Zbb",
    "minu": "Zbb",
    "sext_b": "Zbb",
    "sext_h": "Zbb",
    "zext_h": "Zbb",
    "orc_b": "Zbb",
    "rev8": "Zbb",
    "bclr": "Zbs",
    "bclri": "Zbs",
    "bext": "Zbs",
    "bexti": "Zbs",
    "binv": "Zbs",
    "binvi": "Zbs",
    "bset": "Zbs",
    "bseti": "Zbs",
    "clmul": "Zbc",
    "clmulh": "Zbc",
    "clmulr": "Zbc",
}


def _custom_isa_name(base_isa: str, custom_insns: list[str]) -> str:
    """Return a genchecks-compatible ISA name for standard and LACE models."""
    extensions = sorted({
        CUSTOM_INSN_EXTENSIONS[i.lower()]
        for i in custom_insns
        if i.lower() in CUSTOM_INSN_EXTENSIONS
    })
    has_lace_model = any(i.lower() not in CUSTOM_INSN_EXTENSIONS for i in custom_insns)
    suffix = "".join(extensions)
    if has_lace_model:
        # genchecks.py accepts a multi-letter X extension after an optional
        # underscore. Keep the name stable; each run has a private sandbox.
        suffix += "_Xlace"
    return base_isa + suffix


def _find_base_isa_file(riscv_formal_dir: Path, base_isa: str) -> Path | None:
    """Find the base ISA list file (e.g. isa_rv32imc.txt)."""
    candidate = riscv_formal_dir / "insns" / f"isa_{base_isa}.txt"
    if candidate.exists():
        return candidate
    return None


def _find_extension_isa_file(riscv_formal_dir: Path, ext: str, xlen: int = 32) -> Path | None:
    """Find an extension ISA list file (e.g. isa_rv32iZbb.txt)."""
    # Try rv32i<ext> first
    candidate = riscv_formal_dir / "insns" / f"isa_rv{xlen}i{ext}.txt"
    if candidate.exists():
        return candidate
    # Try rv32<ext>
    candidate = riscv_formal_dir / "insns" / f"isa_rv{xlen}{ext.lower()}.txt"
    if candidate.exists():
        return candidate
    return None


def generate_custom_isa_list(
    riscv_formal_dir: str | Path,
    base_isa: str,
    custom_insns: list[str],
    xlen: int = 32,
) -> Path | None:
    """Generate an ISA instruction list that includes base ISA + custom instructions.

    Args:
        riscv_formal_dir: Path to riscv-formal root
        base_isa: Base ISA string, e.g. "rv32imc"
        custom_insns: List of custom instruction names to add
        xlen: XLEN (32 or 64)

    Returns:
        Path to generated ISA list file, or None if base ISA not found
    """
    riscv_formal_dir = Path(riscv_formal_dir)
    insns_dir = riscv_formal_dir / "insns"
    insns_dir.mkdir(parents=True, exist_ok=True)

    base_file = _find_base_isa_file(riscv_formal_dir, base_isa)
    if base_file is None:
        return None

    # Collect all extensions needed
    extensions: set[str] = set()
    for insn in custom_insns:
        ext = CUSTOM_INSN_EXTENSIONS.get(insn.lower())
        if ext:
            extensions.add(ext)

    # Build a genchecks-compatible ISA name. Unknown/custom LACE models use a
    # private X extension instead of overwriting the base ISA list.
    sorted_exts = sorted(extensions)
    new_isa = _custom_isa_name(base_isa, custom_insns)
    output_file = insns_dir / f"isa_{new_isa}.txt"

    # Start with base ISA instructions
    base_insns = set(base_file.read_text(encoding="utf-8").splitlines())
    base_insns = {i.strip() for i in base_insns if i.strip() and not i.startswith("#")}

    # Add instructions from each extension
    added_insns: set[str] = set()
    for ext in sorted_exts:
        ext_file = _find_extension_isa_file(riscv_formal_dir, ext, xlen)
        if ext_file is None:
            continue
        for line in ext_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            added_insns.add(line)

    # Filter extension lists to only the requested instructions, then add the
    # requested names directly. The direct addition is required for generated
    # LACE models such as aes32esi that are not in an upstream extension list.
    wanted = {i.lower() for i in custom_insns}
    added_insns = {i for i in added_insns if i.lower() in wanted}
    added_insns.update(wanted)

    all_insns = sorted(base_insns | added_insns)
    output_file.write_text("\n".join(all_insns) + "\n", encoding="utf-8")
    return output_file


def update_checks_cfg(
    core_dir: str | Path,
    new_isa: str,
) -> bool:
    """Update the ISA line in a riscv-formal core's checks.cfg.

    Args:
        core_dir: Path to the riscv-formal core directory (e.g. cores/picorv32)
        new_isa: New ISA string, e.g. "rv32imcZbb"

    Returns:
        True if updated successfully
    """
    core_dir = Path(core_dir)
    cfg_path = core_dir / "checks.cfg"
    if not cfg_path.exists():
        return False

    content = cfg_path.read_text(encoding="utf-8")
    new_content = re.sub(
        r"^(isa\s+)\S+",
        rf"\g<1>{new_isa}",
        content,
        flags=re.MULTILINE,
    )
    if new_content == content:
        return False
    cfg_path.write_text(new_content, encoding="utf-8")
    return True


def configure_riscv_formal_for_custom_instructions(
    riscv_formal_dir: str | Path,
    core_name: str,
    base_isa: str,
    custom_insns: list[str],
    xlen: int = 32,
) -> dict[str, Any]:
    """One-shot configuration of riscv-formal for custom instructions.

    Returns:
        dict with keys: isa_file, checks_cfg_updated, new_isa
    """
    riscv_formal_dir = Path(riscv_formal_dir)
    core_dir = riscv_formal_dir / "cores" / core_name

    isa_file = generate_custom_isa_list(
        riscv_formal_dir, base_isa, custom_insns, xlen
    )

    new_isa = _custom_isa_name(base_isa, custom_insns)

    cfg_updated = update_checks_cfg(core_dir, new_isa)

    return {
        "isa_file": str(isa_file) if isa_file else None,
        "checks_cfg_updated": cfg_updated,
        "new_isa": new_isa,
    }
