"""Run riscv-formal checks on a modified CPU core."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import LACEConfig
from src.formal.insn_model import register_custom_instruction, write_insn_model
from src.formal.isa_manager import configure_riscv_formal_for_custom_instructions

logger = logging.getLogger(__name__)

BASELINE_CHECKS = [
    "cover",
    "csr_ill_c00_ch0",
    "csr_ill_c02_ch0",
    "csr_ill_c80_ch0",
    "csr_ill_c82_ch0",
    "csrw_mcycle_ch0",
    "csrw_minstret_ch0",
    "csrc_inc_mcycle_ch0",
    "csrc_inc_minstret_ch0",
    "csrc_upcnt_mcycle_ch0",
    "csrc_upcnt_minstret_ch0",
]


@dataclass
class FormalCheckResult:
    """Result of a single riscv-formal check."""

    name: str
    passed: bool
    elapsed_seconds: float
    error: str = ""
    trace_path: str = ""


class RiscvFormalRunner:
    """Run riscv-formal checks on a modified CPU core.

    Workflow:
    1. Copy modified picorv32.v from workspace to riscv-formal core dir
    2. Run genchecks.py to generate .sby files
    3. Execute selected sby checks
    4. Parse PASS/FAIL results
    """

    def __init__(
        self,
        cpu_name: str = "picorv32",
        riscv_formal_dir: str | None = None,
        solver: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.cpu_name = cpu_name
        # Resolve riscv-formal dir relative to project root if not absolute
        rf_dir = Path(riscv_formal_dir or LACEConfig.RISCV_FORMAL_DIR)
        if not rf_dir.is_absolute():
            # Try relative to current working directory (project root)
            rf_dir = Path.cwd() / rf_dir
        self.riscv_formal_dir = rf_dir.resolve()
        self.core_dir = self.riscv_formal_dir / "cores" / cpu_name
        self.checks_dir = self.core_dir / "checks"
        self.solver = solver or LACEConfig.RISCV_FORMAL_SOLVER
        self.timeout = timeout or LACEConfig.RISCV_FORMAL_TIMEOUT

    def _copy_rtl(
        self,
        workspace_dir: str,
        cpu_top_file: str,
        extra_verilog_files: list[str] | None = None,
    ) -> None:
        """Copy modified CPU RTL from workspace to riscv-formal core dir."""
        src = Path(workspace_dir) / cpu_top_file
        dst = self.core_dir / cpu_top_file
        if not src.exists():
            raise FileNotFoundError(f"Workspace RTL not found: {src}")
        shutil.copy2(str(src), str(dst))

        # Copy any extra Verilog files (e.g. lace_arithmetic.v)
        for fname in extra_verilog_files or []:
            extra_src = Path(workspace_dir) / fname
            extra_dst = self.core_dir / fname
            if extra_src.exists():
                shutil.copy2(str(extra_src), str(extra_dst))

    def _patch_checks_cfg_for_extra_files(
        self, extra_verilog_files: list[str] | None = None
    ) -> None:
        """Ensure checks.cfg [verilog-files] includes extra Verilog files."""
        cfg_path = self.core_dir / "checks.cfg"
        if not cfg_path.exists():
            return

        content = cfg_path.read_text(encoding="utf-8")
        # Find [verilog-files] section and append missing extra files
        section_match = re.search(
            r"(\[verilog-files\]\n)(.*?)(?=\n\[|\Z)",
            content,
            re.DOTALL,
        )
        if not section_match:
            return

        section_start = section_match.start(2)
        section_end = section_match.end(2)
        section_body = section_match.group(2)

        extra_lines: list[str] = []
        for fname in extra_verilog_files or []:
            line = f"@basedir@/cores/@core@/{fname}"
            if line not in section_body:
                extra_lines.append(line)

        if not extra_lines:
            return

        new_section_body = section_body.rstrip() + "\n" + "\n".join(extra_lines) + "\n"
        new_content = content[:section_start] + new_section_body + content[section_end:]
        cfg_path.write_text(new_content, encoding="utf-8")

    def _ensure_genchecks_real(self) -> None:
        """Restore the real genchecks.py if it has been replaced by a stub."""
        genchecks = self.riscv_formal_dir / "checks" / "genchecks.py"
        if not genchecks.exists():
            raise FileNotFoundError(f"genchecks.py not found: {genchecks}")

        content = genchecks.read_text(encoding="utf-8").strip()
        # A real genchecks.py is much larger than a stub and contains the
        # canonical header.  If it looks fake, restore it from the submodule.
        if len(content) < 200 or "Claire Xenia Wolf" not in content:
            logger.warning("genchecks.py appears to be a stub; restoring from git HEAD")
            result = subprocess.run(
                ["git", "checkout", "HEAD", "--", "checks/genchecks.py"],
                cwd=str(self.riscv_formal_dir),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to restore genchecks.py: {result.stderr}")
            if not genchecks.exists():
                raise RuntimeError("genchecks.py missing after git restore")

    def _run_genchecks(self) -> None:
        """Run genchecks.py to regenerate .sby files."""
        self._ensure_genchecks_real()
        genchecks = self.riscv_formal_dir / "checks" / "genchecks.py"

        # Clean old checks
        if self.checks_dir.exists():
            shutil.rmtree(str(self.checks_dir))

        result = subprocess.run(
            ["python3", str(genchecks)],
            cwd=str(self.core_dir),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"genchecks.py failed: {result.stderr}")

    def _patch_solver(self) -> None:
        """Replace solver in all .sby files with the configured solver."""
        pattern = re.compile(r"smtbmc\s+\w+")
        replacement = f"smtbmc {self.solver}"
        patched = 0
        for sby_file in self.checks_dir.glob("*.sby"):
            content = sby_file.read_text()
            new_content, count = pattern.subn(replacement, content)
            if count > 0:
                sby_file.write_text(new_content)
                patched += count
        if patched == 0:
            logger.warning(
                "Solver patching: no 'smtbmc <solver>' lines found in %s. "
                "The .sby files may use an unexpected format.",
                self.checks_dir,
            )

    def _run_sby_check(self, check_name: str) -> FormalCheckResult:
        """Run a single sby check and parse result."""
        sby_file = self.checks_dir / f"{check_name}.sby"
        if not sby_file.exists():
            return FormalCheckResult(
                name=check_name,
                passed=False,
                elapsed_seconds=0.0,
                error=".sby file not found",
            )

        # Clean previous run
        output_dir = self.checks_dir / check_name
        if output_dir.exists():
            shutil.rmtree(str(output_dir))

        start = time.time()
        try:
            result = subprocess.run(
                ["sby", "-f", str(sby_file)],
                cwd=str(self.core_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            elapsed = time.time() - start

            stdout = result.stdout
            passed = "DONE (PASS" in stdout
            failed = "DONE (FAIL" in stdout

            if passed:
                return FormalCheckResult(
                    name=check_name, passed=True, elapsed_seconds=elapsed
                )
            elif failed:
                # Extract trace path if available
                trace_path = ""
                for line in stdout.splitlines():
                    if "counterexample trace:" in line:
                        trace_path = line.split(":", 1)[1].strip()
                        break
                return FormalCheckResult(
                    name=check_name,
                    passed=False,
                    elapsed_seconds=elapsed,
                    error="Assertion failed",
                    trace_path=trace_path,
                )
            else:
                return FormalCheckResult(
                    name=check_name,
                    passed=False,
                    elapsed_seconds=elapsed,
                    error=f"sby exited with rc={result.returncode}. stdout: {stdout[:500]}",
                )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            return FormalCheckResult(
                name=check_name,
                passed=False,
                elapsed_seconds=elapsed,
                error=f"Timeout after {self.timeout}s",
            )
        except Exception as exc:
            elapsed = time.time() - start
            return FormalCheckResult(
                name=check_name,
                passed=False,
                elapsed_seconds=elapsed,
                error=str(exc),
            )

    def run_checks(
        self,
        workspace_dir: str,
        cpu_top_file: str,
        check_names: list[str] | None = None,
        custom_insns: list[str] | None = None,
        extra_verilog_files: list[str] | None = None,
        base_isa: str = "rv32imc",
    ) -> dict[str, Any]:
        """Run riscv-formal checks on the modified CPU.

        Args:
            workspace_dir: Path to workspace containing modified picorv32.v
            cpu_top_file: Name of top-level Verilog file
            check_names: List of check names to run (e.g. ["cover", "csr_ill_c00_ch0"])
                         If None, runs a default fast subset.
            custom_insns: Custom instruction names to register (e.g. ["rol"])
            extra_verilog_files: Additional Verilog files to copy (e.g. ["lace_arithmetic.v"])
            base_isa: Base ISA string for riscv-formal configuration

        Returns:
            dict with keys: passed (bool), results (list), total_time (float), error (str)
        """
        if check_names is None:
            raise ValueError("check_names must be provided; use BASELINE_CHECKS for baseline")

        try:
            # Configure riscv-formal for custom instructions
            if custom_insns:
                configure_riscv_formal_for_custom_instructions(
                    self.riscv_formal_dir,
                    self.cpu_name,
                    base_isa,
                    custom_insns,
                )

            # Ensure insn model files exist for custom instructions
            for insn_name in custom_insns or []:
                existing = register_custom_instruction(insn_name, self.riscv_formal_dir)
                if existing is None:
                    logger.warning(
                        "No insn model file found for '%s' and it is not in "
                        "CUSTOM_INSN_EXTENSIONS. The insn_%s check will likely fail.",
                        insn_name, insn_name,
                    )

            self._copy_rtl(workspace_dir, cpu_top_file, extra_verilog_files)
            self._patch_checks_cfg_for_extra_files(extra_verilog_files)
            self._run_genchecks()
            self._patch_solver()

            results: list[FormalCheckResult] = []
            all_passed = True
            total_time = 0.0

            for name in check_names:
                result = self._run_sby_check(name)
                results.append(result)
                total_time += result.elapsed_seconds
                if not result.passed:
                    all_passed = False

            return {
                "passed": all_passed,
                "results": [
                    {
                        "name": r.name,
                        "passed": r.passed,
                        "elapsed_seconds": r.elapsed_seconds,
                        "error": r.error,
                        "trace_path": r.trace_path,
                    }
                    for r in results
                ],
                "total_time": total_time,
                "error": "",
            }
        except Exception as exc:
            # Surface a traceback summary so callers can diagnose without
            # re-running. The top-level contract (passed=False + error str)
            # is preserved.
            tb = traceback.format_exc()
            logger.error("riscv-formal run_checks failed: %s\n%s", exc, tb)
            return {
                "passed": False,
                "results": [],
                "total_time": 0.0,
                "error": f"{exc}\n{tb}",
            }

    def run_baseline_checks(
        self,
        workspace_dir: str,
        cpu_top_file: str,
        extra_verilog_files: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run baseline RV32I checks (cover + CSR checks).

        These verify that the modified CPU still correctly implements
        the base RISC-V instruction set.
        """
        return self.run_checks(
            workspace_dir,
            cpu_top_file,
            check_names=BASELINE_CHECKS,
            extra_verilog_files=extra_verilog_files,
        )

    def run_custom_instruction_checks(
        self,
        workspace_dir: str,
        cpu_top_file: str,
        insn_name: str,
        extra_verilog_files: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run checks for a custom instruction (e.g. 'rol').

        These will fail if the CPU does not yet implement the instruction.
        """
        return self.run_checks(
            workspace_dir,
            cpu_top_file,
            check_names=[f"insn_{insn_name}_ch0"],
            custom_insns=[insn_name],
            extra_verilog_files=extra_verilog_files,
        )
