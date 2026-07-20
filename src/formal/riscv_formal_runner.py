"""Run riscv-formal checks on a modified CPU core."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fcntl

from src.config import LACEConfig
from src.formal.e203_rvfi_adapter import apply_e203_rvfi_adapter
from src.formal.insn_model import register_custom_instruction, write_insn_model
from src.formal.isa_manager import (
    configure_riscv_formal_for_custom_instructions,
    update_checks_cfg,
)
from src.formal.sandbox import is_formal_sandbox

logger = logging.getLogger(__name__)

# ``genchecks.py`` owns the set of instruction proof jobs.  Keep only the
# non-instruction job here; ``run_baseline_checks`` adds every generated
# ``insn_*_ch0`` job after generation.  The former hard-coded CSR list was
# from an older riscv-formal layout and made a healthy run fail merely because
# those .sby files no longer exist.
BASELINE_CHECKS = ["cover"]

# e203's private RVFI adapter has been proved against the uncompressed RV32I
# instruction suite.  Do not claim RVC coverage until its architectural hint
# behavior is also compatible with the riscv-formal RVC models.
BASELINE_ISA_BY_CPU = {"e203_hbirdv2": "rv32i"}


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
        self._replace_with_copy(src, dst)

        # Copy any extra Verilog files (e.g. lace_arithmetic.v)
        for fname in extra_verilog_files or []:
            extra_src = Path(workspace_dir) / fname
            extra_dst = self.core_dir / fname
            if extra_src.exists():
                self._replace_with_copy(extra_src, extra_dst)

    @staticmethod
    def _replace_with_copy(src: Path, dst: Path) -> None:
        """Replace a sandbox symlink with a private copy without write-through."""
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        elif dst.exists():
            raise IsADirectoryError(f"Expected RTL file but found directory: {dst}")
        shutil.copy2(src, dst)

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

    def _patch_checks_cfg_for_workspace_sources(
        self,
        workspace_dir: str,
        cpu_top_file: str,
    ) -> None:
        """Redirect private formal configs from checkout-relative RTL paths."""
        cfg_path = self.core_dir / "checks.cfg"
        if not cfg_path.exists():
            return
        workspace = Path(workspace_dir).resolve()
        content = cfg_path.read_text(encoding="utf-8")
        prototype_ref = re.compile(r"@basedir@(?:/\.\.)+/cpu_prototype/[^/\s]+")
        rewritten = prototype_ref.sub(str(workspace), content)
        has_formal_source_closure = rewritten != content

        section = re.search(
            r"(\[verilog-files\]\n)(.*?)(?=\n\[|\Z)",
            rewritten,
            re.DOTALL,
        )
        configured_sources = section.group(2).replace("@core@", self.cpu_name) if section else ""
        if section and not has_formal_source_closure and cpu_top_file not in configured_sources:
            top_path = (workspace / cpu_top_file).resolve()
            # Some prototypes retain both a flattened export and the canonical
            # RTL tree.  Use the configured top's nearest ``rtl`` ancestor so
            # the private formal config gets one coherent source closure,
            # rather than loading duplicate module definitions from both.
            source_root = workspace
            for ancestor in (top_path.parent, *top_path.parents):
                if ancestor.name == "rtl" and ancestor.is_relative_to(workspace):
                    source_root = ancestor
                    break
            sources = sorted({
                path.resolve()
                for pattern in ("*.v", "*.sv")
                for path in source_root.rglob(pattern)
            })
            replacement = section.group(1) + "@basedir@/cores/@core@/wrapper.sv\n"
            replacement += "\n".join(str(path) for path in sources) + "\n"
            rewritten = rewritten[:section.start()] + replacement + rewritten[section.end():]

            # genchecks emits one ``read -sv`` command from the config.  Its
            # working directory is the generated check directory, not each
            # source file's directory, so preserve the source tree's include
            # search paths through Yosys' generic Verilog defaults.
            include_dirs = sorted({path.parent for path in sources})
            defaults = "\n".join(
                f"verilog_defaults -add -I{directory}" for directory in include_dirs
            )
            defaults_section = re.search(
                r"(\[script-defines\]\n)(.*?)(?=\n\[|\Z)",
                rewritten,
                re.DOTALL,
            )
            if defaults_section:
                existing = defaults_section.group(2).rstrip()
                replacement = defaults_section.group(1) + existing + "\n" + defaults + "\n"
                rewritten = (
                    rewritten[:defaults_section.start()]
                    + replacement
                    + rewritten[defaults_section.end():]
                )
            else:
                insertion = "[script-defines]\n" + defaults + "\n\n"
                verilog_section = re.search(r"\[verilog-files\]\n", rewritten)
                if verilog_section:
                    rewritten = (
                        rewritten[:verilog_section.start()]
                        + insertion
                        + rewritten[verilog_section.start():]
                    )

        if rewritten != content:
            cfg_path.write_text(rewritten, encoding="utf-8")

    def _apply_private_core_adapter(self, workspace_dir: str) -> None:
        """Apply a CPU-specific formal adapter only to the private sandbox."""
        if self.cpu_name != "e203_hbirdv2":
            return

        source = Path(workspace_dir) / "e203_hbirdv2.v"
        if not source.is_file():
            raise FileNotFoundError(f"e203 flattened source not found: {source}")
        destination = self.core_dir / "e203_hbirdv2.v"
        apply_e203_rvfi_adapter(source, destination)

        cfg_path = self.core_dir / "checks.cfg"
        content = cfg_path.read_text(encoding="utf-8")
        private_ref = "@basedir@/cores/@core@/e203_hbirdv2.v"
        verilog_section = re.search(
            r"(\[verilog-files\]\n)(.*?)(?=\n\[|\Z)", content, re.DOTALL
        )
        if not verilog_section:
            raise RuntimeError("e203 formal config has no [verilog-files] section")
        # The flattened e203 export is a self-contained source closure.  Use
        # it exclusively: earlier runs may have expanded the canonical RTL
        # tree, which would otherwise duplicate every module definition.
        private_sources = (
            verilog_section.group(1)
            + "@basedir@/cores/@core@/wrapper.sv\n"
            + private_ref
            + "\n"
            + str((Path(workspace_dir) / "rtl" / "e203" / "general" / "*.v").resolve())
            + "\n"
        )
        content = (
            content[:verilog_section.start()]
            + private_sources
            + content[verilog_section.end():]
        )
        compressed_define = "`define RISCV_FORMAL_COMPRESSED"
        if compressed_define not in content:
            defines_section = re.search(
                r"(\[defines\]\n)(.*?)(?=\n\[|\Z)", content, re.DOTALL
            )
            if not defines_section:
                raise RuntimeError("e203 formal config has no [defines] section")
            defines = defines_section.group(1) + defines_section.group(2).rstrip()
            defines += "\n" + compressed_define + "\n"
            content = (
                content[:defines_section.start()]
                + defines
                + content[defines_section.end():]
            )
        include_dir = (Path(workspace_dir) / "rtl" / "e203" / "general").resolve()
        include_default = f"verilog_defaults -add -I{include_dir}"
        if not include_dir.is_dir():
            raise FileNotFoundError(f"e203 include directory not found: {include_dir}")
        if include_default not in content:
            script_defines = re.search(
                r"(\[script-defines\]\n)(.*?)(?=\n\[|\Z)", content, re.DOTALL
            )
            if script_defines:
                section = script_defines.group(1) + script_defines.group(2).rstrip()
                section += "\n" + include_default + "\n"
                content = content[:script_defines.start()] + section + content[script_defines.end():]
            else:
                marker = "[verilog-files]\n"
                content = content.replace(
                    marker, f"[script-defines]\n{include_default}\n\n{marker}", 1
                )
        cfg_path.write_text(content, encoding="utf-8")

    def _sby_environment(self) -> dict[str, str] | None:
        """Source the complete OSS CAD Suite environment for formal runs."""
        cfg_path = self.core_dir / "checks.cfg"
        content = cfg_path.read_text(encoding="utf-8") if cfg_path.exists() else ""
        plugin_match = re.search(r"plugin\s+-i\s+([^\s]+slang\.so)", content)

        suite_root: Path | None = None
        if plugin_match:
            suite_root = Path(plugin_match.group(1)).parents[3]
        else:
            configured_formal = Path(LACEConfig.RISCV_FORMAL_DIR)
            if not configured_formal.is_absolute():
                configured_formal = Path.cwd() / configured_formal
            formal_root = configured_formal.resolve()
            candidates = (
                formal_root.parent / "oss-cad-suite",
                Path.cwd() / "tools" / "oss-cad-suite",
                Path.cwd().parent / "tools" / "oss-cad-suite",
            )
            suite_root = next(
                (candidate for candidate in candidates if (candidate / "environment").is_file()),
                None,
            )

        if suite_root is None:
            return None
        suite_bin = suite_root / "bin"
        if not (suite_bin / "yosys").is_file() or not (suite_bin / "sby").is_file():
            return None

        # The release environment also supplies the matching SMT solvers and
        # Python helpers.  Merely prepending ``bin`` can leave an incompatible
        # host solver or helper on PATH, so reproduce ``source .../environment``
        # and pass its exported environment directly to SBY.
        environment_script = suite_root / "environment"
        if environment_script.is_file():
            sourced = subprocess.run(
                ["bash", "-c", 'source "$1" && env -0', "lace-formal-env", str(environment_script)],
                capture_output=True,
                check=False,
                env=os.environ.copy(),
            )
            if sourced.returncode == 0:
                return {
                    key.decode("utf-8"): value.decode("utf-8")
                    for item in sourced.stdout.split(b"\0")
                    if item and b"=" in item
                    for key, value in [item.split(b"=", 1)]
                }
            logger.warning("Could not source OSS CAD Suite environment: %s", sourced.stderr)

        environment = os.environ.copy()
        environment["PATH"] = str(suite_bin) + os.pathsep + environment.get("PATH", "")
        return environment

    def _acquire_sandbox_lock(self) -> Any:
        """Serialize destructive genchecks/SBY work within one formal sandbox.

        ``genchecks.py`` recreates ``cores/<cpu>/checks``.  A retry or a
        second graph invocation sharing the same run directory must therefore
        wait instead of deleting files underneath a live SBY process.
        """
        self.core_dir.mkdir(parents=True, exist_ok=True)
        handle = (self.core_dir / ".lace-formal-run.lock").open("a+")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return handle

    @staticmethod
    def _release_sandbox_lock(handle: Any | None) -> None:
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def _ensure_genchecks_real(self) -> None:
        """Validate genchecks.py without ever modifying the source checkout."""
        genchecks = self.riscv_formal_dir / "checks" / "genchecks.py"
        if not genchecks.exists():
            raise FileNotFoundError(f"genchecks.py not found: {genchecks}")

        content = genchecks.read_text(encoding="utf-8").strip()
        # A real genchecks.py is much larger than a stub and contains the
        # canonical header. Source checkout repair is intentionally not done
        # here: the formal runner must never mutate a shared submodule.
        if len(content) < 200 or "Claire Xenia Wolf" not in content:
            raise RuntimeError(f"genchecks.py is not a trusted upstream file: {genchecks}")

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
            timeout=LACEConfig.RISCV_FORMAL_GENCHECKS_TIMEOUT,
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
            engine_section = re.search(
                r"(\[engines\]\n)(.*?)(?=\n\[|\Z)",
                content,
                re.DOTALL,
            )
            if not engine_section:
                continue
            engine_body, count = pattern.subn(replacement, engine_section.group(2))
            new_content = (
                content[:engine_section.start(2)]
                + engine_body
                + content[engine_section.end(2):]
            )
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
                env=self._sby_environment(),
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
        generated_instruction_checks: bool = False,
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
            generated_instruction_checks: Add every generated ``insn_*_ch0``
                proof job.  Used for baseline verification so the gate follows
                the installed riscv-formal version instead of a stale list.

        Returns:
            dict with keys: passed (bool), results (list), total_time (float), error (str)
        """
        if check_names is None:
            raise ValueError("check_names must be provided; use BASELINE_CHECKS for baseline")

        lock_handle: Any | None = None
        try:
            if not is_formal_sandbox(self.riscv_formal_dir):
                raise RuntimeError(
                    "Refusing to run riscv-formal outside a LACE per-run sandbox"
                )
            lock_handle = self._acquire_sandbox_lock()

            if not update_checks_cfg(self.core_dir, base_isa):
                cfg_path = self.core_dir / "checks.cfg"
                if not cfg_path.exists() or not re.search(
                    rf"^isa\s+{re.escape(base_isa)}$",
                    cfg_path.read_text(encoding="utf-8"),
                    flags=re.MULTILINE,
                ):
                    raise RuntimeError(f"Could not set base ISA in {cfg_path}")

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
            self._patch_checks_cfg_for_workspace_sources(workspace_dir, cpu_top_file)
            self._apply_private_core_adapter(workspace_dir)
            self._patch_checks_cfg_for_extra_files(extra_verilog_files)
            self._run_genchecks()
            self._patch_solver()

            selected_checks = list(check_names)
            if generated_instruction_checks:
                generated = sorted(
                    path.stem for path in self.checks_dir.glob("insn_*_ch0.sby")
                )
                if not generated:
                    raise RuntimeError(
                        "genchecks.py generated no insn_*_ch0 baseline proof jobs"
                    )
                selected_checks.extend(generated)
            # Preserve caller order (cover first) while avoiding a duplicate
            # when a caller explicitly named one of the generated checks.
            selected_checks = list(dict.fromkeys(selected_checks))

            results: list[FormalCheckResult] = []
            all_passed = True
            total_time = 0.0

            for name in selected_checks:
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
        finally:
            self._release_sandbox_lock(lock_handle)

    def run_baseline_checks(
        self,
        workspace_dir: str,
        cpu_top_file: str,
        extra_verilog_files: list[str] | None = None,
        base_isa: str | None = None,
    ) -> dict[str, Any]:
        """Run all generated baseline instruction checks plus ``cover``.

        The instruction list is taken from the current riscv-formal
        ``genchecks.py`` output.  This is intentionally not a hand-maintained
        subset: missing or renamed generated proof jobs cannot be mistaken for
        a successful baseline run.
        """
        return self.run_checks(
            workspace_dir,
            cpu_top_file,
            check_names=BASELINE_CHECKS,
            extra_verilog_files=extra_verilog_files,
            base_isa=base_isa or BASELINE_ISA_BY_CPU.get(self.cpu_name, "rv32imc"),
            generated_instruction_checks=True,
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
