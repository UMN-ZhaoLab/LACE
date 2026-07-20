"""Syntax and function check implementations."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

from typing import Any

from src.config import LACEConfig, get_env
from src.formal.riscv_formal_runner import RiscvFormalRunner
from src.formal.sandbox import prepare_riscv_formal_sandbox
from src.state_types import WorkflowState, ensure_state


def verilator_syntax_check(
    content_or_path: str,
    extra_args: list[str] | None = None,
    include_dir: str | None = None,
    include_dirs: list[str] | None = None,
    source_files: list[str] | None = None,
    top_module: str | None = None,
    verilator_std: str | None = None,
    verilator_waive_flags: list[str] | None = None,
) -> tuple[bool, str]:
    """Check SystemVerilog syntax using Verilator."""
    resolved_include = include_dir or get_env("SV_FILES_DIR", ".")

    if source_files:
        files = source_files
        tmp_file = None
    else:
        try:
            is_path = Path(content_or_path).exists()
        except OSError:
            is_path = False

        if is_path:
            files = [content_or_path]
            tmp_file = None
        else:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".sv")
            tmp_file.write(content_or_path.encode("utf-8"))
            tmp_file.flush()
            tmp_file.close()
            files = [tmp_file.name]

    std_flag = verilator_std or "+1364-2005ext+.v"
    waive_flags = verilator_waive_flags or ["--Wno-MULTITOP"]

    cmd = [
        "verilator",
        "--lint-only",
        f"-I{resolved_include}",
        std_flag,
    ]
    for directory in include_dirs or []:
        flag = f"-I{directory}"
        if flag not in cmd:
            cmd.append(flag)
    if top_module:
        cmd.extend(["--top-module", top_module])
    cmd.extend(waive_flags)
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(files)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("Verilator not found in PATH") from exc
    finally:
        if tmp_file is not None:
            os.unlink(tmp_file.name)

    success = proc.returncode == 0
    output = (proc.stdout or "") + (proc.stderr or "")
    return success, output


def _project_verilator_inputs(state: WorkflowState) -> tuple[list[str], list[str], str] | None:
    """Return a practical RTL source closure for registered multi-file CPUs."""
    if state.cpu_name == "picorv32" or not state.workspace_dir or not state.cpu_top_file:
        return None

    workspace = Path(state.workspace_dir)
    roots: list[Path]
    if state.cpu_name == "e203_hbirdv2":
        roots = [workspace / "rtl" / "e203"]
    elif state.cpu_name == "cv32e40x":
        roots = [workspace / "rtl"]
    elif state.cpu_name == "ibex":
        roots = [workspace / "rtl"]
    else:
        roots = [workspace]

    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(root.rglob("*.v"))
            paths.extend(root.rglob("*.sv"))
    # Generated companion modules such as lace_arithmetic.v live at the run
    # workspace root rather than under the CPU's rtl/ directory.
    for companion in (workspace / "lace_arithmetic.v", workspace / "lace_arithmetic.sv"):
        if companion.exists():
            paths.append(companion)
    if state.cpu_name == "cv32e40x":
        clock_gate = workspace / "bhv" / "cv32e40x_sim_clock_gate.sv"
        if clock_gate.exists():
            paths.append(clock_gate)
    elif state.cpu_name == "ibex":
        prim_root = workspace / "vendor" / "lowrisc_ip" / "ip"
        ibex_prim_sources = (
            prim_root / "prim" / "rtl" / "prim_secded_pkg.sv",
            prim_root / "prim" / "rtl" / "prim_cipher_pkg.sv",
            prim_root / "prim" / "rtl" / "prim_count_pkg.sv",
            prim_root / "prim" / "rtl" / "prim_mubi_pkg.sv",
            prim_root / "prim" / "rtl" / "prim_util_pkg.sv",
            prim_root / "prim_generic" / "rtl" / "prim_ram_1p_pkg.sv",
        )
        paths.extend(path for path in ibex_prim_sources if path.exists())

    top_path = workspace / state.cpu_top_file
    if top_path.exists() and top_path not in paths:
        paths.append(top_path)
    if not paths:
        return None

    # SystemVerilog packages must be parsed before modules that import them.
    unique_paths = sorted(
        set(path.resolve() for path in paths),
        key=lambda path: (0 if path.name.endswith("_pkg.sv") else 1, str(path)),
    )
    include_dirs = sorted({str(path.parent) for path in unique_paths})
    if state.cpu_name == "ibex":
        extra_includes = (
            workspace / "vendor" / "lowrisc_ip" / "ip" / "prim" / "rtl",
            workspace / "vendor" / "lowrisc_ip" / "dv" / "sv" / "dv_utils",
        )
        include_dirs.extend(str(path.resolve()) for path in extra_includes if path.exists())
    top_module = Path(state.cpu_top_file).stem
    return [str(path) for path in unique_paths], include_dirs, top_module


def _extract_ports(code: str) -> set[str]:
    """Naively extract top-level port names from a Verilog module header."""
    match = re.search(r"module\s+\w+\s*\((.*?)\)\s*;", code, re.DOTALL)
    if not match:
        return set()
    ports_text = match.group(1)
    ports: set[str] = set()
    for part in ports_text.split(","):
        part = part.strip()
        if not part:
            continue
        tokens = part.split()
        if tokens:
            ports.add(tokens[-1].strip())
    return ports


def check_semantic_ports(state: WorkflowState) -> WorkflowState:
    """Warn if the modified interface code drops original top-level ports."""
    target_dir = state.workspace_dir or state.cpu_dir
    if not state.interface_code or not target_dir or not state.cpu_top_file:
        return state
    try:
        original = (Path(state.cpu_dir) / state.cpu_top_file).read_text(encoding="utf-8")
    except Exception:
        return state

    orig_ports = _extract_ports(original)
    new_ports = _extract_ports(state.interface_code)
    missing = orig_ports - new_ports
    if missing:
        notes = list(state.notes)
        notes.append(
            f"Semantic warning: ports missing after modification: {', '.join(sorted(missing))}"
        )
        return state.model_copy(update={"notes": notes})
    return state


def check_interface_syntax(state: WorkflowState | dict[str, Any]) -> WorkflowState:
    """Check if interface code passes Verilator syntax check."""
    state = ensure_state(state)
    # Discovery failures occur before any patch is applied.  They are not
    # Verilator failures and retrying the syntax gate with empty code would
    # erase the source-evidence diagnostic.
    if state.needs_review and state.last_error.startswith("RTL discovery"):
        return state
    if not state.interface_code:
        retry = state.interface_retry_count + 1
        needs_review = retry > LACEConfig.MAX_TASK_RETRIES
        return state.model_copy(
            update={
                "interface_syntax_ok": False,
                "advance_op": False,
                "interface_retry_count": retry,
                "needs_review": needs_review,
                "last_error": "Interface writer returned empty code",
            }
        )

    include_dir = state.workspace_dir or state.sv_include_dir or None
    project_inputs = _project_verilator_inputs(state)
    if project_inputs:
        source_files, include_dirs, top_module = project_inputs
        ok, out = verilator_syntax_check(
            state.interface_code,
            include_dir=include_dir,
            include_dirs=include_dirs,
            source_files=source_files,
            top_module=top_module,
            verilator_std=state.verilator_std or None,
            verilator_waive_flags=state.verilator_waive_flags or None,
        )
    else:
        ok, out = verilator_syntax_check(
            state.interface_code,
            include_dir=include_dir,
            verilator_std=state.verilator_std or None,
            verilator_waive_flags=state.verilator_waive_flags or None,
        )
    notes = list(state.notes)
    if not ok:
        notes.append("Interface syntax check failed")
        if out:
            notes.append(out)
        retry = state.interface_retry_count + 1
        needs_review = retry > LACEConfig.MAX_TASK_RETRIES
        # Include the Verilator output in last_error so the retry prompt can
        # show the LLM exactly what went wrong.
        error_detail = "Interface syntax check failed"
        if out:
            error_detail += f"\n\nVerilator output:\n{out[:2000]}"
        return state.model_copy(
            update={
                "interface_syntax_ok": False,
                "notes": notes,
                "advance_op": False,
                "interface_retry_count": retry,
                "needs_review": needs_review,
                "last_error": error_detail,
            }
        )

    # Guard: if all tasks already processed, just mark syntax ok.
    if state.hdl_index >= len(state.hdl_tasks):
        return state.model_copy(
            update={
                "interface_syntax_ok": True,
                "notes": notes,
                "advance_op": False,
                "interface_retry_count": 0,
            }
        )

    # Advance hdl_index to the first task of the NEXT op (or end of list).
    next_index = state.hdl_index + 1
    if (
        state.hdl_task_op_index_map
        and len(state.hdl_task_op_index_map) == len(state.hdl_tasks)
    ):
        current_op_index = state.hdl_task_op_index_map[state.hdl_index]
        while (
            next_index < len(state.hdl_tasks)
            and state.hdl_task_op_index_map[next_index] == current_op_index
        ):
            next_index += 1

    if next_index >= len(state.hdl_tasks):
        return state.model_copy(
            update={
                "hdl_index": next_index,
                "interface_syntax_ok": True,
                "notes": notes,
                "advance_op": False,
                "interface_retry_count": 0,
            }
        )

    return state.model_copy(
        update={
            "hdl_index": next_index,
            "interface_syntax_ok": True,
            "notes": notes,
            "advance_op": False,
            "interface_retry_count": 0,
        }
    )


def check_arithmetic_syntax(state: WorkflowState | dict[str, Any]) -> WorkflowState:
    """Check if arithmetic code passes Verilator syntax check.

    A failed attempt is retried within the dedicated Verilator budget. The
    serial graph must never integrate arithmetic RTL that failed this gate.
    """
    state = ensure_state(state)
    if not state.arithmetic_code:
        retry = state.arithmetic_retry_count + 1
        return state.model_copy(
            update={
                "arithmetic_syntax_ok": False,
                "arithmetic_retry_count": retry,
                "needs_review": retry > LACEConfig.MAX_VERILATOR_RETRIES,
                "last_error": "Arithmetic writer returned empty code",
            }
        )

    include_dir = state.workspace_dir or state.sv_include_dir or None
    ok, out = verilator_syntax_check(
        state.arithmetic_code,
        include_dir=include_dir,
        verilator_std=state.verilator_std or None,
        verilator_waive_flags=state.verilator_waive_flags or None,
    )
    notes = list(state.notes)
    if not ok:
        notes.append("Arithmetic syntax check failed")
        if out:
            notes.append(out)

    if not ok:
        error_detail = "Arithmetic syntax check failed"
        if out:
            error_detail += f"\n\nVerilator output:\n{out[:2000]}"
        retry = state.arithmetic_retry_count + 1
        return state.model_copy(
            update={
                "arithmetic_syntax_ok": False,
                "arithmetic_retry_count": retry,
                "notes": notes,
                "needs_review": retry > LACEConfig.MAX_VERILATOR_RETRIES,
                "last_error": error_detail,
            }
        )
    return state.model_copy(
        update={
            "arithmetic_syntax_ok": True,
            "arithmetic_retry_count": 0,
            "needs_review": False,
            "last_error": "",
            "notes": notes,
        }
    )


def _extract_expected_signals(task: str) -> set[str]:
    """Extract Verilog signal names mentioned in an HDL task description."""
    signals: set[str] = set()
    # Match SCAL-style port names: Name_#_i/o
    signals.update(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*_[0-9]+_[oi]\b", task))
    # Match ISAX internal signals
    signals.update(re.findall(r"\bISAX_[A-Za-z0-9_]+\b", task))
    return signals


def _check_interface_completeness(state: WorkflowState) -> tuple[bool, str]:
    """Check that every HDL task's expected signals appear in the modified code.

    Returns (ok, error_message).
    """
    if not state.hdl_tasks:
        return True, ""

    code = state.interface_code or ""
    if not code and state.workspace_dir and state.cpu_top_file:
        try:
            code = (Path(state.workspace_dir) / state.cpu_top_file).read_text(
                encoding="utf-8"
            )
        except Exception:
            pass

    if not code:
        return True, ""  # Cannot check without code

    missing_signals: list[str] = []
    for task in state.hdl_tasks:
        expected = _extract_expected_signals(task)
        for sig in expected:
            # The signal must appear as a declaration, assignment target,
            # or in an expression--not just as a substring of another word.
            pattern = rf"\b{re.escape(sig)}\b"
            if not re.search(pattern, code):
                missing_signals.append(sig)

    if missing_signals:
        return False, (
            f"Interface code is missing expected signals: "
            f"{', '.join(sorted(set(missing_signals)))}. "
            f"The LLM may have omitted body logic (decode, assignments, etc.)."
        )
    return True, ""


def _is_valid_rtl(rtl_path: Path, cpu_name: str) -> bool:
    """Check if the RTL file looks like a real CPU (not a mock stub)."""
    if not rtl_path.exists():
        return False

    try:
        content = rtl_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False

    # Explicit mock markers (used by test fixtures and the mock LLM) always
    # disqualify a file from formal verification — a mock must never be
    # mistaken for real RTL and silently passed.
    mock_markers = ("// mock generated code", "mock generated code", "MOCK_RTL_PLACEHOLDER")
    if any(marker in content for marker in mock_markers):
        return False

    signatures = {
        "picorv32": {"min_size": 50000, "patterns": ["module picorv32"]},
        "e203_hbirdv2": {"min_size": 10000, "patterns": ["module e203_cpu_top"]},
        "cv32e40x": {"min_size": 10000, "patterns": ["module cv32e40x_core"]},
        "ibex": {"min_size": 10000, "patterns": ["module ibex_top"]},
    }

    sig = signatures.get(cpu_name)
    if sig is None:
        # Unknown CPU: generic heuristic
        if rtl_path.stat().st_size < 10000:
            return False
        return f"module {cpu_name}" in content

    if rtl_path.stat().st_size < sig["min_size"]:
        return False
    return all(pat in content for pat in sig["patterns"])


def _prepare_riscv_formal_runner(state: WorkflowState) -> tuple[RiscvFormalRunner, list[str]] | None:
    """Validate workspace and return a configured runner plus extra Verilog files.

    Returns None if verification should be skipped (missing workspace, mock RTL, etc.).
    """
    if not state.cpu_name or not state.workspace_dir or not state.cpu_top_file:
        return None

    ws_path = Path(state.workspace_dir)
    rtl_path = ws_path / state.cpu_top_file
    if not ws_path.exists() or not rtl_path.exists():
        return None

    if not _is_valid_rtl(rtl_path, state.cpu_name):
        return None

    extra_verilog_files: list[str] = []
    for fname in ["lace_arithmetic.v", "lace_arithmetic.sv"]:
        if (ws_path / fname).exists():
            extra_verilog_files.append(fname)

    sandbox = prepare_riscv_formal_sandbox(
        run_id=state.run_id,
        cpu_name=state.cpu_name,
        workspace_dir=state.workspace_dir,
    )
    return (
        RiscvFormalRunner(
            cpu_name=state.cpu_name,
            riscv_formal_dir=str(sandbox),
        ),
        extra_verilog_files,
    )


def _skipped_formal_state(state: WorkflowState, reason: str) -> WorkflowState:
    """Build a state that records formal verification was skipped.

    A skip is NOT a pass and cannot be repaired by later generation nodes.
    Escalate it immediately so the graph stops without spending another model
    call or pretending that functional verification ran.
    """
    notes = list(state.notes)
    notes.append(f"riscv-formal: skipped ({reason})")
    return state.model_copy(
        update={
            "function_ok": False,
            "advance_op": False,
            "formal_skipped": True,
            "formal_terminal": True,
            "needs_review": True,
            "last_error": f"Formal verification skipped: {reason}",
            "notes": notes,
        }
    )


def _run_riscv_formal_baseline(state: WorkflowState) -> WorkflowState:
    """Run only riscv-formal baseline checks (original ISA, no custom instructions).

    Used by the original_function_checker to verify that interface modifications
    have not broken existing RV32I behavior.
    """
    try:
        prepared = _prepare_riscv_formal_runner(state)
    except Exception as exc:
        return _skipped_formal_state(state, f"sandbox setup failed: {exc}")
    if prepared is None:
        return _skipped_formal_state(state, "no workspace or untrusted RTL")

    runner, extra_verilog_files = prepared
    baseline_result = runner.run_baseline_checks(
        workspace_dir=state.workspace_dir,
        cpu_top_file=state.cpu_top_file,
        extra_verilog_files=extra_verilog_files,
    )

    notes = list(state.notes)
    passed_count = sum(1 for r in baseline_result["results"] if r["passed"])
    notes.append(
        f"riscv-formal baseline: {passed_count}/"
        f"{len(baseline_result['results'])} passed ({baseline_result['total_time']:.1f}s)"
    )

    baseline_executed_zero_checks = baseline_result["passed"] and not baseline_result.get("results")
    if baseline_executed_zero_checks:
        baseline_result = dict(baseline_result)
        baseline_result["passed"] = False
        baseline_result["error"] = "riscv-formal baseline executed zero checks"

    if not baseline_result["passed"]:
        failed = [r for r in baseline_result["results"] if not r["passed"]]
        deterministic_failure = any(
            str(item.get("error", "")) == "Assertion failed" for item in failed
        )
        error_msg = (
            f"riscv-formal baseline failed: {len(failed)}/{len(baseline_result['results'])} checks failed. "
            f"Errors: {'; '.join(r['error'] for r in failed if r['error'])[:200]}"
        )
        if baseline_result["error"]:
            error_msg = f"riscv-formal baseline error: {baseline_result['error']}"
        notes.append(error_msg)
        return state.model_copy(
            update={
                "function_ok": False,
                "advance_op": False,
                "needs_review": True,
                "last_error": error_msg,
                "formal_check_error": error_msg,
                "formal_check_passed": False,
                "formal_terminal": baseline_executed_zero_checks or deterministic_failure,
                "formal_check_results": {"baseline": baseline_result, "custom": {}},
                "notes": notes,
            }
        )

    return state.model_copy(
        update={
            "function_ok": True,
            "advance_op": False,
            "formal_check_results": {"baseline": baseline_result, "custom": {}},
            "notes": notes,
        }
    )


def _run_riscv_formal_check(state: WorkflowState) -> WorkflowState:
    """Run full riscv-formal verification: baseline + custom instruction checks.

    Used by the final_function_checker after the custom instruction model has been
    generated and integrated.
    """
    try:
        prepared = _prepare_riscv_formal_runner(state)
    except Exception as exc:
        return _skipped_formal_state(state, f"sandbox setup failed: {exc}")
    if prepared is None:
        return _skipped_formal_state(state, "no workspace/cpu info or untrusted RTL")

    runner, extra_verilog_files = prepared
    custom_insns = list(state.custom_insn_names)

    # Phase 1: Baseline checks (must pass). original_function_checker runs on
    # this same integrated workspace immediately before model generation, so a
    # complete passing result can be reused. This avoids rerunning every base
    # checker for each retry of a custom instruction check.
    previous_baseline = state.formal_check_results.get("baseline", {})
    if previous_baseline.get("passed") and previous_baseline.get("results"):
        baseline_result = previous_baseline
    else:
        baseline_result = runner.run_baseline_checks(
            workspace_dir=state.workspace_dir,
            cpu_top_file=state.cpu_top_file,
            extra_verilog_files=extra_verilog_files,
        )

    notes = list(state.notes)
    baseline_executed_zero_checks = baseline_result["passed"] and not baseline_result.get("results")
    baseline_passed = baseline_result["passed"] and bool(baseline_result.get("results"))
    if baseline_executed_zero_checks:
        baseline_result = dict(baseline_result)
        baseline_result["passed"] = False
        baseline_result["error"] = "riscv-formal baseline executed zero checks"
    notes.append(
        f"riscv-formal baseline: {sum(1 for r in baseline_result['results'] if r['passed'])}/"
        f"{len(baseline_result['results'])} passed ({baseline_result['total_time']:.1f}s)"
    )

    if not baseline_passed:
        failed = [r for r in baseline_result["results"] if not r["passed"]]
        error_msg = (
            f"riscv-formal baseline failed: {len(failed)}/{len(baseline_result['results'])} checks failed. "
            f"Errors: {'; '.join(r['error'] for r in failed if r['error'])[:200]}"
        )
        if baseline_result["error"]:
            error_msg = f"riscv-formal baseline error: {baseline_result['error']}"
        notes.append(error_msg)
        return state.model_copy(
            update={
                "function_ok": False,
                "advance_op": False,
                "needs_review": True,
                "last_error": error_msg,
                "formal_check_error": error_msg,
                "formal_check_passed": False,
                "formal_terminal": baseline_executed_zero_checks,
                "formal_check_results": {"baseline": baseline_result, "custom": {}},
                "notes": notes,
            }
        )

    # Phase 2: Custom instruction checks
    custom_results: dict[str, Any] = {}
    custom_failures: list[str] = []
    missing_custom_checks: list[str] = []
    deterministic_custom_failure = False

    for insn in custom_insns:
        insn_result = runner.run_custom_instruction_checks(
            workspace_dir=state.workspace_dir,
            cpu_top_file=state.cpu_top_file,
            insn_name=insn,
            extra_verilog_files=extra_verilog_files,
        )
        custom_results[insn] = insn_result
        passed_count = sum(1 for r in insn_result["results"] if r["passed"])
        notes.append(
            f"riscv-formal {insn}: {passed_count}/{len(insn_result['results'])} passed "
            f"({insn_result['total_time']:.1f}s)"
        )
        expected_check = f"insn_{insn}_ch0"
        executed_checks = {
            str(item.get("name", "")) for item in insn_result.get("results", [])
        }
        structural_errors = {
            str(item.get("error", ""))
            for item in insn_result.get("results", [])
            if str(item.get("error", "")) in {".sby file not found"}
        }
        expected_check_missing = expected_check not in executed_checks or bool(structural_errors)
        if any(
            str(item.get("error", "")) == "Assertion failed"
            for item in insn_result.get("results", [])
        ):
            deterministic_custom_failure = True
        if not insn_result["passed"] or expected_check_missing:
            custom_failures.append(insn)
            if expected_check_missing:
                missing_custom_checks.append(expected_check)
                notes.append(
                    f"riscv-formal {insn}: expected check {expected_check} was not executed"
                )

    if custom_failures:
        if missing_custom_checks:
            error_msg = (
                "riscv-formal did not execute required custom checks: "
                f"{', '.join(missing_custom_checks)}"
            )
        else:
            error_msg = (
                f"riscv-formal custom instruction checks failed: {', '.join(custom_failures)}"
            )
        notes.append(error_msg)
        return state.model_copy(
            update={
                "function_ok": False,
                "advance_op": False,
                "needs_review": True,
                "last_error": error_msg,
                "formal_check_error": error_msg,
                "formal_check_passed": False,
                "formal_terminal": bool(missing_custom_checks) or deterministic_custom_failure,
                "formal_check_results": {"baseline": baseline_result, "custom": custom_results},
                "notes": notes,
            }
        )

    return state.model_copy(
        update={
            "function_ok": True,
            "advance_op": False,
            "formal_check_passed": True,
            "formal_check_results": {"baseline": baseline_result, "custom": custom_results},
            "notes": notes,
        }
    )


def function_check(state: WorkflowState | dict[str, Any]) -> WorkflowState:
    """Original function checker: verify baseline ISA still works after integration.

    After interface modifications and arithmetic integration are complete, this node
    runs riscv-formal baseline checks (original RV32I ISA only, no custom
    instructions). This catches regressions in existing instructions before the
    final checker validates the new instruction.
    """
    state = ensure_state(state)
    if state.needs_review:
        # An upstream failure (or a prior formal failure escalated by this
        # node) already flagged review. Mark formal as skipped so formal_gate
        # does not treat this as a retryable formal failure and clear
        # needs_review — BUT only if we have not already run a real formal
        # check. If formal_check_results is populated, formal actually ran
        # and failed; that is not a skip and must not be relabeled as one.
        already_ran = bool(state.formal_check_results)
        return state.model_copy(
            update={
                "advance_op": False,
                "formal_skipped": False if already_ran else True,
            }
        )

    if not state.interface_syntax_ok:
        return state.model_copy(
            update={
                "function_ok": False,
                "advance_op": False,
                "needs_review": True,
                "formal_skipped": True,
                "last_error": "Interface syntax check did not pass; baseline formal check skipped.",
            }
        )

    # If arithmetic code exists, it must also pass syntax check
    if state.arithmetic_code and not state.arithmetic_syntax_ok:
        return state.model_copy(
            update={
                "function_ok": False,
                "advance_op": False,
                "needs_review": True,
                "formal_skipped": True,
                "last_error": "Arithmetic syntax check did not pass; baseline formal check skipped.",
            }
        )

    # Run baseline riscv-formal to verify original ISA is not broken.
    return _run_riscv_formal_baseline(state)


def final_function_check(state: WorkflowState | dict[str, Any]) -> WorkflowState:
    """Final function check: run riscv-formal after all ops are complete.

    This is the FINAL function checker: it runs riscv-formal bounded model
    checking to verify the generated ISA extension against the RISC-V spec.
    It tests both existing RV32I instructions and any newly added instructions.

    A formal skip (no workspace / untrusted RTL / missing sby) is treated as a
    failure here: we refuse to report success without real verification.

    If needs_review is already set on entry (e.g. the graph re-invoked this
    node after the formal gate), preserve the existing function_ok rather than
    forcing it True — otherwise a re-invocation would mask the escalation.
    """
    state = ensure_state(state)
    if state.needs_review:
        # An upstream failure (or a prior formal failure escalated by this
        # node) already flagged review. Mark formal as skipped so formal_gate
        # does not treat this as a retryable formal failure and clear
        # needs_review — BUT only if we have not already run a real formal
        # check. If formal_check_results is populated, formal actually ran
        # and failed; that is not a skip and must not be relabeled as one.
        already_ran = bool(state.formal_check_results)
        return state.model_copy(
            update={
                "advance_op": False,
                "formal_skipped": False if already_ran else True,
            }
        )

    if not state.interface_syntax_ok:
        return state.model_copy(
            update={
                "function_ok": False,
                "advance_op": False,
                "needs_review": True,
                "formal_skipped": True,
                "last_error": "Interface syntax check did not pass; final formal check skipped.",
            }
        )

    # If arithmetic code exists, it must also pass syntax check
    if state.arithmetic_code and not state.arithmetic_syntax_ok:
        return state.model_copy(
            update={
                "function_ok": False,
                "advance_op": False,
                "needs_review": True,
                "formal_skipped": True,
                "last_error": "Arithmetic syntax check did not pass; final formal check skipped.",
            }
        )

    # A successful baseline alone must never be reported as successful custom
    # instruction verification. Model generation is a required stage, and at
    # least one named custom instruction must reach riscv-formal.
    if not state.custom_insn_names or not state.insn_model_code.strip():
        notes = list(state.notes)
        error = "Custom instruction model was not generated; custom formal checks cannot run."
        notes.append(error)
        return state.model_copy(
            update={
                "function_ok": False,
                "advance_op": False,
                "formal_check_passed": False,
                "formal_check_error": error,
                "formal_terminal": True,
                "needs_review": True,
                "last_error": error,
                "notes": notes,
            }
        )

    # Run riscv-formal verification
    result = _run_riscv_formal_check(state)

    # Escalate any skip into a review: never claim success on unverified code.
    if not result.formal_check_passed:
        notes = list(result.notes)
        if result.formal_skipped:
            notes.append(
                "Final check: riscv-formal was skipped, so the integration is "
                "unverified. Flagging for review."
            )
        return result.model_copy(
            update={
                "needs_review": True,
                "function_ok": False,
                "last_error": result.last_error or "Formal verification skipped",
                "notes": notes,
            }
        )
    return result
