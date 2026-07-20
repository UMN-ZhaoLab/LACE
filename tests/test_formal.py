"""Unit tests for riscv-formal integration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.config import LACEConfig
from src.formal.insn_model import (
    normalize_rd_wdata_x0,
    register_custom_instruction,
    write_insn_model,
)
from src.formal.riscv_formal_runner import FormalCheckResult, RiscvFormalRunner
from src.formal.e203_rvfi_adapter import apply_e203_rvfi_adapter
from src.formal.isa_manager import configure_riscv_formal_for_custom_instructions
from src.formal.sandbox import (
    SANDBOX_MARKER,
    formal_sandbox_path,
    prepare_riscv_formal_sandbox,
)
from src.state_types import WorkflowState


def _create_formal_source(root: Path) -> Path:
    """Create the minimal source tree needed by the sandbox tests."""
    source = root / "source-riscv-formal"
    checks = source / "checks"
    insns = source / "insns"
    core = source / "cores" / "picorv32"
    checks.mkdir(parents=True)
    insns.mkdir(parents=True)
    core.mkdir(parents=True)
    (checks / "genchecks.py").write_text(
        "# Copyright (C) 2017  Claire Xenia Wolf\n" + "print('ok')\n" * 50,
        encoding="utf-8",
    )
    (insns / "isa_rv32imc.txt").write_text("add\n", encoding="utf-8")
    (core / "checks.cfg").write_text(
        "[options]\nisa rv32imc\n\n[verilog-files]\n"
        "@basedir@/cores/@core@/picorv32.v\n",
        encoding="utf-8",
    )
    (core / "picorv32.v").write_text(
        "module picorv32; endmodule\n", encoding="utf-8"
    )
    generated = core / "checks"
    generated.mkdir()
    (generated / "stale.sby").write_text("stale", encoding="utf-8")
    return source


class TestInsnModel(unittest.TestCase):
    def test_register_existing_model(self) -> None:
        """register_custom_instruction returns path if file exists upstream."""
        riscv_formal_dir = Path(__file__).resolve().parent.parent / "tools" / "riscv-formal"
        if not (riscv_formal_dir / "insns" / "insn_rol.v").exists():
            self.skipTest("riscv-formal submodule not initialized")
        path = register_custom_instruction("rol", riscv_formal_dir)
        self.assertIsNotNone(path)
        assert path is not None
        self.assertTrue(path.exists())
        self.assertIn("insn_rol.v", str(path))

    def test_register_missing_model(self) -> None:
        """register_custom_instruction returns None for nonexistent model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = register_custom_instruction("nonexistent_insn", tmpdir)
            self.assertIsNone(path)

    def test_write_insn_model(self) -> None:
        """write_insn_model writes Verilog code to insns dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code = "// DO NOT EDIT\nmodule rvfi_insn_test ();\nendmodule\n"
            path = write_insn_model("test", code, tmpdir)
            self.assertTrue(path.exists())
            self.assertIn("insn_test.v", str(path))
            self.assertEqual(path.read_text(), code)

    def test_normalize_rd_wdata_masks_x0(self) -> None:
        code = "assign spec_rd_wdata = result;"
        normalized = normalize_rd_wdata_x0(code)
        self.assertIn("(spec_rd_addr != 0) ? (result)", normalized)
        self.assertIn("{`RISCV_FORMAL_XLEN{1'b0}}", normalized)

    def test_normalize_rd_wdata_is_idempotent(self) -> None:
        code = "assign spec_rd_wdata = spec_rd_addr ? result : 0;"
        self.assertEqual(normalize_rd_wdata_x0(code), code)


class TestRiscvFormalRunner(unittest.TestCase):
    def test_e203_rvfi_adapter_is_private_and_commit_aligned(self) -> None:
        """The e203 overlay must not edit its source and must use EXU commit data."""
        source = Path(__file__).resolve().parent.parent / "cpu_prototype/e203_hbirdv2/e203_hbirdv2.v"
        if not source.is_file():
            self.skipTest("e203 prototype submodule not initialized")
        original = source.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "e203_hbirdv2.v"
            apply_e203_rvfi_adapter(source, destination)
            overlay = destination.read_text(encoding="utf-8")
        self.assertEqual(source.read_text(encoding="utf-8"), original)
        self.assertIn("rvfi_insn <= e203_rvfi_longp_valid ?", overlay)
        self.assertIn("rvfi_pc_rdata <= e203_rvfi_longp_valid ?", overlay)
        self.assertIn("rvfi_load_rmask(e203_rvfi_longp_instr[14:12]", overlay)
        self.assertIn("rvfi_ret_instr = rvfi_instr_r[ret_ptr]", overlay)
        self.assertIn("rvfi_mem_ena & (rvfi_mem_ptr == i)", overlay)
        self.assertIn(".alu_cmt_instr(alu_cmt_instr)", overlay)

    def test_runner_init(self) -> None:
        runner = RiscvFormalRunner(cpu_name="picorv32")
        self.assertEqual(runner.cpu_name, "picorv32")
        self.assertTrue(
            runner.core_dir.exists()
            or str(runner.core_dir).endswith("cores/picorv32")
        )

    def test_copy_rtl_missing_file(self) -> None:
        runner = RiscvFormalRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                runner._copy_rtl(tmpdir, "nonexistent.v")

    def test_cv32e40x_config_paths_and_slang_toolchain_are_private(self) -> None:
        """CV32E40X Slang source paths resolve in the per-run formal sandbox."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            suite = root / "oss-cad-suite"
            plugin = suite / "share" / "yosys" / "plugins" / "slang.so"
            plugin.parent.mkdir(parents=True)
            plugin.write_text("", encoding="utf-8")
            (suite / "bin").mkdir()
            (suite / "bin" / "yosys").write_text("", encoding="utf-8")
            (suite / "bin" / "sby").write_text("", encoding="utf-8")
            (suite / "environment").write_text(
                'export VIRTUAL_ENV="$(dirname "${BASH_SOURCE[0]}")"\n'
                'export PATH="$VIRTUAL_ENV/bin:$PATH"\n',
                encoding="utf-8",
            )

            rf_dir = root / "formal"
            core_dir = rf_dir / "cores" / "cv32e40x"
            core_dir.mkdir(parents=True)
            (core_dir / "checks.cfg").write_text(
                "[script-defines]\n"
                f"plugin -i {plugin}\n\n"
                "[script-sources]\n"
                "-I @basedir@/../../cpu_prototype/cv32e40x/rtl/include "
                "@basedir@/../../cpu_prototype/cv32e40x/rtl/cv32e40x_core.sv\n",
                encoding="utf-8",
            )
            workspace = root / "workspace"
            (workspace / "rtl" / "include").mkdir(parents=True)
            (workspace / "rtl" / "cv32e40x_core.sv").write_text("", encoding="utf-8")

            runner = RiscvFormalRunner(
                cpu_name="cv32e40x", riscv_formal_dir=str(rf_dir)
            )
            runner._patch_checks_cfg_for_workspace_sources(
                str(workspace), "rtl/cv32e40x_core.sv"
            )

            rewritten = (core_dir / "checks.cfg").read_text(encoding="utf-8")
            self.assertNotIn("cpu_prototype/cv32e40x", rewritten)
            self.assertIn(str(workspace), rewritten)
            environment = runner._sby_environment()
            self.assertIsNotNone(environment)
            assert environment is not None
            self.assertTrue(environment["PATH"].startswith(str(suite / "bin")))

    def test_core_template_prevents_unnecessary_source_closure_rebuild(self) -> None:
        """`@core@.v` already names Picorv32's registered top file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rf_dir = root / "formal"
            core_dir = rf_dir / "cores" / "picorv32"
            core_dir.mkdir(parents=True)
            (core_dir / "checks.cfg").write_text(
                "[verilog-files]\n@basedir@/cores/@core@/@core@.v\n",
                encoding="utf-8",
            )
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "picorv32.v").write_text("module picorv32; endmodule\n")
            (workspace / "unrelated_tb.v").write_text("module unrelated_tb; endmodule\n")

            runner = RiscvFormalRunner(
                cpu_name="picorv32", riscv_formal_dir=str(rf_dir)
            )
            runner._patch_checks_cfg_for_workspace_sources(str(workspace), "picorv32.v")

            rewritten = (core_dir / "checks.cfg").read_text(encoding="utf-8")
            self.assertIn("@core@/@core@.v", rewritten)
            self.assertNotIn("unrelated_tb.v", rewritten)

    def test_solver_patch_does_not_modify_source_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = RiscvFormalRunner(riscv_formal_dir=tmpdir)
            runner.checks_dir = Path(tmpdir) / "checks"
            runner.checks_dir.mkdir()
            sby_file = runner.checks_dir / "test.sby"
            sby_file.write_text(
                "[engines]\nsmtbmc boolector\n\n"
                "[script]\nread -sv scripts/smtbmc/example.v\n",
                encoding="utf-8",
            )

            runner._patch_solver()

            rewritten = sby_file.read_text(encoding="utf-8")
            self.assertIn(f"smtbmc {runner.solver}", rewritten)
            self.assertIn("scripts/smtbmc/example.v", rewritten)

    @patch("src.formal.riscv_formal_runner.subprocess.run")
    def test_run_genchecks_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        with tempfile.TemporaryDirectory() as tmpdir:
            rf_dir = Path(tmpdir) / "riscv-formal"
            core_dir = rf_dir / "cores" / "picorv32"
            core_dir.mkdir(parents=True, exist_ok=True)
            # Create a real-looking genchecks.py so the stub guard does not trigger.
            (rf_dir / "checks" / "genchecks.py").parent.mkdir(parents=True, exist_ok=True)
            (rf_dir / "checks" / "genchecks.py").write_text(
                "# Real genchecks.py\n# Copyright (C) 2017  Claire Xenia Wolf\n"
                + "print('ok')\n" * 50
            )
            runner = RiscvFormalRunner(riscv_formal_dir=str(rf_dir))
            runner._run_genchecks()
            mock_run.assert_called_once()
            self.assertEqual(
                mock_run.call_args.kwargs["timeout"],
                LACEConfig.RISCV_FORMAL_GENCHECKS_TIMEOUT,
            )

    @patch("src.formal.riscv_formal_runner.subprocess.run")
    def test_ensure_genchecks_real_rejects_stub_without_checkout(
        self, mock_run: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rf_dir = Path(tmpdir) / "riscv-formal"
            (rf_dir / "checks").mkdir(parents=True, exist_ok=True)
            (rf_dir / "checks" / "genchecks.py").write_text("# fake\n")

            runner = RiscvFormalRunner(riscv_formal_dir=str(rf_dir))
            with self.assertRaisesRegex(RuntimeError, "not a trusted upstream file"):
                runner._ensure_genchecks_real()
            mock_run.assert_not_called()

    @patch("src.formal.riscv_formal_runner.subprocess.run")
    def test_run_sby_check_pass(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="DONE (PASS, rc=0)\n",
        )
        runner = RiscvFormalRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.checks_dir = Path(tmpdir)
            (runner.checks_dir / "test.sby").write_text("[options]\nmode bmc\n")
            with patch.object(runner, "_sby_environment", return_value=None):
                result = runner._run_sby_check("test")
            self.assertTrue(result.passed)
            self.assertEqual(result.name, "test")

    @patch("src.formal.riscv_formal_runner.subprocess.run")
    def test_run_sby_check_fail(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=2,
            stdout="DONE (FAIL, rc=2)\ncounterexample trace: engine_0/trace.vcd\n",
        )
        runner = RiscvFormalRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.checks_dir = Path(tmpdir)
            (runner.checks_dir / "test.sby").write_text("[options]\nmode bmc\n")
            with patch.object(runner, "_sby_environment", return_value=None):
                result = runner._run_sby_check("test")
            self.assertFalse(result.passed)
            self.assertIn("trace.vcd", result.trace_path)

    @patch("src.formal.riscv_formal_runner.subprocess.run")
    def test_run_checks_integration(self, mock_run: MagicMock) -> None:
        """Test full run_checks with mocked sby."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rf_dir = Path(tmpdir) / "riscv-formal"
            core_dir = rf_dir / "cores" / "picorv32"
            core_dir.mkdir(parents=True, exist_ok=True)
            # Create real-looking genchecks.py so the stub guard does not trigger.
            (rf_dir / "checks" / "genchecks.py").parent.mkdir(parents=True, exist_ok=True)
            (rf_dir / "checks" / "genchecks.py").write_text(
                "# Real genchecks.py\n# Copyright (C) 2017  Claire Xenia Wolf\n"
                + "print('ok')\n" * 50
            )
            (rf_dir / SANDBOX_MARKER).write_text("{}")
            (core_dir / "checks.cfg").write_text(
                "[options]\nisa rv32imc\n\n[verilog-files]\n"
            )
            runner = RiscvFormalRunner(riscv_formal_dir=str(rf_dir))

            def mock_subprocess(cmd, **kwargs):
                # genchecks.py call
                if any("genchecks.py" in str(c) for c in cmd):
                    # Create the .sby file that run_checks expects
                    runner.checks_dir.mkdir(parents=True, exist_ok=True)
                    (runner.checks_dir / "cover.sby").write_text(
                        "[options]\nmode bmc\n"
                    )
                    return MagicMock(returncode=0, stderr="")
                # sby call
                return MagicMock(returncode=0, stdout="DONE (PASS, rc=0)\n")

            mock_run.side_effect = mock_subprocess
            with tempfile.TemporaryDirectory() as ws:
                (Path(ws) / "picorv32.v").write_text("module picorv32; endmodule")
                from src.formal.riscv_formal_runner import BASELINE_CHECKS
                with patch.object(runner, "_sby_environment", return_value=None):
                    result = runner.run_checks(ws, "picorv32.v", check_names=["cover"])
                self.assertTrue(result["passed"])
            self.assertEqual(len(result["results"]), 1)
            self.assertEqual(result["results"][0]["name"], "cover")

    @patch("src.formal.riscv_formal_runner.subprocess.run")
    def test_baseline_uses_generated_instruction_jobs(
        self, mock_run: MagicMock
    ) -> None:
        """Baseline follows genchecks output rather than stale check names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rf_dir = Path(tmpdir) / "riscv-formal"
            core_dir = rf_dir / "cores" / "e203_hbirdv2"
            core_dir.mkdir(parents=True, exist_ok=True)
            checks_script = rf_dir / "checks" / "genchecks.py"
            checks_script.parent.mkdir(parents=True, exist_ok=True)
            checks_script.write_text(
                "# Real genchecks.py\n# Copyright (C) 2017  Claire Xenia Wolf\n"
                + "print('ok')\n" * 50
            )
            (rf_dir / SANDBOX_MARKER).write_text("{}")
            (core_dir / "checks.cfg").write_text(
                "[options]\nisa rv32i\n\n[verilog-files]\n"
            )
            runner = RiscvFormalRunner(
                cpu_name="e203_hbirdv2", riscv_formal_dir=str(rf_dir)
            )

            def mock_subprocess(cmd, **kwargs):
                if any("genchecks.py" in str(part) for part in cmd):
                    runner.checks_dir.mkdir(parents=True, exist_ok=True)
                    for name in ("cover", "insn_add_ch0", "insn_sub_ch0"):
                        (runner.checks_dir / f"{name}.sby").write_text("[options]\n")
                    return MagicMock(returncode=0, stderr="")
                return MagicMock(returncode=0, stdout="DONE (PASS, rc=0)\n")

            mock_run.side_effect = mock_subprocess
            with tempfile.TemporaryDirectory() as ws:
                (Path(ws) / "e203_hbirdv2.v").write_text("module e203_hbirdv2; endmodule")
                with (
                    patch.object(runner, "_sby_environment", return_value=None),
                    patch.object(runner, "_apply_private_core_adapter"),
                ):
                    result = runner.run_baseline_checks(ws, "e203_hbirdv2.v")

            self.assertTrue(result["passed"])
            self.assertEqual(
                [item["name"] for item in result["results"]],
                ["cover", "insn_add_ch0", "insn_sub_ch0"],
            )
            self.assertIn("isa rv32i", (core_dir / "checks.cfg").read_text())

    def test_run_checks_refuses_shared_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = _create_formal_source(Path(tmpdir))
            runner = RiscvFormalRunner(riscv_formal_dir=str(source))
            workspace = Path(tmpdir) / "workspace"
            workspace.mkdir()
            (workspace / "picorv32.v").write_text("changed", encoding="utf-8")
            result = runner.run_checks(
                str(workspace), "picorv32.v", check_names=["cover"]
            )
            self.assertFalse(result["passed"])
            self.assertIn("per-run sandbox", result["error"])
            self.assertEqual(
                (source / "cores" / "picorv32" / "picorv32.v").read_text(),
                "module picorv32; endmodule\n",
            )

    def test_generated_custom_model_gets_private_x_isa_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rf_dir = Path(tmpdir)
            insns = rf_dir / "insns"
            core = rf_dir / "cores" / "picorv32"
            insns.mkdir(parents=True)
            core.mkdir(parents=True)
            (insns / "isa_rv32imc.txt").write_text("add\n", encoding="utf-8")
            (core / "checks.cfg").write_text(
                "[options]\nisa rv32imc\n", encoding="utf-8"
            )

            result = configure_riscv_formal_for_custom_instructions(
                rf_dir, "picorv32", "rv32imc", ["aes32esi"]
            )

            self.assertEqual(result["new_isa"], "rv32imc_Xlace")
            custom_list = insns / "isa_rv32imc_Xlace.txt"
            self.assertTrue(custom_list.exists())
            self.assertEqual(custom_list.read_text().splitlines(), ["add", "aes32esi"])
            self.assertIn("isa rv32imc_Xlace", (core / "checks.cfg").read_text())
            self.assertEqual((insns / "isa_rv32imc.txt").read_text(), "add\n")


class TestFormalSandbox(unittest.TestCase):
    def test_private_mutable_files_and_symlinked_framework(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = _create_formal_source(root)
            workspace = root / "runs" / "run-a" / "workspace"
            workspace.mkdir(parents=True)

            sandbox = prepare_riscv_formal_sandbox(
                run_id="run-a",
                cpu_name="picorv32",
                workspace_dir=str(workspace),
                source_dir=source,
            )

            self.assertEqual(sandbox, workspace.parent / "formal")
            self.assertTrue((sandbox / "checks").is_symlink())
            self.assertFalse((sandbox / "insns").is_symlink())
            self.assertFalse(
                (sandbox / "cores" / "picorv32" / "checks.cfg").is_symlink()
            )
            self.assertTrue(
                (sandbox / "cores" / "picorv32" / "picorv32.v").is_symlink()
            )
            self.assertFalse((sandbox / "cores" / "picorv32" / "checks").exists())

            (sandbox / "insns" / "insn_custom.v").write_text("custom")
            sandbox_cfg = sandbox / "cores" / "picorv32" / "checks.cfg"
            sandbox_cfg.write_text("isa rv32imcZbb\n")
            self.assertFalse((source / "insns" / "insn_custom.v").exists())
            self.assertIn(
                "isa rv32imc",
                (source / "cores" / "picorv32" / "checks.cfg").read_text(),
            )

    def test_rtl_copy_breaks_symlink_without_touching_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = _create_formal_source(root)
            workspace = root / "runs" / "run-a" / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "picorv32.v").write_text("modified RTL\n")
            sandbox = prepare_riscv_formal_sandbox(
                run_id="run-a",
                cpu_name="picorv32",
                workspace_dir=str(workspace),
                source_dir=source,
            )
            runner = RiscvFormalRunner(riscv_formal_dir=str(sandbox))

            runner._copy_rtl(str(workspace), "picorv32.v")

            sandbox_rtl = sandbox / "cores" / "picorv32" / "picorv32.v"
            self.assertFalse(sandbox_rtl.is_symlink())
            self.assertEqual(sandbox_rtl.read_text(), "modified RTL\n")
            self.assertEqual(
                (source / "cores" / "picorv32" / "picorv32.v").read_text(),
                "module picorv32; endmodule\n",
            )

    def test_different_runs_use_different_sandboxes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = _create_formal_source(root)
            workspace_a = root / "runs" / "run-a" / "workspace"
            workspace_b = root / "runs" / "run-b" / "workspace"
            workspace_a.mkdir(parents=True)
            workspace_b.mkdir(parents=True)
            sandbox_a = prepare_riscv_formal_sandbox(
                run_id="run-a", cpu_name="picorv32",
                workspace_dir=str(workspace_a), source_dir=source,
            )
            sandbox_b = prepare_riscv_formal_sandbox(
                run_id="run-b", cpu_name="picorv32",
                workspace_dir=str(workspace_b), source_dir=source,
            )
            self.assertNotEqual(sandbox_a, sandbox_b)
            self.assertEqual(formal_sandbox_path("run-a", str(workspace_a)), sandbox_a)
            (sandbox_a / "insns" / "only-a.v").write_text("a")
            self.assertFalse((sandbox_b / "insns" / "only-a.v").exists())


class TestChecksIntegration(unittest.TestCase):
    """Test integration with src.checks module."""

    def test_is_valid_rtl(self) -> None:
        from src.checks import _is_valid_rtl

        with tempfile.TemporaryDirectory() as tmpdir:
            # Valid picorv32 RTL (must be >50KB to pass size check)
            valid = Path(tmpdir) / "picorv32.v"
            valid.write_text(
                "module picorv32 #(\nparameter BARREL_SHIFTER = 0\n)\n" + "x\n" * 30000
            )
            self.assertTrue(_is_valid_rtl(valid, "picorv32"))

            # Too small
            small = Path(tmpdir) / "small.v"
            small.write_text("module picorv32; endmodule")
            self.assertFalse(_is_valid_rtl(small, "picorv32"))

            # Valid for different CPU
            e203 = Path(tmpdir) / "e203.v"
            e203.write_text("module e203_cpu_top ();\n" + "x\n" * 6000)
            self.assertTrue(_is_valid_rtl(e203, "e203_hbirdv2"))

            # Unknown CPU with generic heuristic
            custom = Path(tmpdir) / "custom.v"
            custom.write_text("module mycpu ();\n" + "x\n" * 6000)
            self.assertTrue(_is_valid_rtl(custom, "mycpu"))

    @patch("src.main_graph.retry_gate")
    def test_formal_gate_uses_dedicated_retry_budget(
        self, mock_retry_gate: MagicMock
    ) -> None:
        from src.main_graph import formal_gate

        state = WorkflowState(needs_review=True)
        mock_retry_gate.return_value = state
        formal_gate(state)
        mock_retry_gate.assert_called_once_with(
            state,
            "formal",
            "formal_retry_count",
            max_retries=LACEConfig.MAX_FORMAL_RETRIES,
        )


if __name__ == "__main__":
    unittest.main()
