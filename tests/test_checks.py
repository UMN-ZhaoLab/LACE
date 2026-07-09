import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.checks import (
    check_arithmetic_syntax,
    check_interface_syntax,
    check_semantic_ports,
    function_check,
)
from src.interactive_engine import merge_interface_result
from src.state_types import WorkflowState


class TestChecks(unittest.TestCase):
    def _make_state(self, **kwargs) -> WorkflowState:
        defaults = {
            "spec": "",
            "ops": ["op1", "op2"],
            "hdl_tasks": ["task1", "task2"],
            "interface_code": "",
            "arithmetic_code": "",
            "interface_syntax_ok": False,
            "arithmetic_syntax_ok": False,
            "function_ok": False,
            "advance_op": False,
            "needs_review": False,
            "op_index": 0,
            "hdl_index": 0,
            "notes": [],
        }
        defaults.update(kwargs)
        return WorkflowState(**defaults)

    @patch("src.checks.verilator_syntax_check")
    def test_check_interface_syntax_empty_code(self, mock_verilator) -> None:
        state = self._make_state(interface_code="")
        result = check_interface_syntax(state)
        self.assertFalse(result.interface_syntax_ok)
        self.assertFalse(result.advance_op)
        mock_verilator.assert_not_called()

    @patch("src.checks.verilator_syntax_check")
    def test_check_interface_syntax_fail(self, mock_verilator) -> None:
        mock_verilator.return_value = (False, "syntax error")
        state = self._make_state(interface_code="bad code")
        result = check_interface_syntax(state)
        self.assertFalse(result.interface_syntax_ok)
        self.assertIn("Interface syntax check failed", result.notes)
        self.assertIn("Verilator output", result.last_error)
        self.assertIn("syntax error", result.last_error)
        mock_verilator.assert_called_once()

    @patch("src.checks.verilator_syntax_check")
    def test_check_interface_syntax_success_not_last_task(self, mock_verilator) -> None:
        mock_verilator.return_value = (True, "")
        state = self._make_state(
            interface_code="good code", hdl_tasks=["t1", "t2"], hdl_index=0
        )
        result = check_interface_syntax(state)
        self.assertEqual(result.hdl_index, 1)
        self.assertTrue(result.interface_syntax_ok)
        mock_verilator.assert_called_once()

    @patch("src.checks.verilator_syntax_check")
    def test_check_interface_syntax_success_last_task(self, mock_verilator) -> None:
        mock_verilator.return_value = (True, "")
        state = self._make_state(
            interface_code="good code", hdl_tasks=["t1"], hdl_index=0
        )
        result = check_interface_syntax(state)
        self.assertTrue(result.interface_syntax_ok)
        self.assertFalse(result.advance_op)

    @patch("src.checks.verilator_syntax_check")
    def test_check_interface_syntax_no_tasks(self, mock_verilator) -> None:
        mock_verilator.return_value = (True, "")
        state = self._make_state(interface_code="code", hdl_tasks=[])
        result = check_interface_syntax(state)
        self.assertTrue(result.interface_syntax_ok)
        mock_verilator.assert_called_once()

    @patch("src.checks.verilator_syntax_check")
    def test_check_arithmetic_syntax_empty(self, mock_verilator) -> None:
        state = self._make_state(arithmetic_code="")
        result = check_arithmetic_syntax(state)
        self.assertFalse(result.arithmetic_syntax_ok)
        mock_verilator.assert_not_called()

    @patch("src.checks.verilator_syntax_check")
    def test_check_arithmetic_syntax_fail(self, mock_verilator) -> None:
        mock_verilator.return_value = (False, "error")
        state = self._make_state(arithmetic_code="bad")
        result = check_arithmetic_syntax(state)
        self.assertFalse(result.arithmetic_syntax_ok)
        self.assertIn("Arithmetic syntax check failed", result.notes)

    @patch("src.checks.verilator_syntax_check")
    def test_check_arithmetic_syntax_success(self, mock_verilator) -> None:
        mock_verilator.return_value = (True, "")
        state = self._make_state(arithmetic_code="good")
        result = check_arithmetic_syntax(state)
        self.assertTrue(result.arithmetic_syntax_ok)

    def test_function_check_needs_review(self) -> None:
        state = self._make_state(needs_review=True, interface_syntax_ok=True)
        result = function_check(state)
        self.assertTrue(result.function_ok)
        self.assertFalse(result.advance_op)

    def test_function_check_syntax_not_ok(self) -> None:
        state = self._make_state(interface_syntax_ok=False)
        result = function_check(state)
        self.assertFalse(result.function_ok)
        self.assertFalse(result.advance_op)

    def test_function_check_blocks_on_arithmetic_failure(self) -> None:
        state = self._make_state(
            interface_syntax_ok=True,
            arithmetic_code="module alu; endmodule",
            arithmetic_syntax_ok=False,
            ops=["op1"],
            op_index=0,
        )
        result = function_check(state)
        self.assertFalse(result.function_ok)
        self.assertFalse(result.advance_op)

    def test_function_check_allows_when_no_arithmetic(self) -> None:
        # Without a workspace the formal runner cannot be prepared, so the
        # check is skipped (NOT passed). function_ok is False and
        # formal_skipped records that no real verification happened.
        state = self._make_state(
            interface_syntax_ok=True,
            arithmetic_code="",
            ops=["op1"],
            op_index=0,
        )
        result = function_check(state)
        self.assertFalse(result.function_ok)
        self.assertFalse(result.advance_op)
        self.assertTrue(result.formal_skipped)

    @patch("src.checks._prepare_riscv_formal_runner")
    @patch("src.checks.RiscvFormalRunner")
    def test_function_check_runs_baseline_formal(
        self, mock_runner_cls, mock_prepare
    ) -> None:
        """When workspace is valid, function_check runs riscv-formal baseline."""
        mock_prepare.return_value = (mock_runner_cls.return_value, [])
        mock_runner_cls.return_value.run_baseline_checks.return_value = {
            "passed": True,
            "results": [{"passed": True}],
            "total_time": 1.0,
            "error": "",
        }
        state = self._make_state(
            interface_syntax_ok=True,
            workspace_dir="/tmp/ws",
            cpu_name="picorv32",
            cpu_top_file="picorv32.v",
            ops=["op1"],
            op_index=0,
        )
        result = function_check(state)
        self.assertTrue(result.function_ok)
        self.assertFalse(result.advance_op)
        mock_runner_cls.return_value.run_baseline_checks.assert_called_once()

    @patch("src.checks._prepare_riscv_formal_runner")
    def test_function_check_skips_when_no_workspace(
        self, mock_prepare
    ) -> None:
        """When workspace is missing, function_check skips formal (not a pass)."""
        mock_prepare.return_value = None
        state = self._make_state(
            interface_syntax_ok=True,
            ops=["op1"],
            op_index=0,
        )
        result = function_check(state)
        self.assertFalse(result.function_ok)
        self.assertFalse(result.advance_op)
        self.assertTrue(result.formal_skipped)
        self.assertIn("skipped", " ".join(result.notes).lower())

    def test_check_semantic_ports_warns_on_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cpu_dir = Path(tmpdir) / "cpu"
            cpu_dir.mkdir()
            original = "module top ( input a, output b );\nendmodule\n"
            (cpu_dir / "top.v").write_text(original, encoding="utf-8")
            modified = "module top ( input a );\nendmodule\n"
            state = self._make_state(
                interface_code=modified,
                cpu_dir=str(cpu_dir),
                cpu_top_file="top.v",
            )
            result = check_semantic_ports(state)
            self.assertIn("missing", " ".join(result.notes).lower())
            self.assertIn("b", " ".join(result.notes))

    def test_check_semantic_ports_ok_when_ports_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cpu_dir = Path(tmpdir) / "cpu"
            cpu_dir.mkdir()
            original = "module top ( input a, output b );\nendmodule\n"
            (cpu_dir / "top.v").write_text(original, encoding="utf-8")
            modified = "module top ( input a, output b, output c );\nendmodule\n"
            state = self._make_state(
                interface_code=modified,
                cpu_dir=str(cpu_dir),
                cpu_top_file="top.v",
            )
            result = check_semantic_ports(state)
            self.assertNotIn("missing", " ".join(result.notes).lower())

    def test_check_semantic_ports_no_cpu_info(self) -> None:
        state = self._make_state(interface_code="module top; endmodule")
        result = check_semantic_ports(state)
        self.assertEqual(result.notes, [])

    def test_merge_interface_result_preserves_on_patch_failure(self) -> None:
        """When a SEARCH/REPLACE patch fails, interface_code preserves the
        current file content and interface_syntax_ok is set to False so the
        syntax-check node can trigger a normal retry instead of the empty-code
        guard."""
        with tempfile.TemporaryDirectory() as tmpdir:
            top_file = Path(tmpdir) / "picorv32.v"
            top_file.write_text("module picorv32; endmodule", encoding="utf-8")
            state = self._make_state(
                interface_code="module picorv32; endmodule",
                interface_syntax_ok=True,
                workspace_dir=tmpdir,
                cpu_top_file="picorv32.v",
            )
            bad_patch = "------- SEARCH\nnonexistent\n=======\nreplacement\n+++++++ REPLACE"
            result = merge_interface_result(state, bad_patch)
            self.assertEqual(result.interface_code, "module picorv32; endmodule")
            self.assertFalse(result.interface_syntax_ok)
            self.assertIn("patch error", result.last_error.lower())


if __name__ == "__main__":
    unittest.main()
