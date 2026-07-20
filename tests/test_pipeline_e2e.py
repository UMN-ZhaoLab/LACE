"""End-to-end integration tests for the LACE pipeline (Graph mode only).

The legacy imperative path (src.workflow) has been removed; the compiled
LangGraph (run_graph_segment) is the single execution path. These tests mock
the LLM and Verilator/formal toolchain so the suite runs without API keys or
external tools, while still exercising the real graph topology, retry gates,
and state merging.

A separate tests/test_e2e_real_picorv32.py exercises a real Verilator lint
against the real picorv32 source for correctness regression.
"""

import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.config import LACEConfig
from src.pipeline_runner import run_graph_segment
from src.state_types import WorkflowState


def _ensure_state(s):
    return s if isinstance(s, WorkflowState) else WorkflowState(**s)


class TestPipelineE2E(unittest.TestCase):
    def setUp(self) -> None:
        # Isolate artifact/checkpoint storage per test so the shared
        # SqliteSaver and run_db do not leak state across tests.
        self._tmp = tempfile.mkdtemp(prefix="lace_e2e_")
        self._orig_artifact_dir = LACEConfig.ARTIFACT_DIR
        LACEConfig.ARTIFACT_DIR = self._tmp
        # run_db computes DB_PATH at import time from ARTIFACT_DIR; rebind it
        # so each test gets a fresh database under the temp dir.
        import src.run_db as run_db
        from pathlib import Path
        self._orig_db_path = run_db.DB_PATH
        run_db.DB_PATH = Path(self._tmp) / "runs.db"
        # file_utils caches safe write zones at first use; reset the cache so
        # the temp artifact dir is recognised as a legal write target.
        import src.file_utils as fu
        self._orig_safe_zones = fu._SAFE_ZONES
        fu._SAFE_ZONES = None
        fu.register_safe_zone(self._tmp)

    def tearDown(self) -> None:
        LACEConfig.ARTIFACT_DIR = self._orig_artifact_dir
        import src.run_db as run_db
        run_db.DB_PATH = self._orig_db_path
        import src.file_utils as fu
        fu._SAFE_ZONES = self._orig_safe_zones
        shutil.rmtree(self._tmp, ignore_errors=True)

    # ------------------------------------------------------------------
    # Graph mode (the only path)
    # ------------------------------------------------------------------

    @patch("src.checks.verilator_syntax_check")
    @patch("src.checks._prepare_riscv_formal_runner")
    def test_happy_path_graph_mode(
        self,
        mock_prepare: MagicMock,
        mock_verilator: MagicMock,
    ) -> None:
        """Run the full pipeline via Graph mode with mock LLM.

        NOTE: this is a control-flow test. Verilator and the riscv-formal
        runner are mocked, so function_ok here only asserts the graph reached
        a non-review terminal state — it does NOT validate generated HDL
        correctness (see test_e2e_real_picorv32.py for that).
        """
        mock_verilator.return_value = (True, "")
        # Pretend formal verification ran and passed.
        mock_prepare.return_value = None  # skip path; final checker escalates below

        state, _log, _rid = run_graph_segment(
            spec="Add a custom rotate instruction",
            cpu_name="picorv32",
            mock=True,
        )

        self.assertIsInstance(state, WorkflowState)
        self.assertTrue(state.interface_syntax_ok)
        self.assertTrue(state.arithmetic_syntax_ok)
        self.assertEqual(
            state.hdl_task_op_index_map,
            list(range(len(state.ops))),
            "every micro-operation must receive an HDL plan",
        )
        self.assertTrue(state.candidate_modules)
        self.assertEqual(
            sum(1 for entry in _log if entry["step_name"] == "arithmetic_integrator"),
            1,
        )
        self.assertEqual(
            sum(1 for entry in _log if entry["step_name"] == "original_function_checker"),
            1,
        )
        mock_verilator.assert_called()

    @patch("src.checks.verilator_syntax_check")
    @patch("src.checks._prepare_riscv_formal_runner")
    @patch("src.formal.riscv_formal_runner.RiscvFormalRunner")
    def test_happy_path_formal_passes(
        self,
        mock_runner_cls: MagicMock,
        mock_prepare: MagicMock,
        mock_verilator: MagicMock,
    ) -> None:
        """When the formal runner is available and passes, function_ok is True."""
        mock_verilator.return_value = (True, "")
        mock_prepare.return_value = (mock_runner_cls.return_value, [])
        mock_runner_cls.return_value.run_baseline_checks.return_value = {
            "passed": True,
            "results": [{"passed": True}],
            "total_time": 1.0,
            "error": "",
        }
        mock_runner_cls.return_value.run_custom_instruction_checks.return_value = {
            "passed": True,
            "results": [{"name": "insn_mock_ch0", "passed": True}],
            "total_time": 1.0,
            "error": "",
        }

        state, _log, _rid = run_graph_segment(
            spec="Add a custom rotate instruction",
            cpu_name="picorv32",
            mock=True,
        )

        self.assertIsInstance(state, WorkflowState)
        self.assertTrue(state.function_ok)
        self.assertFalse(state.formal_skipped)
        self.assertFalse(state.needs_review)

    @patch("src.checks.verilator_syntax_check")
    @patch("src.checks._prepare_riscv_formal_runner")
    def test_formal_skip_escalates_to_review(
        self,
        mock_prepare: MagicMock,
        mock_verilator: MagicMock,
    ) -> None:
        """A formal skip must NOT report success; it escalates to needs_review."""
        mock_verilator.return_value = (True, "")
        mock_prepare.return_value = None  # no workspace / untrusted RTL

        state, _log, _rid = run_graph_segment(
            spec="Add a custom rotate instruction",
            cpu_name="picorv32",
            mock=True,
        )

        self.assertTrue(state.formal_skipped)
        # Final checker escalates a skip into review rather than passing.
        self.assertTrue(state.needs_review)
        self.assertFalse(state.function_ok)

    @patch("src.checks._prepare_riscv_formal_runner")
    def test_arithmetic_failure_retries_then_halts_before_formal(
        self,
        mock_prepare: MagicMock,
    ) -> None:
        """Arithmetic syntax failures consume their budget, then halt before formal.

        Regression guard: broken arithmetic must never reach integration or
        be masked by a later gate. The writer gets bounded correction attempts,
        then the serial syntax gate escalates the failure.
        """
        from pathlib import Path
        from langchain_core.messages import AIMessage
        from unittest.mock import MagicMock as _MM
        from src.state_types import (
            CandidateModulesOut, CpuStructureOut, HdlTasksOut, OpsOut,
        )

        mock_prepare.return_value = None

        # Mock LLM: interface writer returns a no-op (keep pristine RTL);
        # arithmetic writer returns broken code containing 'lace_arithmetic'
        # so the real Verilator mock below can target it.
        model = _MM()
        def _wso(schema):
            r = _MM()
            def _inv(m):
                if schema is OpsOut:
                    return OpsOut(ops=["RdInstr()", "WrRD()"], arithmetic_ops="x", confidence="high")
                if schema is CpuStructureOut:
                    return CpuStructureOut(summary="s", module_index=["picorv32.v"])
                if schema is CandidateModulesOut:
                    return CandidateModulesOut(candidates=[{"module": "picorv32.v", "reason": "r"}], confidence="high")
                if schema is HdlTasksOut:
                    return HdlTasksOut(hdl_tasks=["add wires"], confidence="high")
                return schema()
            r.invoke.side_effect = _inv
            return r
        model.with_structured_output.side_effect = _wso
        def _direct_invoke(messages):
            text = " ".join(getattr(m, "content", "") if not isinstance(m, str) else m for m in messages)
            if "RTL integration analyst" in text:
                import json
                item = {
                    "file": "picorv32.v", "lines": [1, 1],
                    "signals": ["module"], "excerpt": "module picorv32",
                }
                return AIMessage(content=json.dumps({key: item for key in ("decode", "writeback", "timing")}))
            if "Required Extension Interface Wires" in text:
                return AIMessage(content="No changes are needed.")
            # Broken arithmetic: missing closing paren — real Verilator rejects it.
            return AIMessage(content="module lace_arithmetic (input clk\n")
        model.invoke.side_effect = _direct_invoke
        model.bind_tools = lambda t: model
        model.bind = lambda **k: model

        with (
            patch("src.agents.get_chat_model", return_value=model),
            patch("src.writers.get_chat_model", return_value=model),
            patch("src.arithmetic_integrator.get_chat_model", return_value=model),
        ):
            state, log, _rid = run_graph_segment(
                spec="Add a custom rotate instruction",
                cpu_name="picorv32",
                mock=False,
            )

        # Must halt in review, NOT silently pass.
        self.assertTrue(state.needs_review)
        self.assertFalse(state.arithmetic_syntax_ok)
        self.assertFalse(state.function_ok)
        self.assertFalse(state.formal_skipped)
        syntax_runs = sum(1 for e in log if e["step_name"] == "check_arithmetic_syntax")
        expected_runs = LACEConfig.MAX_VERILATOR_RETRIES + 1
        self.assertEqual(
            syntax_runs,
            expected_runs,
            f"arithmetic syntax gate ran {syntax_runs} times (expected {expected_runs})",
        )
        self.assertFalse(
            any(e["step_name"] == "original_function_checker" for e in log),
            "formal baseline must not run after arithmetic lint exhaustion",
        )

    @patch("src.checks.verilator_syntax_check")
    @patch("src.checks._prepare_riscv_formal_runner")
    def test_graph_resume_from_checkpoint(
        self,
        mock_prepare: MagicMock,
        mock_verilator: MagicMock,
    ) -> None:
        """A second run with the same run_id resumes from the saved checkpoint."""
        mock_verilator.return_value = (True, "")
        mock_prepare.return_value = None

        spec = "Add a custom rotate instruction"
        run_id = "resume-test-001"

        # First run completes the graph (terminal state).
        state1, _log1, _rid1 = run_graph_segment(
            spec=spec, cpu_name="picorv32", mock=True, run_id=run_id
        )
        self.assertIsInstance(state1, WorkflowState)

        # Resume: same run_id with start_from should load the checkpoint and
        # not crash. We only assert the runner returns a valid state and the
        # graph did not error out (resuming a completed run is a no-op-ish
        # continuation, not a fresh execution).
        state2, _log2, _rid2 = run_graph_segment(
            spec=spec,
            cpu_name="picorv32",
            mock=True,
            run_id=run_id,
            start_from="op2hdl_planner",
        )
        self.assertIsInstance(state2, WorkflowState)
        self.assertEqual(_rid2, run_id)


if __name__ == "__main__":
    unittest.main()
