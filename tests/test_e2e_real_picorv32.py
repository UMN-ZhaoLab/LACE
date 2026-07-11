"""Real end-to-end test: mock LLM + REAL Verilator lint against real picorv32.

Unlike tests/test_pipeline_e2e.py (which mocks Verilator too), this test runs
the actual `verilator --lint-only` against the real picorv32.v source copied
into a per-test workspace. It catches regressions that pure-mock tests cannot:
SEARCH/REPLACE application corrupting the file, bad module splicing, and
genuine syntax breakage in the integration path.

The LLM is still mocked (for determinism and to avoid API cost), but the mock
is tuned so the interface writer returns a no-op (the workspace file stays the
pristine picorv32 source), letting Verilator validate real RTL.

Skipped automatically when Verilator or the picorv32 submodule is unavailable.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.config import LACEConfig
from src.pipeline_runner import run_graph_segment
from src.state_types import (
    CandidateModulesOut,
    CpuStructureOut,
    HdlTasksOut,
    OpsOut,
    WorkflowState,
)

PICORV32_TOP = Path("cpu_prototype/picorv32/picorv32.v")
VERILATOR = shutil.which("verilator")


def _make_noop_interface_mock() -> MagicMock:
    """Mock LLM whose interface writer returns a no-op (keep original file).

    The arithmetic writer returns a minimal but syntactically valid module so
    Verilator lint passes. Structured-output calls (spec2op/op2hdl/candidate)
    return canned payloads identical to the pipeline_runner mock.
    """
    model = MagicMock()

    def _with_structured_output(schema):
        runnable = MagicMock()

        def _invoke(messages):
            if schema is OpsOut:
                return OpsOut(
                    ops=["RdInstr()", "RdRS1()", "RdRS2()", "WrRD()"],
                    arithmetic_ops="SLICE(imm, 4, 0)",
                    confidence="high",
                )
            if schema is CpuStructureOut:
                return CpuStructureOut(
                    summary="multi-cycle RV32IMC core (picorv32)",
                    module_index=["picorv32.v"],
                )
            if schema is CandidateModulesOut:
                return CandidateModulesOut(
                    candidates=[
                        {"module": "picorv32.v", "reason": "top module", "related_ops": ["ROL"]}
                    ],
                    confidence="high",
                )
            if schema is HdlTasksOut:
                return HdlTasksOut(hdl_tasks=["add rotate interface wires"], confidence="high")
            return schema()

        runnable.invoke.side_effect = _invoke
        return runnable

    model.with_structured_output.side_effect = _with_structured_output

    def _direct_invoke(messages):
        # Heuristic: only the interface_writer prompt contains the literal
        # "Required Extension Interface Wires" (see writers.py). For that one
        # we return a no-op so the workspace keeps pristine RTL for real
        # Verilator lint. Every other writer gets a valid minimal module.
        text = " ".join(
            getattr(m, "content", "") if not isinstance(m, str) else m
            for m in messages
        )
        if "Required Extension Interface Wires" in text:
            return AIMessage(content="No changes are needed; the interface wires are already present.")
        return AIMessage(content="module lace_arithmetic (input clk); endmodule\n")

    model.invoke.side_effect = _direct_invoke
    model.bind_tools = lambda tools: model
    model.bind = lambda **kw: model
    return model


@unittest.skipUnless(
    VERILATOR and PICORV32_TOP.exists(),
    "Verilator or cpu_prototype/picorv32/picorv32.v not available",
)
class TestE2ERealPicorv32(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="lace_real_e2e_")
        self._orig_artifact_dir = LACEConfig.ARTIFACT_DIR
        LACEConfig.ARTIFACT_DIR = self._tmp
        import src.run_db as run_db
        self._orig_db_path = run_db.DB_PATH
        run_db.DB_PATH = Path(self._tmp) / "runs.db"
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

    @patch("src.checks._prepare_riscv_formal_runner")
    def test_real_verilator_lint_on_picorv32(self, mock_prepare: MagicMock) -> None:
        """The real picorv32.v must pass Verilator lint after the (no-op) interface step.

        This is a correctness gate the all-mock suite cannot provide: if the
        SEARCH/REPLACE / module-splicing logic ever corrupts the file, real
        Verilator will reject it here.
        """
        mock_prepare.return_value = None  # formal skipped (no sby in this env)
        mock_llm = _make_noop_interface_mock()

        with (
            patch("src.agents.get_chat_model", return_value=mock_llm),
            patch("src.writers.get_chat_model", return_value=mock_llm),
            patch("src.arithmetic_integrator.get_chat_model", return_value=mock_llm),
        ):
            state, _log, _rid = run_graph_segment(
                spec="Add a ROL (rotate left) instruction to picorv32.",
                cpu_name="picorv32",
                mock=False,  # we inject our own mock above
                run_id="real-e2e-001",
            )

        # The interface step must have passed REAL Verilator lint.
        self.assertTrue(
            state.interface_syntax_ok,
            f"Real Verilator lint failed on picorv32 workspace: {state.last_error}",
        )
        # Arithmetic module lint also runs against real Verilator.
        self.assertTrue(state.arithmetic_syntax_ok)
        # Formal is skipped (no sby), so the run must end in review, not success.
        self.assertTrue(state.formal_skipped)
        self.assertTrue(state.needs_review)


if __name__ == "__main__":
    unittest.main()
