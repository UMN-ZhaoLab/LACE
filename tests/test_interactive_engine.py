"""Tests for the interactive engine prompt builders."""

from __future__ import annotations

import unittest

from src.interactive_engine import (
    LOCAL_STEP_NAMES,
    STEP_REGISTRY,
    advance_step,
    build_interface_prompt,
    get_current_step,
    merge_op2hdl_result,
)
from src.state_types import CandidateModule, HdlTasksOut, WorkflowState


class TestBuildInterfacePrompt(unittest.TestCase):
    def test_human_prefers_native_writeback(self) -> None:
        state = WorkflowState(
            spec="ROL rotate left, opcode=0001011, funct3=001, funct7=0000000",
            ops=["RdInstr()", "RdRS1()", "RdRS2()", "WrRD()"],
            hdl_tasks=["task1", "task2", "task3", "task4"],
            hdl_task_op_index_map=[0, 1, 2, 3],
            hdl_index=0,
            cpu_name="picorv32",
            cpu_dir="cpu_prototype/picorv32",
            cpu_top_file="picorv32.v",
            cpu_summary="picorv32 uses reg_op1/reg_op2 and reg_out/cpuregs_wrdata.",
        )
        prompt = build_interface_prompt(state)
        human = prompt["human"].lower()
        self.assertIn("source-proven normal result/writeback", human)
        self.assertIn("rvfi overlap", human)
        self.assertIn("every source-proven decode path", human)


class TestOp2HdlAccumulation(unittest.TestCase):
    def test_accumulates_tasks_for_all_ops_and_replaces_retry(self) -> None:
        state = WorkflowState(
            ops=["RdInstr()", "WrRD()"],
            op_index=0,
            run_id="test",
        )
        state = merge_op2hdl_result(
            state, HdlTasksOut(hdl_tasks=["plan instruction"], confidence="high")
        )
        state = state.model_copy(update={"op_index": 1})
        state = merge_op2hdl_result(
            state, HdlTasksOut(hdl_tasks=["plan writeback"], confidence="high")
        )
        self.assertEqual(state.hdl_tasks, ["plan instruction", "plan writeback"])
        self.assertEqual(state.hdl_task_op_index_map, [0, 1])

        state = merge_op2hdl_result(
            state, HdlTasksOut(hdl_tasks=["revised writeback"], confidence="high")
        )
        self.assertEqual(state.hdl_tasks, ["plan instruction", "revised writeback"])
        self.assertEqual(state.hdl_task_op_index_map, [0, 1])


class TestInteractiveWorkflow(unittest.TestCase):
    def test_plans_every_operation_before_candidate_selection(self) -> None:
        state = WorkflowState(
            cpu_dir="cpu",
            cpu_summary="summary",
            ops=["RdInstr()", "WrRD()"],
            run_id="interactive-test",
        )
        self.assertEqual(get_current_step(state), "op2hdl_tasks")

        state, log = advance_step(
            state,
            "op2hdl_tasks",
            {"hdl_tasks": ["decode task"], "confidence": "high"},
        )
        self.assertTrue(log["valid"])
        self.assertEqual(get_current_step(state), "advance_op")

        state, log = advance_step(state, "advance_op", "")
        self.assertTrue(log["local"])
        self.assertEqual(state.op_index, 1)
        self.assertEqual(get_current_step(state), "op2hdl_tasks")

        state, _ = advance_step(
            state,
            "op2hdl_tasks",
            {"hdl_tasks": ["writeback task"], "confidence": "high"},
        )
        self.assertEqual(get_current_step(state), "candidate_modules")
        self.assertEqual(state.hdl_task_op_index_map, [0, 1])

    def test_codegen_and_formal_steps_are_reachable(self) -> None:
        candidate = CandidateModule(module="picorv32.v", reason="top")
        state = WorkflowState(
            cpu_dir="cpu",
            cpu_summary="summary",
            ops=["RdInstr()"],
            op_index=0,
            hdl_tasks=["decode task"],
            hdl_task_op_index_map=[0],
            hdl_index=0,
            candidate_modules=[candidate],
            last_stage="candidate_modules",
        )
        self.assertEqual(get_current_step(state), "rag_retriever")
        state = state.model_copy(update={"last_stage": "rag_retriever"})
        self.assertEqual(get_current_step(state), "interface_writer")
        state = state.model_copy(update={"last_stage": "interface_writer"})
        self.assertEqual(get_current_step(state), "interface_syntax_check")

        state = state.model_copy(
            update={
                "hdl_index": 1,
                "interface_syntax_ok": True,
                "last_stage": "interface_syntax_check",
            }
        )
        self.assertEqual(get_current_step(state), "arithmetic_writer")
        state = state.model_copy(
            update={"arithmetic_code": "module lace_arithmetic; endmodule"}
        )
        self.assertEqual(get_current_step(state), "check_arithmetic_syntax")
        state = state.model_copy(update={"arithmetic_syntax_ok": True})
        self.assertEqual(get_current_step(state), "arithmetic_integrator")
        state = state.model_copy(
            update={
                "integrated_interface_code": "module picorv32; endmodule",
                "last_stage": "arithmetic_integrator",
            }
        )
        self.assertEqual(get_current_step(state), "semantic_port_check")
        state = state.model_copy(update={"last_stage": "semantic_port_check"})
        self.assertEqual(get_current_step(state), "original_function_checker")
        state = state.model_copy(update={"last_stage": "original_function_checker"})
        self.assertEqual(get_current_step(state), "insn_model_writer")
        state = state.model_copy(update={"last_stage": "insn_model_writer"})
        self.assertEqual(get_current_step(state), "final_function_checker")
        state = state.model_copy(update={"last_stage": "final_function_checker"})
        self.assertIsNone(get_current_step(state))

    def test_only_generation_steps_require_client_llm(self) -> None:
        expected_local = {
            "cpu_resolver",
            "advance_op",
            "rag_retriever",
            "interface_syntax_check",
            "check_arithmetic_syntax",
            "semantic_port_check",
            "original_function_checker",
            "final_function_checker",
        }
        self.assertEqual(LOCAL_STEP_NAMES, expected_local)
        for name in {
            "interface_writer",
            "arithmetic_writer",
            "arithmetic_integrator",
            "insn_model_writer",
        }:
            self.assertTrue(STEP_REGISTRY[name].needs_llm)


if __name__ == "__main__":
    unittest.main()
