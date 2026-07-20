"""Tests for the op->HDL task planner prompts."""

from __future__ import annotations

import unittest

from src.prompts.op2hdl import get_prompt_for_op


class TestOp2HdlNativePath(unittest.TestCase):
    def test_rol_tasks_mention_native_path(self) -> None:
        ops = ["RdInstr()", "RdRS1()", "RdRS2()", "WrRD()"]
        cpu_summary = (
            "picorv32 is a multi-cycle RISC-V CPU. "
            "Decode uses mem_rdata_latched/mem_rdata_q. "
            "Register reads produce reg_op1/reg_op2. "
            "ALU result is reg_out; writeback uses cpuregs_wrdata."
        )
        spec = "ROL: rotate left. Encoding: opcode=0001011, funct3=001, funct7=0000000."

        rd_instr_prompt = get_prompt_for_op(ops, 0, cpu_summary=cpu_summary, spec=spec)
        self.assertIn("search", rd_instr_prompt.lower())
        self.assertIn("generate all of them", rd_instr_prompt.lower())
        self.assertIn("collide", rd_instr_prompt.lower())
        self.assertIn("every source-proven decode site", rd_instr_prompt.lower())
        self.assertIn("RVFI overlap", rd_instr_prompt)

        rd_rs1_prompt = get_prompt_for_op(ops, 1, cpu_summary=cpu_summary, spec=spec)
        self.assertIn("source-proven rs1", rd_rs1_prompt.lower())
        self.assertIn("Reuse", rd_rs1_prompt)

        wr_prompt = get_prompt_for_op(ops, 3, cpu_summary=cpu_summary, spec=spec)
        self.assertIn("source-proven normal rd writeback", wr_prompt.lower())
        self.assertIn("existing register-file write logic", wr_prompt.lower())


if __name__ == "__main__":
    unittest.main()
