import unittest

from src.arithmetic_skeleton import (
    _get_isax_op_name,
    _parse_op_call,
    generate_arithmetic_skeleton,
)


class TestParseOpCall(unittest.TestCase):
    def test_simple(self) -> None:
        self.assertEqual(_parse_op_call("MUL(rs1, rs2)"), ("MUL", ["rs1", "rs2"]))

    def test_no_args(self) -> None:
        self.assertEqual(_parse_op_call("RdInstr()"), ("RdInstr", []))

    def test_spaces(self) -> None:
        self.assertEqual(_parse_op_call("ADD( rs1 , rs2 )"), ("ADD", ["rs1", "rs2"]))

    def test_plain_name(self) -> None:
        self.assertEqual(_parse_op_call("CustomLogic"), ("CustomLogic", []))

    def test_assignment_syntax(self) -> None:
        """Ops may use assignment syntax like 'var = Op(args)'."""
        self.assertEqual(_parse_op_call("insn = RdInstr()"), ("RdInstr", []))
        self.assertEqual(_parse_op_call("result = MUL(rs1, rs2)"), ("MUL", ["rs1", "rs2"]))


class TestGetIsaxOpName(unittest.TestCase):
    def test_from_arithmetic_op(self) -> None:
        ops = ["RdInstr()", "MUL(rs1, rs2)", "WrRD()"]
        self.assertEqual(_get_isax_op_name(ops), "mul")

    def test_fallback_lace(self) -> None:
        ops = ["RdInstr()", "RdRS1()", "WrRD()"]
        self.assertEqual(_get_isax_op_name(ops), "lace")


class TestGenerateSkeleton(unittest.TestCase):
    def test_basic_structure(self) -> None:
        ops = ["RdInstr()", "RdRS1()", "RdRS2()", "MUL(rs1, rs2)", "WrRD()"]
        code = generate_arithmetic_skeleton(ops)

        self.assertIn("module lace_arithmetic (", code)
        self.assertIn("endmodule", code)

    def test_ports_derived_from_ops(self) -> None:
        ops = ["RdInstr()", "RdRS1()", "RdRS2()", "MUL(rs1, rs2)", "WrRD()"]
        code = generate_arithmetic_skeleton(ops)

        self.assertIn("RdInstr_0_i", code)
        self.assertIn("RdRS1_1_i", code)
        self.assertNotIn("RdRS1_1_o", code)
        self.assertIn("RdRS2_1_i", code)
        self.assertNotIn("RdRS2_1_o", code)
        self.assertIn("WrRD_2_o", code)
        self.assertIn("WrRD_validReq_2_o", code)

    def test_arithmetic_isax_ports(self) -> None:
        ops = ["RdInstr()", "RdRS1()", "MUL(rs1, rs2)", "WrRD()"]
        code = generate_arithmetic_skeleton(ops)

        self.assertIn("WrRD_mul_2_i", code)
        self.assertIn("WrRD_validReq_mul_2_i", code)
        self.assertIn("RdIValid_mul_1_o", code)

    def test_todo_comments_present(self) -> None:
        ops = ["RdInstr()", "RdRS1()", "MUL(rs1, rs2)", "WrRD()"]
        code = generate_arithmetic_skeleton(ops)

        self.assertIn("TODO LLM:", code)
        self.assertIn("DECODE:", code)

    def test_pipeline_registers(self) -> None:
        ops = ["RdInstr()", "RdRS1()", "MUL(rs1, rs2)", "WrRD()"]
        code = generate_arithmetic_skeleton(ops)

        self.assertIn("always@(posedge clk_i)", code)
        self.assertIn("RdIValid_mul_1_reg", code)
        self.assertIn("RdIValid_mul_2_reg", code)

    def test_no_unused_operand_pass_through_outputs(self) -> None:
        ops = ["RdInstr()", "RdRS1()", "RdRS2()", "MUL(rs1, rs2)", "WrRD()"]
        code = generate_arithmetic_skeleton(ops)

        self.assertNotIn("assign RdRS1_1_o = RdRS1_1_i;", code)
        self.assertNotIn("assign RdRS2_1_o = RdRS2_1_i;", code)

    def test_no_arithmetic_ops(self) -> None:
        ops = ["RdInstr()", "RdRS1()", "WrRD()"]
        code = generate_arithmetic_skeleton(ops)

        # Should still generate a module, but without ISAX-specific signals
        self.assertIn("module lace_arithmetic (", code)
        self.assertNotIn("RdIValid_", code)

    def test_custom_logic_op_name(self) -> None:
        ops = ["RdInstr()", "RdRS1()", "CustomLogic(a, b)", "WrRD()"]
        code = generate_arithmetic_skeleton(ops)

        self.assertIn("WrRD_customlogic_2_i", code)

    def test_multiple_arithmetic_ops(self) -> None:
        ops = ["RdInstr()", "RdRS1()", "MUL(rs1, rs2)", "ADD(result, imm)", "WrRD()"]
        code = generate_arithmetic_skeleton(ops)

        # First arithmetic op drives the ISAX port name
        self.assertIn("WrRD_mul_2_i", code)
        # Both ops should appear in TODO comments
        self.assertIn("MUL(rs1, rs2)", code)
        self.assertIn("ADD(result, imm)", code)

    def test_flush_stall_ports(self) -> None:
        ops = ["RdInstr()", "RdRS1()", "RdRS2()", "MUL(rs1, rs2)", "WrRD()"]
        code = generate_arithmetic_skeleton(ops)

        # 3 stages (0,1,2) -> 3 flush, 2 stall
        self.assertIn("RdFlush_0_i", code)
        self.assertIn("RdFlush_1_i", code)
        self.assertIn("RdFlush_2_i", code)
        self.assertIn("RdStall_0_i", code)
        self.assertIn("RdStall_1_i", code)

    def test_wrpc_ports(self) -> None:
        ops = ["RdInstr()", "RdPC()", "WrPC()"]
        code = generate_arithmetic_skeleton(ops)

        self.assertIn("RdPC_0_i", code)
        self.assertIn("WrPC_3_o", code)
        self.assertIn("WrPC_validReq_3_o", code)


if __name__ == "__main__":
    unittest.main()
