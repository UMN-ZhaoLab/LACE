"""Tests for the arithmetic integrator helper functions."""

from __future__ import annotations

import unittest

from src.arithmetic_integrator import (
    _build_integration_prompt,
    _ensure_instance_wires,
    _fix_instance_ports,
    _remove_submodule_output_drivers,
)
from src.state_types import WorkflowState


class TestArithmeticIntegrator(unittest.TestCase):
    def test_ensure_instance_wires_adds_missing(self) -> None:
        cpu = """\
module picorv32 (
    input clk,
    input resetn
);
    // core body
    lace_arithmetic u_lace_arithmetic (
        .RdInstr_0_i(RdInstr_0_o),
        .WrRD_2_o(WrRD_2_i)
    );
endmodule
"""
        fixed = _ensure_instance_wires(cpu)
        self.assertIn("wire RdInstr_0_o;", fixed)
        self.assertIn("wire WrRD_2_i;", fixed)

    def test_fix_instance_ports_removes_invented_ports(self) -> None:
        arithmetic = """\
module lace_arithmetic (
    input  wire        clk_i,
    input  wire        rst_i,
    input  wire [31:0] RdInstr_0_i,
    input  wire [31:0] RdRS1_1_i,
    input  wire [31:0] RdRS2_1_i,
    output wire [31:0] WrRD_2_o,
    output wire        WrRD_validReq_2_o
);
endmodule
"""
        cpu = """\
module picorv32 (
    input clk,
    input resetn
);
    wire [31:0] RdInstr_0_o;
    wire [31:0] RdRS1_1_o;
    wire [31:0] RdRS2_1_o;
    wire [31:0] WrRD_2_i;
    wire        WrRD_validReq_2_i;

    lace_arithmetic u_lace_arithmetic (
        .clk_i(clk),
        .rst_i(resetn),
        .RdInstr_0_i(RdInstr_0_o),
        .RdRS1_1_i(RdRS1_1_o),
        .RdRS2_1_i(RdRS2_1_o),
        .WrRD_2_o(WrRD_2_i),
        .WrRD_validReq_2_o(WrRD_validReq_2_i),
        .WrPC_2_o(WrPC_2_i)
    );
endmodule
"""
        fixed = _fix_instance_ports(cpu, arithmetic)
        self.assertIn(".clk_i", fixed)
        self.assertIn("(clk)", fixed)
        self.assertIn(".RdInstr_0_i", fixed)
        self.assertIn("(RdInstr_0_o)", fixed)
        self.assertNotIn("WrPC_2", fixed)

    def test_fix_instance_ports_keeps_existing_custom_signal(self) -> None:
        arithmetic = """\
module lace_arithmetic (
    input  wire [31:0] RdInstr_0_i,
    output wire [31:0] WrRD_2_o,
    output wire        WrRD_validReq_2_o
);
endmodule
"""
        cpu = """\
module picorv32 ();
    wire [31:0] my_instr;
    wire [31:0] my_result;
    wire        my_valid;

    lace_arithmetic u_lace_arithmetic (
        .RdInstr_0_i(my_instr),
        .WrRD_2_o(my_result),
        .WrRD_validReq_2_o(my_valid)
    );
endmodule
"""
        fixed = _fix_instance_ports(cpu, arithmetic)
        self.assertIn(".RdInstr_0_i", fixed)
        self.assertIn("(my_instr)", fixed)
        self.assertIn(".WrRD_2_o", fixed)
        self.assertIn("(my_result)", fixed)
        self.assertIn(".WrRD_validReq_2_o", fixed)
        self.assertIn("(my_valid)", fixed)

    def test_fix_instance_ports_drops_operand_passthrough_outputs(self) -> None:
        arithmetic = """\
module lace_arithmetic (
    output wire [31:0] RdRS1_1_o,
    output wire [31:0] RdRS2_1_o,
    input  wire [31:0] RdRS1_1_i,
    input  wire [31:0] RdRS2_1_i,
    output wire [31:0] WrRD_2_o
);
endmodule
"""
        cpu = """\
module picorv32;
    lace_arithmetic u_lace_arithmetic (
        .RdRS1_1_o(RdRS1_1_o),
        .RdRS2_1_o(RdRS2_1_o),
        .RdRS1_1_i(RdRS1_1_o),
        .RdRS2_1_i(RdRS2_1_o),
        .WrRD_2_o(WrRD_2_i)
    );
endmodule
"""
        fixed = _fix_instance_ports(cpu, arithmetic)
        self.assertNotIn(".RdRS1_1_o", fixed)
        self.assertNotIn(".RdRS2_1_o", fixed)
        self.assertIn(".RdRS1_1_i", fixed)
        self.assertIn(".RdRS2_1_i", fixed)
        self.assertIn(".WrRD_2_o", fixed)

    def test_submodule_outputs_are_single_driver_wires(self) -> None:
        arithmetic = """\
module lace_arithmetic (
    output reg [31:0] WrRD_2_o,
    output reg WrRD_validReq_2_o,
    input [31:0] RdInstr_0_i
);
endmodule
"""
        cpu = """\
module picorv32 (input clk);
    reg [31:0] WrRD_2_i;
    reg WrRD_validReq_2_i;
    always @(posedge clk) begin
        WrRD_2_i <= 0;
        WrRD_validReq_2_i <= 0;
    end
    wire use_result = WrRD_validReq_2_i;
    lace_arithmetic u_lace_arithmetic (
        .WrRD_2_o(WrRD_2_i),
        .WrRD_validReq_2_o(WrRD_validReq_2_i),
        .RdInstr_0_i(32'b0)
    );
endmodule
"""
        fixed = _remove_submodule_output_drivers(cpu, arithmetic)
        self.assertIn("wire [31:0] WrRD_2_i;", fixed)
        self.assertIn("wire WrRD_validReq_2_i;", fixed)
        self.assertNotIn("WrRD_2_i <=", fixed)
        self.assertNotIn("WrRD_validReq_2_i <=", fixed)
        self.assertIn("wire use_result = WrRD_validReq_2_i;", fixed)

    def test_integration_prompt_reinforces_native_path(self) -> None:
        state = WorkflowState(
            spec="ROL rotate left",
            ops=["RdInstr()", "RdRS1()", "RdRS2()", "WrRD()"],
            interface_code="module picorv32(); endmodule",
            arithmetic_code="module lace_arithmetic(); endmodule",
        )
        prompt = _build_integration_prompt(state)
        human = prompt["human"].lower()
        self.assertIn("selected from rtl evidence", human)
        self.assertIn("existing register-file write logic", human)

    def test_integration_rejects_lint_failure(self) -> None:
        """Regression: an integrated file that fails Verilator lint must be
        caught here, not silently passed downstream with interface_syntax_ok=True.

        The interface syntax check runs BEFORE integration; the integrator can
        introduce duplicate declarations / bad splices. Without this gate a
        syntactically broken CPU reached riscv-formal and interface_syntax_ok
        was a false positive (observed in a real LLM run).
        """
        import shutil
        import tempfile
        from pathlib import Path
        from unittest.mock import patch, MagicMock
        from langchain_core.messages import AIMessage
        from src.arithmetic_integrator import arithmetic_integrator

        verilator = shutil.which("verilator")
        if not verilator:
            self.skipTest("verilator not available")

        # interface_code is syntactically valid; the LLM "integration" will
        # duplicate a declaration, breaking lint — exactly the real-world failure.
        interface_code = (
            "module picorv32 (\n  input clk,\n  input resetn\n);\n"
            "  reg alu_out_0, alu_out_0_q;\n"
            "  wire [31:0] RdInstr_0_o;\n"
            "  wire [31:0] WrRD_2_i;\n"
            "  lace_arithmetic u_lace_arithmetic (\n"
            "    .clk_i(clk), .rst_i(resetn),\n"
            "    .RdInstr_0_i(RdInstr_0_o), .WrRD_2_o(WrRD_2_i)\n"
            "  );\n"
            "endmodule\n"
        )
        # Integrator's LLM returns interface_code WITH a duplicate declaration.
        broken_integrated = interface_code.replace(
            "  reg alu_out_0, alu_out_0_q;\n",
            "  reg alu_out_0, alu_out_0_q;\n  reg alu_out_0, alu_out_0_q;\n",
            1,
        )
        arithmetic_code = (
            "module lace_arithmetic (\n"
            "  input clk_i, input rst_i,\n"
            "  input [31:0] RdInstr_0_i, output [31:0] WrRD_2_o,\n"
            "  output WrRD_validReq_2_o\n);\nendmodule\n"
        )

        tmp = tempfile.mkdtemp(prefix="lace_integ_lint_")
        state = WorkflowState(
            interface_code=interface_code,
            arithmetic_code=arithmetic_code,
            workspace_dir=tmp,
            cpu_top_file="picorv32.v",
            cpu_name="picorv32",
            verilator_std="+1364-2005ext+.v",
            verilator_waive_flags=["--Wno-MULTITOP"],
            op_index=0,
            hdl_index=0,
        )
        model = MagicMock()
        model.invoke.return_value = AIMessage(content=broken_integrated)
        model.bind_tools = lambda t: model
        model.bind = lambda **k: model
        with patch("src.arithmetic_integrator.get_chat_model", return_value=model):
            result = arithmetic_integrator(state)

        try:
            self.assertTrue(result.needs_review)
            self.assertFalse(result.interface_syntax_ok)
            self.assertIn("failed Verilator lint", result.last_error)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
