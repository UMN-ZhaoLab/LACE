rd_instr_example = """
instruction signal: mem_rdata_latched
clock signal: clk
reading instruction flag: mem_do_rinst
read memory done flag: mem_done
fetch_state: cpu_state_fetch

instr_trap:
	assign instr_trap = (CATCH_ILLINSN || WITH_PCPI) && !{instr_lui, instr_auipc, instr_jal, instr_jalr,
			instr_beq, instr_bne, instr_blt, instr_bge, instr_bltu, instr_bgeu,
			instr_lb, instr_lh, instr_lw, instr_lbu, instr_lhu, instr_sb, instr_sh, instr_sw,
			instr_addi, instr_slti, instr_sltiu, instr_xori, instr_ori, instr_andi, instr_slli, instr_srli, instr_srai,
			instr_add, instr_sub, instr_sll, instr_slt, instr_sltu, instr_xor, instr_srl, instr_sra, instr_or, instr_and,
			instr_rdcycle, instr_rdcycleh, instr_rdinstr, instr_rdinstrh, instr_fence,
			instr_getq, instr_setq, instr_retirq, instr_maskirq, instr_waitirq, instr_timer};
"""

rd_rs1 = """
the execution-stage source register 1 signal: reg_op1
"""

rd_rs2 = """
the read source register 2 signal: cpuregs_rs2
"""

rd_cust_reg = """
Create a custom register/regfile, and connect the value to the RdCustReg output port
"""

rd_pc = """
Search for the program counter value and connect to the RdPC output port
"""

rd_mem = """
1. Connect the `RdMem_addr_n_i` input port to the memory address port
2. Connect `RdMem_validReq` and `RdMem_addr_valid` input port to the enable signal of the read memory process
3. Connect the output of the read memory process to the `RdMem_n_o` output port
"""

wr_rd = """
	always @* begin
		cpuregs_write = 0;
		cpuregs_wrdata = 'bx;

		if (cpu_state == cpu_state_fetch) begin
			(* parallel_case *)
			case (1'b1)
				latched_branch: begin
					cpuregs_wrdata = reg_pc + (latched_compr ? 2 : 4);
					cpuregs_write = 1;
				end
				latched_store && !latched_branch: begin
					cpuregs_wrdata = latched_stalu ? alu_out_q : reg_out;
					cpuregs_write = 1;
				end
				ENABLE_IRQ && irq_state[0]: begin
					cpuregs_wrdata = reg_next_pc | latched_compr;
					cpuregs_write = 1;
				end
				ENABLE_IRQ && irq_state[1]: begin
					cpuregs_wrdata = irq_pending & ~irq_mask;
					cpuregs_write = 1;
				end
			endcase
		end
	end

"""

wr_cust_reg = """
1. Connect the `WrCustReg_validReq_n_i` input signal to the write general regfile enable signal
2. Connect the `WrCustReg_n_i` input signal to the write regfile data signal
"""

wr_pc = """
1. Connect the `WrPC_validReq_n_i` input signal to the write program counter enable signal
2. Connect the `WrPC_n_i` input signal to the next pc signal
"""

rd_flush = """
In state-machine based architecture, you just flush the state if this state do not work properly
"""

rd_stall = """
In state-machine based architecture, you just stall the state if this state do not work properly
"""


wr_mem = """
1. Use the picorv32 memory interface (`mem_valid`, `mem_addr`, `mem_wdata`, `mem_wstrb`, `mem_ready`)
2. When `WrMem_validReq_n_i` is asserted, drive `mem_valid` and map `WrMem_n_i` to address/data/byte enables
3. Hold the request until `mem_ready` is high
"""

wr_flush = """
1. Use the picorv32 control flow flush/reset mechanism to clear the pipeline/state
2. When `WrFlush_validReq_n_i` is asserted, trigger the flush and propagate `WrFlush_n_i` to the selected control signal
"""
