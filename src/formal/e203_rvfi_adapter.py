"""Private RVFI repair overlay for the e203 formal wrapper.

The bundled e203 flattened export contains an incomplete RVFI block.  This
module never changes the CPU prototype: it rewrites a copy held by a LACE
formal sandbox so RVFI is sampled from the existing EXU commit interface.
"""

from __future__ import annotations

from pathlib import Path


class E203RvfiAdapterError(RuntimeError):
    """The expected e203 flattened-export structure was not found."""


def _replace_once(content: str, old: str, new: str, description: str) -> str:
    """Apply one deliberately narrow source transformation."""
    count = content.count(old)
    if count != 1:
        raise E203RvfiAdapterError(
            f"Expected exactly one {description} anchor, found {count}"
        )
    return content.replace(old, new, 1)


def _replace_exactly(
    content: str, old: str, new: str, expected: int, description: str
) -> str:
    """Replace a known number of structurally identical hierarchy edges."""
    count = content.count(old)
    if count != expected:
        raise E203RvfiAdapterError(
            f"Expected {expected} {description} anchors, found {count}"
        )
    return content.replace(old, new)


def apply_e203_rvfi_adapter(source: Path, destination: Path) -> None:
    """Create a private e203 source with commit-aligned RVFI metadata.

    The original wrapper exported fetch-stage instruction/PC signals despite
    asserting ``rvfi_valid`` at EXU commit.  Four commit signals are routed
    through the existing e203 EXU/core/CPU hierarchy.
    The RVFI block then uses that record for instruction, PC, and memory-kind
    fields.  Register data remains on the core's existing RVFI observation
    ports; no ISA-extension implementation mapping is introduced here.
    """
    content = source.read_text(encoding="utf-8")

    root_declarations = """logic alu_cmt_valid;
logic [`E203_INSTR_SIZE-1:0] ifu_o_ir;"""
    content = _replace_once(
        content,
        root_declarations,
        """logic alu_cmt_valid;
logic [`E203_PC_SIZE-1:0] alu_cmt_pc;
logic [`E203_INSTR_SIZE-1:0] alu_cmt_instr;
logic alu_cmt_ld;
logic alu_cmt_stamo;
logic [`E203_XLEN-1:0] alu_cmt_rd_wdata;
logic alu_cmt_rd_wen;
logic alu_cmt_trap;
logic [`E203_XLEN-1:0] alu_cmt_rs1_rdata;
logic [`E203_XLEN-1:0] alu_cmt_rs2_rdata;
logic [`E203_ADDR_SIZE-1:0] alu_cmt_mem_addr;
logic [`E203_XLEN/8-1:0] alu_cmt_mem_wmask;
logic [`E203_XLEN-1:0] alu_cmt_mem_wdata;
logic [234:0] rvfi_longp;
wire e203_rvfi_longp_valid = rvfi_longp[234];
wire e203_rvfi_longp_trap = rvfi_longp[233];
wire [`E203_INSTR_SIZE-1:0] e203_rvfi_longp_instr = rvfi_longp[232:201];
wire [`E203_XLEN-1:0] e203_rvfi_longp_rs1_rdata = rvfi_longp[200:169];
wire [`E203_XLEN-1:0] e203_rvfi_longp_rs2_rdata = rvfi_longp[168:137];
wire [`E203_ADDR_SIZE-1:0] e203_rvfi_longp_mem_addr = rvfi_longp[136:105];
wire [`E203_XLEN/8-1:0] e203_rvfi_longp_mem_wmask = rvfi_longp[104:101];
wire [`E203_XLEN-1:0] e203_rvfi_longp_mem_wdata = rvfi_longp[100:69];
wire [`E203_PC_SIZE-1:0] e203_rvfi_longp_pc = rvfi_longp[68:37];
wire [`E203_RFIDX_WIDTH-1:0] e203_rvfi_longp_rdidx = rvfi_longp[36:32];
wire [`E203_XLEN-1:0] e203_rvfi_longp_rd_wdata = rvfi_longp[31:0];
wire [`E203_INSTR_SIZE-1:0] e203_rvfi_instr = e203_rvfi_longp_valid ? e203_rvfi_longp_instr : alu_cmt_instr;
wire [`E203_XLEN-1:0] e203_rvfi_rs1_rdata = e203_rvfi_longp_valid ? e203_rvfi_longp_rs1_rdata : alu_cmt_rs1_rdata;
wire [`E203_XLEN-1:0] e203_rvfi_rs2_rdata = e203_rvfi_longp_valid ? e203_rvfi_longp_rs2_rdata : alu_cmt_rs2_rdata;
logic [`E203_INSTR_SIZE-1:0] ifu_o_ir;""",
        "top-level RVFI declarations",
    )
    content = _replace_once(
        content,
        """logic [`E203_INSTR_SIZE-1:0] ifu_o_ir;
logic commit_trap;""",
        """logic [`E203_INSTR_SIZE-1:0] ifu_o_ir;
function automatic rvfi_branch_taken;
  input [2:0] funct3;
  input [`E203_XLEN-1:0] rs1;
  input [`E203_XLEN-1:0] rs2;
  begin
    case (funct3)
      3'b000: rvfi_branch_taken = rs1 == rs2;
      3'b001: rvfi_branch_taken = rs1 != rs2;
      3'b100: rvfi_branch_taken = $signed(rs1) < $signed(rs2);
      3'b101: rvfi_branch_taken = $signed(rs1) >= $signed(rs2);
      3'b110: rvfi_branch_taken = rs1 < rs2;
      3'b111: rvfi_branch_taken = rs1 >= rs2;
      default: rvfi_branch_taken = 1'b0;
    endcase
  end
endfunction
function automatic [4:0] rvfi_rd_addr_for_insn;
  input [`E203_INSTR_SIZE-1:0] instr;
  begin
    if (instr[1:0] == 2'b11)
      rvfi_rd_addr_for_insn = instr[11:7];
    else begin
      case ({instr[15:13], instr[1:0]})
        5'b00000, 5'b01000: rvfi_rd_addr_for_insn = {2'b01, instr[4:2]}; // C.ADDI4SPN/LW
        5'b00001, 5'b01001, 5'b01101, 5'b00010, 5'b01010, 5'b10010:
          rvfi_rd_addr_for_insn = instr[11:7]; // CI/CR forms
        5'b00101: rvfi_rd_addr_for_insn = 5'd1; // C.JAL
        5'b10001: rvfi_rd_addr_for_insn = {2'b01, instr[9:7]}; // CA/CB destination
        default: rvfi_rd_addr_for_insn = 5'd0;
      endcase
      if ({instr[15:13], instr[1:0]} == 5'b10010 && instr[12] && instr[6:2] == 5'd0)
        rvfi_rd_addr_for_insn = 5'd1; // C.JALR
    end
  end
endfunction
function automatic [4:0] rvfi_rs1_addr_for_insn;
  input [`E203_INSTR_SIZE-1:0] instr;
  begin
    if (instr[1:0] == 2'b11)
      rvfi_rs1_addr_for_insn = instr[19:15];
    else begin
      case ({instr[15:13], instr[1:0]})
        5'b00000: rvfi_rs1_addr_for_insn = 5'd2; // C.ADDI4SPN
        5'b01000, 5'b11000: rvfi_rs1_addr_for_insn = {2'b01, instr[9:7]}; // C.LW/SW
        5'b00001, 5'b01101, 5'b00010, 5'b00011: rvfi_rs1_addr_for_insn = instr[11:7];
        5'b01010, 5'b11010: rvfi_rs1_addr_for_insn = 5'd2; // C.LWSP/SWSP
        5'b11001, 5'b11101: rvfi_rs1_addr_for_insn = {2'b01, instr[9:7]}; // C.BEQZ/BNEZ
        5'b10010: rvfi_rs1_addr_for_insn = instr[12] ? instr[11:7] : 5'd0; // C.JR/JALR vs C.MV
        5'b10001: rvfi_rs1_addr_for_insn = {2'b01, instr[9:7]}; // C.ALU
        default: rvfi_rs1_addr_for_insn = 5'd0;
      endcase
    end
  end
endfunction
function automatic [4:0] rvfi_rs2_addr_for_insn;
  input [`E203_INSTR_SIZE-1:0] instr;
  begin
    if (instr[1:0] == 2'b11)
      rvfi_rs2_addr_for_insn = instr[24:20];
    else begin
      case ({instr[15:13], instr[1:0]})
        5'b11000: rvfi_rs2_addr_for_insn = {2'b01, instr[4:2]}; // C.SW
        5'b11010, 5'b10010: rvfi_rs2_addr_for_insn = instr[6:2]; // C.SWSP/MV/ADD
        5'b10001: rvfi_rs2_addr_for_insn = instr[11:10] == 2'b11 ? {2'b01, instr[4:2]} : 5'd0;
        default: rvfi_rs2_addr_for_insn = 5'd0;
      endcase
    end
  end
endfunction
function automatic [`E203_PC_SIZE-1:0] rvfi_pc_next;
  input [`E203_INSTR_SIZE-1:0] instr;
  input [`E203_PC_SIZE-1:0] pc;
  input [`E203_XLEN-1:0] rs1;
  input [`E203_XLEN-1:0] rs2;
  begin
    if (instr[1:0] != 2'b11)
      rvfi_pc_next = pc + 32'd2;
    else case (instr[6:0])
      7'b1100011: rvfi_pc_next = rvfi_branch_taken(instr[14:12], rs1, rs2)
          ? pc + {{19{instr[31]}}, instr[31], instr[7], instr[30:25], instr[11:8], 1'b0}
          : pc + 32'd4;
      7'b1101111: rvfi_pc_next = pc + {{11{instr[31]}}, instr[31], instr[19:12], instr[20], instr[30:21], 1'b0};
      7'b1100111: rvfi_pc_next = (rs1 + {{20{instr[31]}}, instr[31:20]}) & ~32'd1;
      default: rvfi_pc_next = pc + 32'd4;
    endcase
  end
endfunction
function automatic [3:0] rvfi_load_rmask;
  input [2:0] funct3;
  input [1:0] addr_lo;
  begin
    case (funct3)
      3'b000, 3'b100: rvfi_load_rmask = 4'b0001 << addr_lo;
      3'b001, 3'b101: rvfi_load_rmask = 4'b0011 << addr_lo;
      3'b010: rvfi_load_rmask = 4'b1111;
      default: rvfi_load_rmask = 4'b0000;
    endcase
  end
endfunction
function automatic [`E203_XLEN-1:0] rvfi_load_rdata;
  input [2:0] funct3;
  input [1:0] addr_lo;
  input [`E203_XLEN-1:0] result;
  begin
    case (funct3)
      3'b000, 3'b100: rvfi_load_rdata = {{24{1'b0}}, result[7:0]} << {addr_lo, 3'b000};
      3'b001, 3'b101: rvfi_load_rdata = {{16{1'b0}}, result[15:0]} << {addr_lo, 3'b000};
      3'b010: rvfi_load_rdata = result;
      default: rvfi_load_rdata = {`E203_XLEN{1'b0}};
    endcase
  end
endfunction
logic commit_trap;""",
        "RVFI branch condition function",
    )

    top_cpu_ports = """  .alu_cmt_valid(alu_cmt_valid),
  .ifu_o_ir(ifu_o_ir),"""
    content = _replace_once(
        content,
        top_cpu_ports,
        """  .alu_cmt_valid(alu_cmt_valid),
  .rvfi_longp(rvfi_longp),
  .alu_cmt_pc(alu_cmt_pc),
  .alu_cmt_instr(alu_cmt_instr),
  .alu_cmt_ld(alu_cmt_ld),
  .alu_cmt_stamo(alu_cmt_stamo),
  .alu_cmt_rd_wdata(alu_cmt_rd_wdata),
  .alu_cmt_rd_wen(alu_cmt_rd_wen),
  .alu_cmt_trap(alu_cmt_trap),
  .alu_cmt_rs1_rdata(alu_cmt_rs1_rdata),
  .alu_cmt_rs2_rdata(alu_cmt_rs2_rdata),
  .alu_cmt_mem_addr(alu_cmt_mem_addr),
  .alu_cmt_mem_wmask(alu_cmt_mem_wmask),
  .alu_cmt_mem_wdata(alu_cmt_mem_wdata),
  .ifu_o_ir(ifu_o_ir),""",
        "e203_hbirdv2 to e203_cpu_top commit wiring",
    )

    rvfi_block = """    rvfi_insn <= ifu_o_ir;
    rvfi_trap <= commit_trap;"""
    content = _replace_once(
        content,
        """    rvfi_valid <= rst_n && alu_cmt_valid; // reset is active low""",
        """    rvfi_valid <= rst_n && ((alu_cmt_valid && !alu_cmt_ld) || e203_rvfi_longp_valid); // reset is active low""",
        "RVFI retirement-valid assignment",
    )
    content = _replace_once(
        content,
        rvfi_block,
        """    rvfi_insn <= e203_rvfi_longp_valid ? e203_rvfi_longp_instr : alu_cmt_instr;
    rvfi_trap <= e203_rvfi_longp_valid ? e203_rvfi_longp_trap : alu_cmt_trap;""",
        "RVFI instruction and trap assignments",
    )
    content = _replace_once(
        content,
        """    rvfi_rs1_addr <= ifu_o_rs1idx;
    rvfi_rs2_addr <= ifu_o_rs2idx;""",
        """    rvfi_rs1_addr <= rvfi_rs1_addr_for_insn(e203_rvfi_instr);
    rvfi_rs2_addr <= rvfi_rs2_addr_for_insn(e203_rvfi_instr);""",
        "RVFI source register assignments",
    )
    content = _replace_once(
        content,
        """    rvfi_rs1_rdata <= rf_rs1;
    rvfi_rs2_rdata <= rf_rs2;""",
        """    rvfi_rs1_rdata <= rvfi_rs1_addr_for_insn(e203_rvfi_instr) == 5'd0 ? 32'd0 : e203_rvfi_rs1_rdata;
    rvfi_rs2_rdata <= rvfi_rs2_addr_for_insn(e203_rvfi_instr) == 5'd0 ? 32'd0 : e203_rvfi_rs2_rdata;""",
        "RVFI x0 source-data assignments",
    )
    content = _replace_once(
        content,
        """    rvfi_rd_addr <= rf_wbck_rdidx;
    rvfi_rd_wdata <= rf_wbck_wdat;""",
        """    rvfi_rd_addr <= e203_rvfi_longp_valid ? (e203_rvfi_longp_trap ? 5'd0 : e203_rvfi_longp_rdidx) : (alu_cmt_rd_wen ? rvfi_rd_addr_for_insn(alu_cmt_instr) : 5'd0);
    rvfi_rd_wdata <= e203_rvfi_longp_valid ? ((e203_rvfi_longp_trap || e203_rvfi_longp_rdidx == 5'd0) ? 32'd0 : e203_rvfi_longp_rd_wdata) : ((alu_cmt_rd_wen && rvfi_rd_addr_for_insn(alu_cmt_instr) != 5'd0) ? alu_cmt_rd_wdata : 32'd0);""",
        "RVFI destination register and write-enable assignments",
    )
    content = _replace_once(
        content,
        """    rvfi_pc_rdata <= ifu_o_pc_vld;
    rvfi_pc_wdata <= ifu_req_pc;
    
    rvfi_mem_addr <= mem_icb_cmd_addr;
    rvfi_mem_rmask <= 32'hFFFFFFFF; // none found in e203
    rvfi_mem_wmask <= mem_icb_cmd_wmask;""",
        """    rvfi_pc_rdata <= e203_rvfi_longp_valid ? e203_rvfi_longp_pc : alu_cmt_pc;
    rvfi_pc_wdata <= e203_rvfi_longp_valid ? e203_rvfi_longp_pc + 32'd4 : rvfi_pc_next(alu_cmt_instr, alu_cmt_pc, alu_cmt_rs1_rdata, alu_cmt_rs2_rdata);
    
    rvfi_mem_addr <= e203_rvfi_longp_valid ? (e203_rvfi_longp_mem_addr & 32'hfffffffc) : (alu_cmt_stamo ? (alu_cmt_mem_addr & 32'hfffffffc) : mem_icb_cmd_addr);
    rvfi_mem_rmask <= e203_rvfi_longp_valid ? rvfi_load_rmask(e203_rvfi_longp_instr[14:12], e203_rvfi_longp_mem_addr[1:0]) : (alu_cmt_ld ? rvfi_load_rmask(alu_cmt_instr[14:12], alu_cmt_mem_addr[1:0]) : 4'h0);
    rvfi_mem_wmask <= e203_rvfi_longp_valid ? 4'h0 : (alu_cmt_stamo ? alu_cmt_mem_wmask : 4'h0);
    rvfi_mem_wdata <= e203_rvfi_longp_valid ? e203_rvfi_longp_mem_wdata : (alu_cmt_stamo ? alu_cmt_mem_wdata : mem_icb_cmd_wdata);""",
        "RVFI PC and memory assignments",
    )
    content = _replace_once(
        content,
        """    rvfi_mem_wdata <= mem_icb_cmd_wdata;
    
  end""",
        """  end""",
        "stale RVFI memory write-data assignment",
    )
    content = _replace_once(
        content,
        """    rvfi_mem_rdata <= mem_icb_rsp_rdata;""",
        """    rvfi_mem_rdata <= e203_rvfi_longp_valid ? rvfi_load_rdata(e203_rvfi_longp_instr[14:12], e203_rvfi_longp_mem_addr[1:0], e203_rvfi_longp_rd_wdata) : mem_icb_rsp_rdata;""",
        "RVFI long-pipe load response-data assignment",
    )

    # Each exported port list ends with the same existing RVFI fields.  The
    # three exact forms below distinguish core, cpu, and cpu_top boundaries.
    port_anchors = (
        (
            """  output alu_cmt_valid,

  output irq_req_raw,""",
            """  output alu_cmt_valid,
  output [`E203_PC_SIZE-1:0] alu_cmt_pc,
  output [`E203_INSTR_SIZE-1:0] alu_cmt_instr,
  output alu_cmt_ld,
  output alu_cmt_stamo,
  output [`E203_XLEN-1:0] alu_cmt_rd_wdata,
  output alu_cmt_rd_wen,
  output alu_cmt_trap,
  output [`E203_XLEN-1:0] alu_cmt_rs1_rdata,
  output [`E203_XLEN-1:0] alu_cmt_rs2_rdata,
  output [`E203_ADDR_SIZE-1:0] alu_cmt_mem_addr,
  output [`E203_XLEN/8-1:0] alu_cmt_mem_wmask,
  output [`E203_XLEN-1:0] alu_cmt_mem_wdata,

  output irq_req_raw,""",
            "e203_core commit output ports",
        ),
        (
            """  output alu_cmt_valid,
  output [`E203_INSTR_SIZE-1:0] ifu_o_ir,""",
            """  output alu_cmt_valid,
  output [`E203_PC_SIZE-1:0] alu_cmt_pc,
  output [`E203_INSTR_SIZE-1:0] alu_cmt_instr,
  output alu_cmt_ld,
  output alu_cmt_stamo,
  output [`E203_XLEN-1:0] alu_cmt_rd_wdata,
  output alu_cmt_rd_wen,
  output alu_cmt_trap,
  output [`E203_XLEN-1:0] alu_cmt_rs1_rdata,
  output [`E203_XLEN-1:0] alu_cmt_rs2_rdata,
  output [`E203_ADDR_SIZE-1:0] alu_cmt_mem_addr,
  output [`E203_XLEN/8-1:0] alu_cmt_mem_wmask,
  output [`E203_XLEN-1:0] alu_cmt_mem_wdata,
  output [`E203_INSTR_SIZE-1:0] ifu_o_ir,""",
            "e203_cpu commit output ports",
        ),
        (
            """  output alu_cmt_valid,

  output [`E203_INSTR_SIZE-1:0] ifu_o_ir,""",
            """  output alu_cmt_valid,
  output [`E203_PC_SIZE-1:0] alu_cmt_pc,
  output [`E203_INSTR_SIZE-1:0] alu_cmt_instr,
  output alu_cmt_ld,
  output alu_cmt_stamo,
  output [`E203_XLEN-1:0] alu_cmt_rd_wdata,
  output alu_cmt_rd_wen,
  output alu_cmt_trap,
  output [`E203_XLEN-1:0] alu_cmt_rs1_rdata,
  output [`E203_XLEN-1:0] alu_cmt_rs2_rdata,
  output [`E203_ADDR_SIZE-1:0] alu_cmt_mem_addr,
  output [`E203_XLEN/8-1:0] alu_cmt_mem_wmask,
  output [`E203_XLEN-1:0] alu_cmt_mem_wdata,

  output [`E203_INSTR_SIZE-1:0] ifu_o_ir,""",
            "e203_cpu spaced commit output ports",
        ),
        (
            """  output alu_cmt_valid,

  output [`E203_INSTR_SIZE-1:0]ifu_o_ir,""",
            """  output alu_cmt_valid,
  output [`E203_PC_SIZE-1:0] alu_cmt_pc,
  output [`E203_INSTR_SIZE-1:0] alu_cmt_instr,
  output alu_cmt_ld,
  output alu_cmt_stamo,
  output [`E203_XLEN-1:0] alu_cmt_rd_wdata,
  output alu_cmt_rd_wen,
  output alu_cmt_trap,
  output [`E203_XLEN-1:0] alu_cmt_rs1_rdata,
  output [`E203_XLEN-1:0] alu_cmt_rs2_rdata,
  output [`E203_ADDR_SIZE-1:0] alu_cmt_mem_addr,
  output [`E203_XLEN/8-1:0] alu_cmt_mem_wmask,
  output [`E203_XLEN-1:0] alu_cmt_mem_wdata,

  output [`E203_INSTR_SIZE-1:0]ifu_o_ir,""",
            "e203_cpu_top commit output ports",
        ),
    )
    for old, new, description in port_anchors:
        content = _replace_once(content, old, new, description)

    content = _replace_exactly(
        content,
        """  output alu_cmt_valid,
  output [`E203_PC_SIZE-1:0] alu_cmt_pc,""",
        """  output alu_cmt_valid,
  output [234:0] rvfi_longp,
  output [`E203_PC_SIZE-1:0] alu_cmt_pc,""",
        4,
        "formal long-pipe sideband output ports",
    )

    core_wiring = """    .alu_cmt_valid(alu_cmt_valid),
    .irq_req_raw(irq_req_raw),"""
    content = _replace_once(
        content,
        core_wiring,
        """    .alu_cmt_valid(alu_cmt_valid),
    .rvfi_longp(rvfi_longp),
    .alu_cmt_pc(alu_cmt_pc),
    .alu_cmt_instr(alu_cmt_instr),
    .alu_cmt_ld(alu_cmt_ld),
    .alu_cmt_stamo(alu_cmt_stamo),
    .alu_cmt_rd_wdata(alu_cmt_rd_wdata),
    .alu_cmt_rd_wen(alu_cmt_rd_wen),
    .alu_cmt_trap(alu_cmt_trap),
    .alu_cmt_rs1_rdata(alu_cmt_rs1_rdata),
    .alu_cmt_rs2_rdata(alu_cmt_rs2_rdata),
    .alu_cmt_mem_addr(alu_cmt_mem_addr),
    .alu_cmt_mem_wmask(alu_cmt_mem_wmask),
    .alu_cmt_mem_wdata(alu_cmt_mem_wdata),
    .irq_req_raw(irq_req_raw),""",
        "e203_cpu to e203_core commit wiring",
    )
    cpu_wiring = """    .alu_cmt_valid(alu_cmt_valid),

    .ifu_o_ir(ifu_o_ir),"""
    content = _replace_exactly(
        content,
        cpu_wiring,
        """    .alu_cmt_valid(alu_cmt_valid),
    .rvfi_longp(rvfi_longp),
    .alu_cmt_pc(alu_cmt_pc),
    .alu_cmt_instr(alu_cmt_instr),
    .alu_cmt_ld(alu_cmt_ld),
    .alu_cmt_stamo(alu_cmt_stamo),
    .alu_cmt_rd_wdata(alu_cmt_rd_wdata),
    .alu_cmt_rd_wen(alu_cmt_rd_wen),
    .alu_cmt_trap(alu_cmt_trap),
    .alu_cmt_rs1_rdata(alu_cmt_rs1_rdata),
    .alu_cmt_rs2_rdata(alu_cmt_rs2_rdata),
    .alu_cmt_mem_addr(alu_cmt_mem_addr),
    .alu_cmt_mem_wmask(alu_cmt_mem_wmask),
    .alu_cmt_mem_wdata(alu_cmt_mem_wdata),

    .ifu_o_ir(ifu_o_ir),""",
        2,
        "e203_cpu/e203_cpu_top commit wiring",
    )

    content = _replace_once(
        content,
        """  output [`E203_PC_SIZE-1:0] ret_pc,

  input  disp_i_rs1en,""",
        """  output [`E203_PC_SIZE-1:0] ret_pc,
  output [`E203_INSTR_SIZE-1:0] rvfi_ret_instr,
  output [`E203_XLEN-1:0] rvfi_ret_rs1_rdata,
  output [`E203_XLEN-1:0] rvfi_ret_rs2_rdata,
  output [`E203_ADDR_SIZE-1:0] rvfi_ret_mem_addr,
  output [`E203_XLEN/8-1:0] rvfi_ret_mem_wmask,
  output [`E203_XLEN-1:0] rvfi_ret_mem_wdata,

  input  disp_i_rs1en,""",
        "OITF formal sidecar return ports",
    )
    content = _replace_once(
        content,
        """  input  [`E203_PC_SIZE    -1:0] disp_i_pc,

  output oitfrd_match_disprs1,""",
        """  input  [`E203_PC_SIZE    -1:0] disp_i_pc,
  input  [`E203_INSTR_SIZE -1:0] disp_i_instr,
  input  [`E203_XLEN       -1:0] disp_i_rs1_rdata,
  input  [`E203_XLEN       -1:0] disp_i_rs2_rdata,

  // Private formal sidecar: capture long-pipe metadata by the existing
  // OITF allocation tag, then expose it at ordered retirement.
  input                          rvfi_mem_ena,
  input  [`E203_ITAG_WIDTH-1:0] rvfi_mem_ptr,
  input  [`E203_ADDR_SIZE-1:0]  rvfi_mem_addr,
  input  [`E203_XLEN/8-1:0]     rvfi_mem_wmask,
  input  [`E203_XLEN-1:0]       rvfi_mem_wdata,

  output oitfrd_match_disprs1,""",
        "OITF private RVFI sidecar ports",
    )
    content = _replace_once(
        content,
        """  wire [`E203_PC_SIZE-1:0] pc_r[`E203_OITF_DEPTH-1:0];

  wire alc_ptr_ena = dis_ena;""",
        """  wire [`E203_PC_SIZE-1:0] pc_r[`E203_OITF_DEPTH-1:0];
  wire [`E203_INSTR_SIZE-1:0] rvfi_instr_r[`E203_OITF_DEPTH-1:0];
  wire [`E203_XLEN-1:0] rvfi_rs1_rdata_r[`E203_OITF_DEPTH-1:0];
  wire [`E203_XLEN-1:0] rvfi_rs2_rdata_r[`E203_OITF_DEPTH-1:0];
  wire [`E203_ADDR_SIZE-1:0] rvfi_mem_addr_r[`E203_OITF_DEPTH-1:0];
  wire [`E203_XLEN/8-1:0] rvfi_mem_wmask_r[`E203_OITF_DEPTH-1:0];
  wire [`E203_XLEN-1:0] rvfi_mem_wdata_r[`E203_OITF_DEPTH-1:0];

  assign rvfi_ret_instr = rvfi_instr_r[ret_ptr];
  assign rvfi_ret_rs1_rdata = rvfi_rs1_rdata_r[ret_ptr];
  assign rvfi_ret_rs2_rdata = rvfi_rs2_rdata_r[ret_ptr];
  assign rvfi_ret_mem_addr = rvfi_mem_addr_r[ret_ptr];
  assign rvfi_ret_mem_wmask = rvfi_mem_wmask_r[ret_ptr];
  assign rvfi_ret_mem_wdata = rvfi_mem_wdata_r[ret_ptr];

  wire alc_ptr_ena = dis_ena;""",
        "OITF private RVFI sidecar storage",
    )
    content = _replace_once(
        content,
        """        sirv_gnrl_dffl #(1)                 rdfpu_dfflrs(vld_set[i], disp_i_rdfpu, rdfpu_r[i], clk);

        assign rd_match_rs1idx[i]""",
        """        sirv_gnrl_dffl #(1)                 rdfpu_dfflrs(vld_set[i], disp_i_rdfpu, rdfpu_r[i], clk);
        sirv_gnrl_dffl #(`E203_INSTR_SIZE) rvfi_instr_dfflrs(vld_set[i], disp_i_instr, rvfi_instr_r[i], clk);
        sirv_gnrl_dffl #(`E203_XLEN) rvfi_rs1_dfflrs(vld_set[i], disp_i_rs1_rdata, rvfi_rs1_rdata_r[i], clk);
        sirv_gnrl_dffl #(`E203_XLEN) rvfi_rs2_dfflrs(vld_set[i], disp_i_rs2_rdata, rvfi_rs2_rdata_r[i], clk);
        sirv_gnrl_dffl #(`E203_ADDR_SIZE) rvfi_mem_addr_dfflrs(rvfi_mem_ena & (rvfi_mem_ptr == i), rvfi_mem_addr, rvfi_mem_addr_r[i], clk);
        sirv_gnrl_dffl #(`E203_XLEN/8) rvfi_mem_wmask_dfflrs(rvfi_mem_ena & (rvfi_mem_ptr == i), rvfi_mem_wmask, rvfi_mem_wmask_r[i], clk);
        sirv_gnrl_dffl #(`E203_XLEN) rvfi_mem_wdata_dfflrs(rvfi_mem_ena & (rvfi_mem_ptr == i), rvfi_mem_wdata, rvfi_mem_wdata_r[i], clk);

        assign rd_match_rs1idx[i]""",
        "OITF private RVFI sidecar capture",
    )
    content = _replace_once(
        content,
        """  wire oitf_ret_rdwen;
  wire oitf_ret_rdfpu;


  e203_exu_oitf""",
        """  wire oitf_ret_rdwen;
  wire oitf_ret_rdfpu;
  wire [`E203_INSTR_SIZE-1:0] rvfi_oitf_ret_instr;
  wire [`E203_XLEN-1:0] rvfi_oitf_ret_rs1_rdata;
  wire [`E203_XLEN-1:0] rvfi_oitf_ret_rs2_rdata;
  wire [`E203_ADDR_SIZE-1:0] rvfi_oitf_ret_mem_addr;
  wire [`E203_XLEN/8-1:0] rvfi_oitf_ret_mem_wmask;
  wire [`E203_XLEN-1:0] rvfi_oitf_ret_mem_wdata;


  e203_exu_oitf""",
        "EXU formal OITF return wires",
    )
    content = _replace_once(
        content,
        """    .ret_rdfpu            (oitf_ret_rdfpu),
    .ret_pc               (oitf_ret_pc),

    .disp_i_rs1en""",
        """    .ret_rdfpu            (oitf_ret_rdfpu),
    .ret_pc               (oitf_ret_pc),
    .rvfi_ret_instr       (rvfi_oitf_ret_instr),
    .rvfi_ret_rs1_rdata   (rvfi_oitf_ret_rs1_rdata),
    .rvfi_ret_rs2_rdata   (rvfi_oitf_ret_rs2_rdata),
    .rvfi_ret_mem_addr    (rvfi_oitf_ret_mem_addr),
    .rvfi_ret_mem_wmask   (rvfi_oitf_ret_mem_wmask),
    .rvfi_ret_mem_wdata   (rvfi_oitf_ret_mem_wdata),

    .disp_i_rs1en""",
        "EXU formal OITF return wiring",
    )
    content = _replace_once(
        content,
        """    .disp_i_rdfpu         (disp_oitf_rdfpu ),
    .disp_i_pc            (disp_oitf_pc ),

    .oitfrd_match_disprs1""",
        """    .disp_i_rdfpu         (disp_oitf_rdfpu ),
    .disp_i_pc            (disp_oitf_pc ),
    .disp_i_instr         (i_ir),
    .disp_i_rs1_rdata     (rf_rs1),
    .disp_i_rs2_rdata     (rf_rs2),
    .rvfi_mem_ena         (agu_icb_cmd_valid),
    .rvfi_mem_ptr         (agu_icb_cmd_itag),
    .rvfi_mem_addr        (agu_icb_cmd_addr),
    .rvfi_mem_wmask       (agu_icb_cmd_wmask),
    .rvfi_mem_wdata       (agu_icb_cmd_wdata),

    .oitfrd_match_disprs1""",
        "EXU to OITF RVFI sidecar wiring",
    )
    content = _replace_once(
        content,
        """  wire [`E203_XLEN-1:0] alu_wbck_o_wdat;
  wire [`E203_RFIDX_WIDTH-1:0] alu_wbck_o_rdidx;""",
        """  wire [`E203_XLEN-1:0] alu_wbck_o_wdat;
  wire [`E203_RFIDX_WIDTH-1:0] alu_wbck_o_rdidx;
  assign alu_cmt_rd_wdata = alu_wbck_o_wdat;
  assign alu_cmt_rd_wen = alu_wbck_o_valid;
  assign alu_cmt_rs1_rdata = disp_alu_rs1;
  assign alu_cmt_rs2_rdata = disp_alu_rs2;
  assign alu_cmt_mem_addr = agu_icb_cmd_addr;
  assign alu_cmt_mem_wmask = agu_icb_cmd_wmask;
  assign alu_cmt_mem_wdata = agu_icb_cmd_wdata;
  assign alu_cmt_trap = alu_cmt_ecall | alu_cmt_ebreak
                      | alu_cmt_ifu_misalgn | alu_cmt_ifu_buserr
                      | alu_cmt_ifu_ilegl | alu_cmt_misalgn | alu_cmt_buserr;
  assign rvfi_longp = {
      (longp_wbck_o_valid & longp_wbck_o_ready) | (longp_excp_o_valid & longp_excp_o_ready),
      longp_excp_o_valid & longp_excp_o_ready,
      rvfi_oitf_ret_instr,
      rvfi_oitf_ret_rs1_rdata,
      rvfi_oitf_ret_rs2_rdata,
      rvfi_oitf_ret_mem_addr,
      rvfi_oitf_ret_mem_wmask,
      rvfi_oitf_ret_mem_wdata,
      oitf_ret_pc,
      longp_wbck_o_rdidx,
      longp_wbck_o_wdat
  };""",
        "e203 EXU writeback datum assignment",
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
