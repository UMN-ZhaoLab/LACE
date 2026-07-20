arithmetic_template = """
## Example of arithmetic implementation

 

// SystemVerilog file 
 
module SCAL (
    // Interface to the ISAX Module
    
    input  [32 -1 : 0] WrRD_bitwise_rotation_2_i,// ISAX
    input   WrRD_validReq_bitwise_rotation_2_i,// ISAX
    output  [32 -1 : 0] RdRS1_1_o,// ISAX
    output  [32 -1 : 0] RdRS2_1_o,// ISAX
    output   RdIValid_bitwise_rotation_1_o,// ISAX
    
    
    // Interface to the Core
    
    output reg [32 -1 : 0] WrRD_2_o,// ISAX
    output reg  WrRD_validReq_2_o,// ISAX
    input  [32 -1 : 0] RdRS1_1_i,// ISAX
    input  [32 -1 : 0] RdRS2_1_i,// ISAX
    input   RdFlush_0_i,// ISAX
    input   RdFlush_1_i,// ISAX
    input   RdFlush_2_i,// ISAX
    input   RdStall_0_i,// ISAX
    input   RdStall_1_i,// ISAX
    input  [32 -1 : 0] RdInstr_0_i,// ISAX
    
    input clk_i,
    input rst_i
    
    
);
// Declare local signals
wire  RdIValid_bitwise_rotation_0_s;
wire  RdIValid_bitwise_rotation_1_s;
reg  RdIValid_bitwise_rotation_1_reg;
wire  RdIValid_bitwise_rotation_2_s;
reg  RdIValid_bitwise_rotation_2_reg;
wire  RdFlush_2_s;
wire  RdFlush_1_s;
wire  RdFlush_0_s;
wire RdStall_0_s;
wire RdStall_1_s;


// Logic
assign RdRS1_1_o = RdRS1_1_i;
assign RdRS2_1_o = RdRS2_1_i;
assign RdIValid_bitwise_rotation_0_s = (  ( RdInstr_0_i [ 14 ]  ==  1'b0 )  &&  ( RdInstr_0_i [ 13 ]  ==  1'b0 )  &&  ( RdInstr_0_i [ 12 ]  ==  1'b0 )  &&  ( RdInstr_0_i [ 6 ]  ==  1'b0 )  &&  ( RdInstr_0_i [ 5 ]  ==  1'b1 )  &&  ( RdInstr_0_i [ 4 ]  ==  1'b1 )  &&  ( RdInstr_0_i [ 3 ]  ==  1'b0 )  &&  ( RdInstr_0_i [ 2 ]  ==  1'b0 )  &&  ( RdInstr_0_i [ 1 ]  ==  1'b1 )  &&  ( RdInstr_0_i [ 0 ]  ==  1'b1 )  ) && !RdFlush_0_s;
assign RdIValid_bitwise_rotation_1_o = RdIValid_bitwise_rotation_1_s;
assign RdIValid_bitwise_rotation_1_s = RdIValid_bitwise_rotation_1_reg && !RdFlush_1_s;
always@(posedge clk_i) begin
    if (rst_i)
        RdIValid_bitwise_rotation_1_reg <= 0;
    else if (!(RdStall_0_s))
        RdIValid_bitwise_rotation_1_reg <= RdIValid_bitwise_rotation_0_s;
end;

assign RdIValid_bitwise_rotation_2_s = RdIValid_bitwise_rotation_2_reg && !RdFlush_2_s;
always@(posedge clk_i) begin
    if (rst_i)
        RdIValid_bitwise_rotation_2_reg <= 0;
    else if (!(RdStall_1_s))
        RdIValid_bitwise_rotation_2_reg <= RdIValid_bitwise_rotation_1_s;
end;

always @(*)  WrRD_2_o = WrRD_bitwise_rotation_2_i;
always @(*) begin 
    case(1'b1)
        RdIValid_bitwise_rotation_2_s : WrRD_validReq_2_o = WrRD_validReq_bitwise_rotation_2_i;
        default : WrRD_validReq_2_o = ~1;
    endcase
end
assign RdFlush_2_o = RdFlush_2_s;
assign RdFlush_2_s = RdFlush_2_i;
assign RdFlush_1_o = RdFlush_1_s;
assign RdFlush_1_s = RdFlush_1_i;
assign RdFlush_0_o = RdFlush_0_s;
assign RdFlush_0_s = RdFlush_0_i;
assign RdStall_0_s = RdStall_0_i;
assign RdStall_1_s = RdStall_1_i;


endmodule

"""
