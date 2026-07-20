system_prompt = """
You are a RISC-V ISA expert with strong instruction-set design and microarchitecture analysis skills.
Your task is to decompose a custom ISA extension into a sequence of predefined micro-operations.

You will be given:
- The ISA extension description and encoding
- A reference table of predefined operations

CRITICAL FORMAT RULES:
1. EVERY operation MUST be assigned to a variable on the left-hand side, except writes (WrRD, WrPC, WrMem, WrCustReg).
   Correct:   `insn = RdInstr()`, `rs1 = RdRS1()`, `sum = ADD(a, b)`
   Wrong:     `RdInstr()`, `RdRS1()`, `ADD(a, b)`  (missing assignment)
2. RdInstr(), RdRS1(), RdRS2() return values. Capture them: `insn = RdInstr()`, NOT `RdInstr()` alone.
3. WrRD(value) must receive the value to write. Do NOT write bare `WrRD()`.
4. WrPC(new_pc) must receive the new PC value.
5. `ops` MUST ONLY use the provided INTERFACE operations. No arithmetic operations (ADD, MUL, SLL, AND, OR, etc.) are allowed in `ops`. If you need arithmetic, put it in `arithmetic_ops`.
6. Every op in `ops` must be a valid assignment or write statement using only interface operations.

You MUST return TWO outputs:
- `ops`: The simplified interface-level micro-ops list (JSON array of strings), used for HDL integration. ONLY use interface operations (RdInstr, RdRS1, RdRS2, RdPC, RdMem, WrRD, WrPC, WrMem, CreateCustReg, RdCustReg, WrCustReg).
- `arithmetic_ops`: A SINGLE text block containing the precise arithmetic semantics in Sail-like pseudocode. This describes the exact computation: bit widths, zero/sign extensions, arithmetic/logic operations, and standard primitive calls (e.g., `aes_sbox_fwd`, `vector_add_byte`). Use Sail syntax: `let var : bits(N) = expression;`. Register reads/writes in Sail use `X(addr)` notation.

Return a confidence level (high/medium/low).
"""

bitwise_rotation = """
# bitwise rotation instruction doc
## encoding
- opcode [6:0] = 0110011
- rd [11:7]
- funct3 [14:12] = 001
- rs1 [19:15]
- rs2 [24:20]
- funct7 [31:25] = 0110000

## function description
*rol*: This instruction performs a rotate left of rs1 register by the amount in least-significant five bits of rs2 (4:0).
"""

sbox_aes = """
# sbox aes instruction doc
## encoding
- opcode [6:0] = 0110011
- rd [11:7]
- funct3 [14:12] = 000
- rs1 [19:15]
- rs2 [24:20]
- funct7 [29:25] = 10101
- bs [31:30]

## function description
*aes32esi*:
1. extract byte from rs2 at offset (bs times 8)
2. apply AES forward S-Box, zero-extend to 32 bits
3. left-rotate by offset bits, XOR with rs1,
4. write result to rd.
"""

vector_add = """
# vector add instruction doc
## encoding
- opcode [6:0] = 1010111
- vd [11:7]
- funct3 [14:12] = 000
- vs1 [19:15]
- vs2 [24:20]
- vm [25] = 1
- funct6 [31:26]

## function description
*vadd.vv*: add the vector vs1 and vector vs2 together. The vector registers are not included in the standard RV32I processor and should be added. A vector register is divided into 4 equal-length variables and calculations are performed on them respectively.
"""

indirect_jump = """
# indirect jump instruction doc
## encoding
- opcode [6:0] = 0001011
- rd [11:7]
- funct3 [14:12] = 000
- rs1 [19:15]
- offset [31:20]

## function description
*ijumpl*: This is a jump-and-link instruction. It computes the target address as `rs1 + sign_extended(offset)`, saves `PC + 4` into `rd`, and updates the PC to the target address. It does NOT read memory.
"""

load_multiple = """
# load multiple instruction doc
## encoding
- opcode [6:0] = 0101011
- rd [11:7]
- funct3 [14:12] = 000
- rs1 [19:15]
- bitmask [31:20]

## function description
*ldmul*: This instruction conditionally loads multiple callee-saved registers from memory. The 12-bit bitmask [11:0] controls which registers to load: bit 0 -> x8 (s0), bit 1 -> x9 (s1), bits 2-11 -> x18-x27 (s2-s11). The base address is in rs1. Memory is read sequentially starting at rs1 with 4-byte increments. Only when the corresponding bitmask bit is 1, the loaded value is written to the target register. Registers whose bit is 0 remain unchanged.
"""

from src.ops_registry import format_predefined_ops
predefined_ops = format_predefined_ops(include_arithmetic=False)

spec2op_example = """
Spec:
# 4x8bit dot-product instruction doc
## encoding
- opcode [6:0] = 0001011
- rd [11:7]
- funct3 [14:12] = 000
- rs1 [19:15]
- rs2 [24:20]
- funct7 [31:25] = 0000000

## function description
*x_dotp*: This instruction performs four eight bit dot-product.

Operations (ops - interface-level ONLY):
insn = RdInstr()
rs1 = RdRS1()
rs2 = RdRS2()
WrRD(result)

Arithmetic operations (arithmetic_ops - Sail-level):
```sail
let rs1 : bits(32) = X(insn[19 .. 15]);
let rs2 : bits(32) = X(insn[24 .. 20]);

let part1 : bits(16) = unsigned(rs1[7 .. 0])   * unsigned(rs2[7 .. 0]);
let part2 : bits(16) = unsigned(rs1[15 .. 8])  * unsigned(rs2[15 .. 8]);
let part3 : bits(16) = unsigned(rs1[23 .. 16]) * unsigned(rs2[23 .. 16]);
let part4 : bits(16) = unsigned(rs1[31 .. 24]) * unsigned(rs2[31 .. 24]);

let result : bits(32) = zero_extend(part1, 32) + zero_extend(part2, 32)
                        + zero_extend(part3, 32) + zero_extend(part4, 32);

X(insn[11 .. 7]) = result;
```

---

Spec:
# sbox aes instruction doc
## encoding
- opcode [6:0] = 0110011
- rd [11:7]
- funct3 [14:12] = 000
- rs1 [19:15]
- rs2 [24:20]
- funct7 [29:25] = 10101
- bs [31:30]

## function description
*aes32esi*:
1. extract byte from rs2 at offset (bs times 8)
2. apply AES forward S-Box, zero-extend to 32 bits
3. left-rotate by offset bits, XOR with rs1,
4. write result to rd.

Operations (ops - interface-level ONLY):
insn = RdInstr()
rs1 = RdRS1()
rs2 = RdRS2()
WrRD(result)

Arithmetic operations (arithmetic_ops - Sail-level, from official sail-riscv model):
```sail
let bs    : bits(2)  = insn[31 .. 30];
let shamt : bits(5)  = bs @ 0b000;                // shamt = bs*8
let si    : bits(8)  = (X(rs2) >> shamt)[7..0];   // SBox input byte
let so    : bits(32) = 0x000000 @ aes_sbox_fwd(si);
let result : bits(32) = X(rs1)[31..0] ^ (so <<< shamt);

X(insn[11 .. 7]) = sign_extend(result);
```
"""
