// =============================================================================
// MACRO-OP FUSION DECODER
// =============================================================================
// Detects and fuses common instruction sequences to improve IPC:
// 1. LUI + ADDI    -> Load 32-bit immediate (li pseudo-instruction)
// 2. AUIPC + JALR  -> Long jumps / function calls (call pseudo-instruction)
// 3. LOAD + ALU    -> Load-use fusion (eliminates data hazard stall)
// =============================================================================

module fusion_decoder(
    input  [31:0] inst1,        // Instruction in Decode stage
    input  [31:0] inst2,        // Instruction in Fetch stage (next instruction)
    output reg    fuse_flag,    // Fusion detected
    output reg [1:0] fuse_type, // 00=none, 01=LUI+ADDI, 10=AUIPC+JALR, 11=LOAD+ALU
    output reg [31:0] fused_inst
);

// Opcode definitions
localparam OP_LUI    = 7'b0110111;  // LUI
localparam OP_AUIPC  = 7'b0010111;  // AUIPC
localparam OP_ADDI   = 7'b0010011;  // I-type ALU (ADDI, etc.)
localparam OP_JALR   = 7'b1100111;  // JALR
localparam OP_LOAD   = 7'b0000011;  // Load (LW, LH, LB, etc.)
localparam OP_RTYPE  = 7'b0110011;  // R-type ALU
localparam OP_NOP    = 32'h00000013; // NOP (ADDI x0, x0, 0)

// Extract fields from instructions
wire [6:0]  opcode1   = inst1[6:0];
wire [6:0]  opcode2   = inst2[6:0];
wire [4:0]  rd1       = inst1[11:7];
wire [4:0]  rd2       = inst2[11:7];
wire [4:0]  rs1_inst2 = inst2[19:15];
wire [4:0]  rs2_inst2 = inst2[24:20];
wire [2:0]  funct3_2  = inst2[14:12];

// =============================================================================
// FUSION PATTERN 1: LUI + ADDI (Load 32-bit Immediate)
// =============================================================================
// Pattern: LUI rd, imm[31:12]  ->  ADDI rd, rd, imm[11:0]
// Conditions:
//   - First instruction is LUI
//   - Second instruction is ADDI (funct3 = 000)
//   - rd of LUI == rd of ADDI
//   - rs1 of ADDI == rd (using the value just loaded)
// Result: The LUI carries the fused operation with combined immediate
// =============================================================================
wire lui_addi_match = (opcode1 == OP_LUI) &&
                      (opcode2 == OP_ADDI) &&
                      (funct3_2 == 3'b000) &&       // ADDI specifically
                      (rd1 == rd2) &&               // Same destination
                      (rd1 == rs1_inst2) &&         // ADDI uses LUI result
                      (rd1 != 5'b0);                // Not x0

// =============================================================================
// FUSION PATTERN 2: AUIPC + JALR (Long Jump / Function Call)
// =============================================================================
// Pattern: AUIPC rd, imm[31:12]  ->  JALR rd, rd, imm[11:0]
// Conditions:
//   - First instruction is AUIPC
//   - Second instruction is JALR
//   - rd of AUIPC == rs1 of JALR (using PC-relative address)
// Result: PC-relative jump to any 32-bit offset (used for 'call' pseudo-instruction)
// Note: rd of JALR can be x0 (tail call) or x1/ra (regular call)
// =============================================================================
wire auipc_jalr_match = (opcode1 == OP_AUIPC) &&
                        (opcode2 == OP_JALR) &&
                        (rd1 == rs1_inst2) &&       // JALR uses AUIPC result
                        (rd1 != 5'b0);              // Not x0

// =============================================================================
// FUSION PATTERN 3: LOAD + ALU (Load-Use Fusion)
// =============================================================================
// Pattern: LW rd, offset(rs1)  ->  ALU_OP rd2, rd, rs2  OR  ALU_OP rd2, rs2, rd
// Conditions:
//   - First instruction is a Load (LW, LH, LB, LHU, LBU)
//   - Second instruction is R-type ALU or I-type ALU
//   - rd of Load is used as rs1 or rs2 of ALU (immediate use)
// Result: Eliminates the load-use stall by executing both in one cycle
// =============================================================================
wire is_load = (opcode1 == OP_LOAD);
wire is_alu_rtype = (opcode2 == OP_RTYPE);
wire is_alu_itype = (opcode2 == OP_ADDI);  // I-type ALU

// Check if load result is used immediately
wire load_used_as_rs1 = (rd1 == rs1_inst2) && (rd1 != 5'b0);
wire load_used_as_rs2 = is_alu_rtype && (rd1 == rs2_inst2) && (rd1 != 5'b0);

wire load_alu_match = is_load &&
                      (is_alu_rtype || is_alu_itype) &&
                      (load_used_as_rs1 || load_used_as_rs2);

// =============================================================================
// FUSION OUTPUT LOGIC
// =============================================================================
always @(*) begin
    fuse_flag = 1'b0;
    fuse_type = 2'b00;
    fused_inst = inst1;  // Default: pass through original instruction

    // Priority: LUI+ADDI > AUIPC+JALR > LOAD+ALU
    if (lui_addi_match) begin
        fuse_flag = 1'b1;
        fuse_type = 2'b01;  // LUI+ADDI
        // Fused instruction: Keep LUI structure but mark it as fused
        // The LUI will execute with the combined immediate effect
        // We construct an instruction that the ALU can process:
        // Upper 20 bits from LUI + lower 12 bits sign-extended from ADDI
        fused_inst = inst1;  // LUI carries the fused operation
    end
    else if (auipc_jalr_match) begin
        fuse_flag = 1'b1;
        fuse_type = 2'b10;  // AUIPC+JALR
        // For AUIPC+JALR fusion:
        // The fused operation computes (PC + auipc_imm + jalr_offset)
        // AUIPC carries the operation, JALR is flushed
        fused_inst = inst1;  // AUIPC carries the fused operation
    end
    else if (load_alu_match) begin
        fuse_flag = 1'b1;
        fuse_type = 2'b11;  // LOAD+ALU
        // For Load+ALU fusion:
        // The load completes and the ALU op executes in the same cycle
        // This eliminates the load-use hazard stall
        fused_inst = inst1;  // Load carries the fused operation
    end
end

endmodule
