module fusion_decoder(
    input  [31:0] inst1,
    input  [31:0] inst2,
    output reg    fuse_flag,
    output reg [31:0] fused_inst
);

always @(*) begin
    fuse_flag = 0;
    fused_inst = 32'b0;

    // Pattern: LUI + ADDI (load 32-bit immediate)
    if (inst1[6:0] == 7'b0110111 &&  // LUI opcode
        inst2[6:0] == 7'b0010011 &&  // ADDI opcode
        inst1[11:7] == inst2[11:7] && // rd matches
        inst1[11:7] == inst2[19:15])  // rs1 matches
    begin
        fuse_flag = 1;
        // Fuse into one li pseudo-instruction
        fused_inst = {inst1[31:12], inst2[31:20], inst2[11:7], 7'b0010011};
    end
end

endmodule
