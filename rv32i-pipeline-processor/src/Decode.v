module decode (
    input wire clk,
    input wire rst,
    input wire valid,
    input wire reg_write_en_in,
    input wire load_control_signal,
    input wire [31:0] instruction,
    input wire [31:0] pc_address,
    input wire [31:0] rd_wb_data,
    input wire [31:0] instruction_rd,
    
    // Forwarding Inputs
    input wire [31:0] alu_result_execute,
    input wire [4:0]  rd_execute,
    input wire        reg_write_execute,
    input wire [31:0] alu_result_mem,
    input wire [4:0]  rd_mem,
    input wire        reg_write_mem,

    output wire load,
    output wire store,
    output wire jalr,
    output wire next_sel,
    output wire branch_result,
    output wire reg_write_en_out,
    output wire [3:0]  alu_control,
    output wire [1:0]  mem_to_reg,
    output wire [4:0]  rs1 , rs2,
    output wire [31:0] opb_data,
    output wire [31:0] opa_mux_out,
    output wire [31:0] opb_mux_out,
    output wire operand_a_out,
    output wire operand_b_out
    );

    wire branch;
    wire operand_a;
    wire operand_b;
    wire [2:0]  imm_sel;
    wire [31:0] op_a , op_b;
    wire [31:0] imm_mux_out;
    wire [31:0] i_immo , s_immo , sb_immo , uj_immo , u_immo;
    
    reg [31:0] branch_op_a;
    reg [31:0] branch_op_b;

    // CONTROL UNIT
    controlunit u_cu0 
    (
        .opcode(instruction[6:0]),
        .fun3(instruction[14:12]),
        .fun7(instruction[30]),
        .valid(valid),
        .reg_write(reg_write_en_out),
        .imm_sel(imm_sel),
        .next_sel(next_sel),
        .operand_b(operand_b),
        .operand_a(operand_a),
        .mem_to_reg(mem_to_reg),
        .Load(load),
        .Store(store),
        .jalr_out(jalr),
        .Branch(branch),
        .load_control(load_control_signal),
        .alu_control(alu_control)
    );

    // IMMEDIATE GENERATION
    immediategen u_imm_gen0 (
        .instr(instruction),
        .i_imme(i_immo),
        .sb_imme(sb_immo),
        .s_imme(s_immo),
        .uj_imme(uj_immo),
        .u_imme(u_immo)
    );

    //IMMEDIATE SELECTION MUX
    mux3_8 u_mux0(
        .a(i_immo),
        .b(s_immo),
        .c(sb_immo),
        .d(uj_immo),
        .e(u_immo),
        .sel(imm_sel),
        .out(imm_mux_out)
    );

    // REGISTER FILE
    registerfile u_regfile0 
    (
        .clk(clk),
        .rst(rst),
        .en(reg_write_en_in),
        .rs1(instruction[19:15]),
        .rs2(instruction[24:20]),
        .rd(instruction_rd[11:7]),
        .data(rd_wb_data),
        .op_a(op_a),
        .op_b(op_b)
    );

    assign rs1 = instruction[19:15];
    assign rs2 = instruction[24:20];
    assign opb_data = op_b ;

    //SELECTION OF PROGRAM COUNTER OR OPERAND A
    mux u_mux1 
    (
        .a(op_a),
        .b(pc_address),
        .sel(operand_a),
        .out(opa_mux_out)
    );
    
    //SELECTION OF OPERAND B OR IMMEDIATE     
    mux u_mux2(
        .a(op_b),
        .b(imm_mux_out),
        .sel(operand_b),
        .out(opb_mux_out)
    );

    //BRANCH FORWARDING LOGIC
    always @(*) begin
        // Forwarding for Operand A (RS1)
        if (reg_write_execute && (rd_execute != 0) && (rd_execute == instruction[19:15])) begin
            branch_op_a = alu_result_execute;
        end else if (reg_write_mem && (rd_mem != 0) && (rd_mem == instruction[19:15])) begin
            branch_op_a = alu_result_mem;
        end else begin
            branch_op_a = op_a;
        end

        // Forwarding for Operand B (RS2)
        if (reg_write_execute && (rd_execute != 0) && (rd_execute == instruction[24:20])) begin
            branch_op_b = alu_result_execute;
        end else if (reg_write_mem && (rd_mem != 0) && (rd_mem == instruction[24:20])) begin
            branch_op_b = alu_result_mem;
        end else begin
            branch_op_b = op_b;
        end
    end

    //BRANCH
    branch u_branch0(
        .en(branch),
        .op_a(branch_op_a),
        .op_b(branch_op_b),
        .fun3(instruction[14:12]),
        .result(branch_result)
    );

    assign operand_a_out = operand_a;
    assign operand_b_out = operand_b;
endmodule
