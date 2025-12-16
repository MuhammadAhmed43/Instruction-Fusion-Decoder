module memory_stage(
    input wire rst,
    input wire load,
    input wire store,
    input wire [31:0] op_b,
    input wire [31:0] instruction,
    input wire [31:0] alu_out_address,
    input wire [31:0] wrap_load_in,
    input wire data_valid,
    input wire valid,

    output wire [3:0] mask,
    output wire we_re,
    output wire request,
    output wire [31:0] store_data_out,
    output wire [31:0] wrap_load_out
);

    wire [2:0] fun3;
    assign fun3 = instruction[14:12];
    
    wire [1:0] byteadd;
    assign byteadd = alu_out_address[1:0];

    wire mem_en;
    assign mem_en = load | store;

    assign we_re = store;
    assign request = mem_en;

    wrappermem u_wrappermem(
        .data_i(op_b),
        .byteadd(byteadd),
        .fun3(fun3),
        .mem_en(mem_en),
        .Load(load),
        .data_valid(data_valid),
        .wrap_load_in(wrap_load_in),
        .masking(mask),
        .data_o(store_data_out),
        .wrap_load_out(wrap_load_out)
    );

endmodule
