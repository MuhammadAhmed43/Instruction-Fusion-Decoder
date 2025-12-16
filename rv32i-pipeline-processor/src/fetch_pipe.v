module fetch_pipe(
  input wire clk,
  input wire rst,
  input wire [31:0] pre_address_pc,
  input wire [31:0] instruction_fetch,
  input wire next_select,
  input wire branch_result,
  input wire jalr,
  input wire load,
  input wire fuse_flush,

  output wire [31:0] pre_address_out,
  output wire [31:0] instruction
);

  reg [31:0] pre_address, instruc;
  reg flush_pipeline , flush_pipeline2;

  always @ (posedge clk or negedge rst) begin
    if (!rst) begin
      pre_address     <= 32'b0;
      instruc         <= 32'b0;
      flush_pipeline  <= 0;
      flush_pipeline2 <= 0;
    end
    else begin
      if (next_select | branch_result | jalr) begin
        pre_address     <= 32'b0;
        instruc         <= 32'b0;
        flush_pipeline  <= 1;
      end 
      else if (flush_pipeline) begin
        pre_address     <= 32'b0;
        instruc         <= 32'b0;
        flush_pipeline  <= 0;
        flush_pipeline2 <= 1;
      end
      else if (flush_pipeline2) begin
        pre_address     <= 32'b0;
        instruc         <= 32'b0;
        flush_pipeline2 <= 0;
      end
      else if (fuse_flush) begin
        pre_address     <= 32'b0;
        instruc         <= 32'h00000013;
      end
      else if (load) begin
        pre_address     <= pre_address;
        instruc         <= instruc;
      end
      else begin
        pre_address     <= pre_address_pc;
        instruc         <= instruction_fetch;
      end
    end
  end

  assign pre_address_out = pre_address;
  assign instruction     = instruc;
endmodule
