`timescale 1ns/1ps
module microprocessor_tb();
    reg clk;
    reg [31:0]instruction;
    reg rst;

 

    microprocessor u_microprocessor0
    (
        .clk(clk),
        .instruction(instruction),
        .rst(rst)
    );

    integer log_file;

    initial begin
        clk = 0;
        rst = 1;
        
        // Open a log file for the Python GUI to read
        log_file = $fopen("temp/simulation.log", "w");
        $fwrite(log_file, "Time,PC,Instruction,Reset,FuseFlag\n");

        rst = 1; // Start inactive
        #10;
        rst = 0; // Assert Reset (Active Low)
        #10;
        rst = 1; // Deassert Reset
        
        #5000; // Run for 5000ns
        
        $fclose(log_file);
        $finish;       
    end

    // Monitor block to write to the log file every cycle
    always @(posedge clk) begin
        if (rst) begin
            $fwrite(log_file, "%0d,%h,%h,%b,%b\n", $time, u_microprocessor0.u_core.pc_address, u_microprocessor0.u_core.instruction_decode, rst, u_microprocessor0.u_core.fuse_flag);
        end
    end

     initial begin
       $dumpfile("temp/microprocessor.vcd");
       $dumpvars(0,microprocessor_tb);
    end

    always begin
        #5 clk= ~clk;
    end
endmodule