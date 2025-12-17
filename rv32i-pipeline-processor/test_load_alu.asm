// Multi-Pattern Fusion Benchmark
// Tests: LUI+ADDI, LOAD+ALU with same destination

// Initialize some values in registers first
addi x1, x0, 100
addi x2, x0, 200
addi x3, x0, 50

// Store values to memory for loading
sw x1, 0(x0)
sw x2, 4(x0)
sw x3, 8(x0)
sw x1, 12(x0)
sw x2, 16(x0)

// Test #6: lw x9, add x9 (same destination)
lw x9, 0(x0)
add x9, x1, x2

// Test #7: lw x10, sub x10 (same destination)
lw x10, 4(x0)
sub x10, x1, x2

// Test #8: lw x11, and x11 (same destination)
lw x11, 8(x0)
and x11, x1, x2

// Test #9: lw x12, or x12 (same destination)
lw x12, 12(x0)
or x12, x1, x2

// Test #10: lw x14, xor x14 (same destination)
lw x14, 16(x0)
xor x14, x1, x2

// End with NOPs
nop
nop
nop
