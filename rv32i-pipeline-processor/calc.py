#!/usr/bin/env python3
"""
=============================================================================
FUSION VERIFICATION CALCULATOR (calc.py)
=============================================================================
This script verifies that the fusion detection logic works correctly by:
1. Parsing instruction memory (instr.mem) to find fusable sequences
2. Analyzing simulation logs to verify fusion actually triggered
3. Comparing expected vs actual fusion counts

Fusion Types:
  01 = LUI + ADDI   (Load 32-bit immediate)
  10 = AUIPC + JALR (Long jump / function call)
  11 = LOAD + ALU   (Load-use fusion)
=============================================================================
"""

import sys
import os
from pathlib import Path

# RISC-V Opcode definitions
OPCODES = {
    0b0110111: 'LUI',
    0b0010111: 'AUIPC',
    0b0010011: 'I-ALU',    # ADDI, SLTI, etc.
    0b0110011: 'R-ALU',    # ADD, SUB, etc.
    0b1100111: 'JALR',
    0b0000011: 'LOAD',     # LW, LH, LB, etc.
    0b0100011: 'STORE',    # SW, SH, SB
    0b1100011: 'BRANCH',   # BEQ, BNE, etc.
    0b1101111: 'JAL',
}

def decode_instruction(hex_str):
    """Decode a hex instruction string into its fields."""
    try:
        inst = int(hex_str.strip(), 16)
    except ValueError:
        return None
    
    opcode = inst & 0x7F
    rd = (inst >> 7) & 0x1F
    funct3 = (inst >> 12) & 0x7
    rs1 = (inst >> 15) & 0x1F
    rs2 = (inst >> 20) & 0x1F
    funct7 = (inst >> 25) & 0x7F
    
    op_name = OPCODES.get(opcode, f'UNK({opcode:07b})')
    
    return {
        'hex': hex_str.strip(),
        'raw': inst,
        'opcode': opcode,
        'op_name': op_name,
        'rd': rd,
        'rs1': rs1,
        'rs2': rs2,
        'funct3': funct3,
        'funct7': funct7
    }

def check_lui_addi(inst1, inst2):
    """Check if two instructions form a LUI+ADDI fusion pair."""
    if inst1['opcode'] != 0b0110111:  # LUI
        return False
    if inst2['opcode'] != 0b0010011:  # I-ALU
        return False
    if inst2['funct3'] != 0:  # ADDI specifically
        return False
    if inst1['rd'] != inst2['rd']:  # Same destination
        return False
    if inst1['rd'] != inst2['rs1']:  # ADDI uses LUI result
        return False
    if inst1['rd'] == 0:  # Not x0
        return False
    return True

def check_auipc_jalr(inst1, inst2):
    """Check if two instructions form an AUIPC+JALR fusion pair."""
    if inst1['opcode'] != 0b0010111:  # AUIPC
        return False
    if inst2['opcode'] != 0b1100111:  # JALR
        return False
    if inst1['rd'] != inst2['rs1']:  # JALR uses AUIPC result
        return False
    if inst1['rd'] == 0:  # Not x0
        return False
    return True

def check_load_alu(inst1, inst2):
    """Check if two instructions form a LOAD+ALU fusion pair."""
    if inst1['opcode'] != 0b0000011:  # LOAD
        return False
    
    # Second must be R-type or I-type ALU
    is_rtype = (inst2['opcode'] == 0b0110011)
    is_itype = (inst2['opcode'] == 0b0010011)
    
    if not (is_rtype or is_itype):
        return False
    
    # Load result must be used as rs1 or rs2
    rd1 = inst1['rd']
    if rd1 == 0:
        return False
    
    used_as_rs1 = (rd1 == inst2['rs1'])
    used_as_rs2 = is_rtype and (rd1 == inst2['rs2'])
    
    return used_as_rs1 or used_as_rs2

def analyze_instruction_memory(filepath):
    """Analyze instruction memory file and find all fusable sequences."""
    print(f"\n{'='*70}")
    print("INSTRUCTION MEMORY ANALYSIS")
    print(f"{'='*70}")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    instructions = []
    for i, line in enumerate(lines):
        inst = decode_instruction(line)
        if inst:
            inst['pc'] = i * 4  # Assuming 4-byte aligned
            inst['line'] = i + 1
            instructions.append(inst)
    
    print(f"Total instructions loaded: {len(instructions)}")
    print()
    
    # Find fusion opportunities
    fusion_opportunities = {
        'lui_addi': [],
        'auipc_jalr': [],
        'load_alu': []
    }
    
    for i in range(len(instructions) - 1):
        inst1 = instructions[i]
        inst2 = instructions[i + 1]
        
        if check_lui_addi(inst1, inst2):
            fusion_opportunities['lui_addi'].append({
                'pc1': inst1['pc'],
                'pc2': inst2['pc'],
                'inst1': inst1,
                'inst2': inst2
            })
        
        if check_auipc_jalr(inst1, inst2):
            fusion_opportunities['auipc_jalr'].append({
                'pc1': inst1['pc'],
                'pc2': inst2['pc'],
                'inst1': inst1,
                'inst2': inst2
            })
        
        if check_load_alu(inst1, inst2):
            fusion_opportunities['load_alu'].append({
                'pc1': inst1['pc'],
                'pc2': inst2['pc'],
                'inst1': inst1,
                'inst2': inst2
            })
    
    # Print results
    print("FUSION OPPORTUNITIES DETECTED:")
    print("-" * 70)
    
    print(f"\n[1] LUI + ADDI (Load 32-bit Immediate): {len(fusion_opportunities['lui_addi'])} pairs")
    for pair in fusion_opportunities['lui_addi']:
        print(f"    PC 0x{pair['pc1']:04x}: {pair['inst1']['hex']} ({pair['inst1']['op_name']} rd=x{pair['inst1']['rd']})")
        print(f"    PC 0x{pair['pc2']:04x}: {pair['inst2']['hex']} ({pair['inst2']['op_name']} rd=x{pair['inst2']['rd']} rs1=x{pair['inst2']['rs1']})")
        print()
    
    print(f"\n[2] AUIPC + JALR (Long Jump): {len(fusion_opportunities['auipc_jalr'])} pairs")
    for pair in fusion_opportunities['auipc_jalr']:
        print(f"    PC 0x{pair['pc1']:04x}: {pair['inst1']['hex']} ({pair['inst1']['op_name']} rd=x{pair['inst1']['rd']})")
        print(f"    PC 0x{pair['pc2']:04x}: {pair['inst2']['hex']} ({pair['inst2']['op_name']} rd=x{pair['inst2']['rd']} rs1=x{pair['inst2']['rs1']})")
        print()
    
    print(f"\n[3] LOAD + ALU (Load-Use): {len(fusion_opportunities['load_alu'])} pairs")
    for pair in fusion_opportunities['load_alu']:
        print(f"    PC 0x{pair['pc1']:04x}: {pair['inst1']['hex']} ({pair['inst1']['op_name']} rd=x{pair['inst1']['rd']})")
        print(f"    PC 0x{pair['pc2']:04x}: {pair['inst2']['hex']} ({pair['inst2']['op_name']} rd=x{pair['inst2']['rd']} rs1=x{pair['inst2']['rs1']} rs2=x{pair['inst2']['rs2']})")
        print()
    
    total = sum(len(v) for v in fusion_opportunities.values())
    print(f"\nTOTAL FUSION OPPORTUNITIES: {total}")
    
    return fusion_opportunities

def analyze_simulation_log(filepath, expected=None):
    """Analyze simulation log to count actual fusion triggers."""
    print(f"\n{'='*70}")
    print("SIMULATION LOG ANALYSIS")
    print(f"{'='*70}")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        print("ERROR: Log file is empty or missing header")
        return None
    
    header = lines[0].strip().split(',')
    print(f"Log columns: {header}")
    
    # Check if FuseType column exists
    has_fuse_type = 'FuseType' in header
    
    fusion_counts = {
        'lui_addi': 0,    # Type 01
        'auipc_jalr': 0,  # Type 10
        'load_alu': 0,    # Type 11
        'total': 0
    }
    
    fusion_events = []
    
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) < 5:
            continue
        
        try:
            time = int(parts[0])
            pc = int(parts[1], 16)
            inst = int(parts[2], 16)
            reset = parts[3]
            fuse_flag = parts[4]
            fuse_type = parts[5] if has_fuse_type and len(parts) > 5 else '00'
        except (ValueError, IndexError):
            continue
        
        if fuse_flag == '1':
            fusion_counts['total'] += 1
            
            event = {
                'time': time,
                'pc': pc,
                'inst': inst,
                'type': fuse_type
            }
            fusion_events.append(event)
            
            if fuse_type == '01':
                fusion_counts['lui_addi'] += 1
            elif fuse_type == '10':
                fusion_counts['auipc_jalr'] += 1
            elif fuse_type == '11':
                fusion_counts['load_alu'] += 1
    
    print(f"\nFUSION TRIGGERS DETECTED:")
    print("-" * 70)
    print(f"  LUI + ADDI:    {fusion_counts['lui_addi']}")
    print(f"  AUIPC + JALR:  {fusion_counts['auipc_jalr']}")
    print(f"  LOAD + ALU:    {fusion_counts['load_alu']}")
    print(f"  TOTAL:         {fusion_counts['total']}")
    
    # Show first few events
    print(f"\nFirst 10 fusion events:")
    for event in fusion_events[:10]:
        type_name = {'01': 'LUI+ADDI', '10': 'AUIPC+JALR', '11': 'LOAD+ALU'}.get(event['type'], 'UNKNOWN')
        print(f"  Time={event['time']:5d}ns  PC=0x{event['pc']:04x}  Type={type_name}")
    
    # Compare with expected if provided
    if expected:
        print(f"\n{'='*70}")
        print("VERIFICATION RESULTS")
        print(f"{'='*70}")
        
        exp_lui = len(expected.get('lui_addi', []))
        exp_auipc = len(expected.get('auipc_jalr', []))
        exp_load = len(expected.get('load_alu', []))
        
        # Note: In a loop, each opportunity may trigger multiple times
        print(f"  LUI+ADDI:    Expected at least {exp_lui}, Got {fusion_counts['lui_addi']}")
        print(f"  AUIPC+JALR:  Expected at least {exp_auipc}, Got {fusion_counts['auipc_jalr']}")
        print(f"  LOAD+ALU:    Expected at least {exp_load}, Got {fusion_counts['load_alu']}")
        
        all_ok = (fusion_counts['lui_addi'] >= exp_lui and 
                  fusion_counts['auipc_jalr'] >= exp_auipc and
                  fusion_counts['load_alu'] >= exp_load)
        
        if all_ok:
            print(f"\n[PASS] VERIFICATION PASSED: All fusion types detected correctly!")
        else:
            print(f"\n[FAIL] VERIFICATION FAILED: Some fusion types not detected as expected")
    
    return fusion_counts

def generate_test_instructions():
    """Generate test instructions for all three fusion types."""
    print(f"\n{'='*70}")
    print("TEST INSTRUCTION GENERATOR")
    print(f"{'='*70}")
    print("Generating test sequences for all fusion types...\n")
    
    test_program = []
    
    # Header comment
    test_program.append("// Test program for Macro-Op Fusion verification")
    test_program.append("// Generated by calc.py")
    test_program.append("")
    
    # 1. LUI + ADDI sequence (load 32-bit immediate into x1)
    # lui x1, 0x12345  -> 0x12345137
    # addi x1, x1, 0x678 -> 0x67808093
    test_program.append("// LUI + ADDI: Load 0x12345678 into x1")
    test_program.append("12345137  // lui x1, 0x12345")
    test_program.append("67808093  // addi x1, x1, 0x678")
    test_program.append("")
    
    # 2. AUIPC + JALR sequence (long jump to PC + offset)
    # auipc x2, 0x10  -> 0x00010117
    # jalr x1, x2, 0x100 -> 0x100100e7
    test_program.append("// AUIPC + JALR: Long jump (function call pattern)")
    test_program.append("00010117  // auipc x2, 0x10")
    test_program.append("100100e7  // jalr x1, x2, 0x100")
    test_program.append("")
    
    # 3. LOAD + ALU sequence (load then use immediately)
    # lw x3, 0(x1)     -> 0x0000a183
    # add x4, x3, x5   -> 0x00518233
    test_program.append("// LOAD + ALU: Load word and use immediately")
    test_program.append("0000a183  // lw x3, 0(x1)")
    test_program.append("00518233  // add x4, x3, x5")
    test_program.append("")
    
    # Print generated instructions
    print("Generated test sequences (hex):")
    print("-" * 40)
    for line in test_program:
        if not line.startswith("//") and line.strip():
            print(f"  {line}")
    
    # Also show assembly
    print("\nAssembly breakdown:")
    print("-" * 40)
    print("  [LUI+ADDI]    lui x1, 0x12345       -> 12345137")
    print("                addi x1, x1, 0x678    -> 67808093")
    print("")
    print("  [AUIPC+JALR]  auipc x2, 0x10        -> 00010117")
    print("                jalr x1, x2, 0x100    -> 100100e7")
    print("")
    print("  [LOAD+ALU]    lw x3, 0(x1)          -> 0000a183")
    print("                add x4, x3, x5        -> 00518233")
    
    return test_program

def main():
    print("=" * 70)
    print("       MACRO-OP FUSION VERIFICATION TOOL")
    print("=" * 70)
    
    # Get paths
    script_dir = Path(__file__).parent
    instr_mem_path = script_dir / "tb" / "instr.mem"
    sim_log_path = script_dir / "temp" / "simulation.log"
    
    # Analyze instruction memory
    expected = analyze_instruction_memory(instr_mem_path)
    
    # Analyze simulation log if it exists
    if sim_log_path.exists():
        analyze_simulation_log(sim_log_path, expected)
    else:
        print(f"\nNote: Simulation log not found at {sim_log_path}")
        print("Run the simulation first to generate the log file.")
    
    # Generate test instructions for reference
    generate_test_instructions()
    
    print(f"\n{'='*70}")
    print("VERIFICATION COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
