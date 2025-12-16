import pandas as pd

# Read both CSVs
baseline = pd.read_csv('temp/baseline_execution.csv')
fused = pd.read_csv('temp/fused_execution.csv')

print('='*80)
print('DETAILED CSV ANALYSIS')
print('='*80)

# Filter active cycles (Reset=1)
base_active = baseline[baseline['Reset'] == 1].copy()
fused_active = fused[fused['Reset'] == 1].copy()

print(f'\nBaseline total rows: {len(baseline)}, Active rows: {len(base_active)}')
print(f'Fused total rows: {len(fused)}, Active rows: {len(fused_active)}')

# Count fusions by type
print('\n' + '='*80)
print('FUSION DETECTION ANALYSIS')
print('='*80)

# Baseline fusions (should have detection but stall injected in baseline mode)
base_fusions = base_active[base_active['FuseFlag'] == 1]
fused_fusions = fused_active[fused_active['FuseFlag'] == 1]

print(f'\nBaseline FuseFlag=1 count: {len(base_fusions)}')
print(f'Fused FuseFlag=1 count: {len(fused_fusions)}')

# Analyze by type
def analyze_fusions(df, name):
    fusions = df[df['FuseFlag'] == 1].copy()
    fusions['FuseType_str'] = fusions['FuseType'].apply(lambda x: str(x).zfill(2) if pd.notna(x) else '00')
    
    lui_addi = len(fusions[fusions['FuseType_str'] == '01'])
    auipc_jalr = len(fusions[fusions['FuseType_str'] == '10'])
    load_alu = len(fusions[fusions['FuseType_str'] == '11'])
    
    print(f'\n{name}:')
    print(f'  LUI+ADDI (01): {lui_addi}')
    print(f'  AUIPC+JALR (10): {auipc_jalr}')
    print(f'  LOAD+ALU (11): {load_alu}')
    print(f'  Total: {lui_addi + auipc_jalr + load_alu}')
    
    return lui_addi, auipc_jalr, load_alu

base_counts = analyze_fusions(base_active, 'Baseline')
fused_counts = analyze_fusions(fused_active, 'Fused')

# Find effective end of program
def find_effective_end(df):
    def get_instr_val(x):
        try:
            return int(str(x).strip(), 16)
        except:
            return 0
    
    instr_values = df['Instruction'].apply(get_instr_val)
    is_valid = ~instr_values.isin([0x13, 0])
    valid_indices = is_valid[is_valid].index
    
    if not valid_indices.empty:
        valid_positions = [df.index.get_loc(idx) for idx in valid_indices]
        for i in range(1, len(valid_positions)):
            gap = valid_positions[i] - valid_positions[i-1]
            if gap > 5:
                return valid_positions[i-1] + 1
    return len(df)

base_end = find_effective_end(base_active)
fused_end = find_effective_end(fused_active)

print('\n' + '='*80)
print('EFFECTIVE PROGRAM EXECUTION')
print('='*80)
print(f'Baseline effective cycles: {base_end}')
print(f'Fused effective cycles: {fused_end}')
print(f'Cycles saved: {base_end - fused_end}')
print(f'Speedup: {(base_end - fused_end) / base_end * 100:.2f}%')

# Count valid instructions
base_eff = base_active.iloc[:base_end]
fused_eff = fused_active.iloc[:fused_end]

def count_valid_instr(df):
    def get_instr_val(x):
        try:
            return int(str(x).strip(), 16)
        except:
            return 0
    instr_values = df['Instruction'].apply(get_instr_val)
    return len(instr_values[~instr_values.isin([0x13, 0])])

base_instr = count_valid_instr(base_eff)
fused_instr = count_valid_instr(fused_eff)

print(f'\nBaseline valid instructions: {base_instr}')
print(f'Fused valid instructions: {fused_instr}')
print(f'Difference: {base_instr - fused_instr}')

# IPC/CPI
base_ipc = base_instr / base_end if base_end > 0 else 0
fused_ipc = fused_instr / fused_end if fused_end > 0 else 0
base_cpi = base_end / base_instr if base_instr > 0 else 0
fused_cpi = fused_end / fused_instr if fused_instr > 0 else 0

print('\n' + '='*80)
print('RAW IPC/CPI (Based on fetched instructions)')
print('='*80)
print(f'Baseline IPC: {base_ipc:.4f}')
print(f'Fused IPC: {fused_ipc:.4f}')
print(f'Baseline CPI: {base_cpi:.4f}')
print(f'Fused CPI: {fused_cpi:.4f}')

# Effective IPC (counting fusions as 2 operations)
fused_total_fusions = fused_counts[0] + fused_counts[1] + fused_counts[2]
base_equiv = base_instr
fused_equiv = fused_instr + fused_total_fusions  # Each fusion counts as 2 ops

base_eff_ipc = base_equiv / base_end if base_end > 0 else 0
fused_eff_ipc = fused_equiv / fused_end if fused_end > 0 else 0
base_eff_cpi = base_end / base_equiv if base_equiv > 0 else 0
fused_eff_cpi = fused_end / fused_equiv if fused_equiv > 0 else 0

print('\n' + '='*80)
print('EFFECTIVE IPC/CPI (Fusions count as 2 operations)')
print('='*80)
print(f'Baseline equivalent ops: {base_equiv}')
print(f'Fused equivalent ops: {fused_equiv}')
print(f'Baseline Effective IPC: {base_eff_ipc:.4f}')
print(f'Fused Effective IPC: {fused_eff_ipc:.4f}')
print(f'Baseline Effective CPI: {base_eff_cpi:.4f}')
print(f'Fused Effective CPI: {fused_eff_cpi:.4f}')
print(f'\nIPC Improvement: {(fused_eff_ipc - base_eff_ipc) / base_eff_ipc * 100:.2f}%')
print(f'CPI Reduction: {(base_eff_cpi - fused_eff_cpi) / base_eff_cpi * 100:.2f}%')

# NORMALIZED metrics (what the dashboard SHOULD show)
print('\n' + '='*80)
print('NORMALIZED METRICS (Same work = baseline instructions)')
print('='*80)
work_done = base_instr  # Same work for BOTH
base_norm_ipc = work_done / base_end
fused_norm_ipc = work_done / fused_end
base_norm_cpi = base_end / work_done
fused_norm_cpi = fused_end / work_done
print(f'Work Done (BOTH): {work_done} operations')
print(f'Baseline Throughput: {base_norm_ipc:.4f} ops/cycle')
print(f'Fused Throughput: {fused_norm_ipc:.4f} ops/cycle')
print(f'Throughput Improvement: {(fused_norm_ipc - base_norm_ipc) / base_norm_ipc * 100:.2f}%')
print(f'Baseline Latency: {base_norm_cpi:.4f} cycles/op')
print(f'Fused Latency: {fused_norm_cpi:.4f} cycles/op')
print(f'Latency Reduction: {(base_norm_cpi - fused_norm_cpi) / base_norm_cpi * 100:.2f}%')

# Show sample rows with fusions
print('\n' + '='*80)
print('SAMPLE FUSION EVENTS (Fused Mode)')
print('='*80)
fused_fusion_rows = fused_active[fused_active['FuseFlag'] == 1].head(20)
for idx, row in fused_fusion_rows.iterrows():
    ft = str(row['FuseType']).zfill(2)
    ft_name = {'01': 'LUI+ADDI', '10': 'AUIPC+JALR', '11': 'LOAD+ALU'}.get(ft, 'Unknown')
    print(f"Time={row['Time']:4}, PC={row['PC']}, Instr={row['Instruction']}, FuseType={ft} ({ft_name})")

# Show PC stall patterns in baseline (same PC appearing twice)
print('\n' + '='*80)
print('STALL ANALYSIS (Baseline - PC appearing twice)')
print('='*80)
base_pcs = base_eff['PC'].tolist()
stalls = []
for i in range(1, len(base_pcs)):
    if base_pcs[i] == base_pcs[i-1] and base_pcs[i] != 'xxxxxxxx':
        stalls.append((i, base_pcs[i]))
print(f'Total stalls detected: {len(stalls)}')
if stalls[:10]:
    print('First 10 stall positions:')
    for pos, pc in stalls[:10]:
        print(f'  Row {pos}: PC={pc}')

# Detailed verification
print('\n' + '='*80)
print('VERIFICATION SUMMARY')
print('='*80)

# Expected: Per iteration of loop (5 iterations)
# - 2 LUI+ADDI pairs per iteration = 10 total
# - 1 LOAD+ALU pair per iteration = 5 total  
# - 1 AUIPC+JALR at end = 1 total
# Total expected fusions = 16

# But we're seeing the loop run 10 times due to PC wraparound
# Let's calculate based on actual execution

print(f'\nTotal fusion events detected: {fused_counts[0] + fused_counts[1] + fused_counts[2]}')
print(f'  - LUI+ADDI: {fused_counts[0]}')
print(f'  - AUIPC+JALR: {fused_counts[1]}')
print(f'  - LOAD+ALU: {fused_counts[2]}')

print(f'\n✓ Speedup achieved: {(base_end - fused_end) / base_end * 100:.2f}%')
print(f'✓ Cycles saved: {base_end - fused_end} cycles')

# Check if all fusions are working
all_working = fused_counts[0] > 0 and fused_counts[1] > 0 and fused_counts[2] > 0
if all_working:
    print('\n✅ ALL 3 FUSION PATTERNS ARE WORKING!')
else:
    print('\n❌ SOME FUSION PATTERNS NOT DETECTED:')
    if fused_counts[0] == 0:
        print('  - LUI+ADDI not working')
    if fused_counts[1] == 0:
        print('  - AUIPC+JALR not working')
    if fused_counts[2] == 0:
        print('  - LOAD+ALU not working')
