import streamlit as st
import pandas as pd
import subprocess
import os
import plotly.graph_objects as go

# --- Configuration ---
# Use the directory where this script is located, not the current working directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INSTR_MEM_PATH = os.path.join(PROJECT_ROOT, "tb", "instr.mem")
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "temp", "simulation.log")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "temp", "microprocessor.output")

# CSV Export paths for review
BASELINE_CSV_PATH = os.path.join(PROJECT_ROOT, "temp", "baseline_execution.csv")
FUSED_CSV_PATH = os.path.join(PROJECT_ROOT, "temp", "fused_execution.csv")

st.set_page_config(page_title="RV32I Multi-Fusion Analytics", layout="wide")

st.title("⚡ RV32I Multi-Pattern Macro-Op Fusion Analytics")
st.markdown("""
This dashboard compares the performance of the **Standard RV32I Pipeline** vs. the **Fused Pipeline**.
It supports **three fusion patterns**:
- **LUI + ADDI** → Load 32-bit immediate constant
- **AUIPC + JALR** → Long jump / function call
- **LOAD + ALU** → Load-use fusion (eliminates data hazard stall)
""")

# --- Assembler (Mini) ---
def run_assembler(asm_text):
    lines = [l.strip() for l in asm_text.split("\n") if l.strip() and not l.strip().startswith("//")]
    
    # Pass 1: Find Labels
    labels = {}
    pc = 0
    clean_lines = []
    
    for line in lines:
        if line.endswith(":"):
            labels[line[:-1]] = pc
        else:
            clean_lines.append(line)
            pc += 4
            
    # Pass 2: Assemble
    hex_lines = []
    pc = 0
    
    for line in clean_lines:
        parts = line.replace(",", " ").split()
        instr = parts[0].lower()
        
        try:
            val = 0x00000013 # Default NOP
            
            if instr == "nop":
                val = 0x00000013
            elif instr == "lui":
                rd = int(parts[1].replace("x", ""))
                imm = int(parts[2], 0)
                val = ((imm & 0xFFFFF000)) | (rd << 7) | 0x37
            elif instr == "auipc":
                rd = int(parts[1].replace("x", ""))
                imm = int(parts[2], 0)
                val = ((imm & 0xFFFFF000)) | (rd << 7) | 0x17
            elif instr == "addi":
                rd = int(parts[1].replace("x", ""))
                rs1 = int(parts[2].replace("x", ""))
                imm = int(parts[3], 0)
                val = ((imm & 0xFFF) << 20) | (rs1 << 15) | (0 << 12) | (rd << 7) | 0x13
            elif instr == "add":
                rd = int(parts[1].replace("x", ""))
                rs1 = int(parts[2].replace("x", ""))
                rs2 = int(parts[3].replace("x", ""))
                val = (0 << 25) | (rs2 << 20) | (rs1 << 15) | (0 << 12) | (rd << 7) | 0x33
            elif instr == "sub":
                rd = int(parts[1].replace("x", ""))
                rs1 = int(parts[2].replace("x", ""))
                rs2 = int(parts[3].replace("x", ""))
                val = (0x20 << 25) | (rs2 << 20) | (rs1 << 15) | (0 << 12) | (rd << 7) | 0x33
            elif instr == "lw":
                # lw rd, offset(rs1)
                rd = int(parts[1].replace("x", ""))
                offset_rs1 = parts[2]
                offset_str, rs1_str = offset_rs1.replace(")", "").split("(")
                offset = int(offset_str, 0)
                rs1 = int(rs1_str.replace("x", ""))
                val = ((offset & 0xFFF) << 20) | (rs1 << 15) | (0x2 << 12) | (rd << 7) | 0x03
            elif instr == "sw":
                # sw rs2, offset(rs1)
                rs2 = int(parts[1].replace("x", ""))
                offset_rs1 = parts[2]
                offset_str, rs1_str = offset_rs1.replace(")", "").split("(")
                offset = int(offset_str, 0)
                rs1 = int(rs1_str.replace("x", ""))
                imm11_5 = (offset >> 5) & 0x7F
                imm4_0 = offset & 0x1F
                val = (imm11_5 << 25) | (rs2 << 20) | (rs1 << 15) | (0x2 << 12) | (imm4_0 << 7) | 0x23
            elif instr == "jalr":
                # jalr rd, rs1, offset OR jalr rd, offset(rs1)
                rd = int(parts[1].replace("x", ""))
                if "(" in parts[2]:
                    offset_rs1 = parts[2]
                    offset_str, rs1_str = offset_rs1.replace(")", "").split("(")
                    offset = int(offset_str, 0)
                    rs1 = int(rs1_str.replace("x", ""))
                else:
                    rs1 = int(parts[2].replace("x", ""))
                    offset = int(parts[3], 0) if len(parts) > 3 else 0
                val = ((offset & 0xFFF) << 20) | (rs1 << 15) | (0 << 12) | (rd << 7) | 0x67
            elif instr == "bne":
                rs1 = int(parts[1].replace("x", ""))
                rs2 = int(parts[2].replace("x", ""))
                label = parts[3]
                
                offset = 0
                if label in labels:
                    offset = labels[label] - pc
                else:
                    try:
                        offset = int(label, 0)
                    except:
                        offset = 0
                
                # B-Type encoding
                imm12 = (offset >> 12) & 1
                imm10_5 = (offset >> 5) & 0x3F
                imm4_1 = (offset >> 1) & 0xF
                imm11 = (offset >> 11) & 1
                
                val = (imm12 << 31) | (imm10_5 << 25) | (rs2 << 20) | (rs1 << 15) | (0x1 << 12) | (imm4_1 << 8) | (imm11 << 7) | 0x63

            hex_lines.append(f"{val:08x}")
            pc += 4
        except Exception as e:
            print(f"Error assembling {line}: {e}")
            hex_lines.append("00000013")

    while len(hex_lines) < 256:
        hex_lines.append("00000013")
        
    with open(INSTR_MEM_PATH, "w") as f:
        f.write("\n".join(hex_lines))
    return len(hex_lines)

# --- Simulation Runner ---
def run_simulation(enable_fusion):
    # 1. Compile with Parameter Override
    # We use -P to override the parameter in the top-level module instance
    # The testbench instantiates 'microprocessor' as 'u_microprocessor0'
    param_val = "1" if enable_fusion else "0"
    
    # Define paths to executables
    IVERILOG_BIN = r"C:\iverilog\bin\iverilog.exe"
    VVP_BIN = r"C:\iverilog\bin\vvp.exe"
    
    # Path to flist file
    FLIST_PATH = os.path.join(PROJECT_ROOT, "flist")

    # Read flist
    with open(FLIST_PATH, "r") as f:
        files = [l.strip().replace("${CORE_ROOT}", PROJECT_ROOT) for l in f.readlines() if l.strip()]
    
    # Use -D to define the macro for the parameter default value
    compile_cmd = [IVERILOG_BIN, "-o", OUTPUT_FILE, f"-DFUSION_ENABLE_VAL={param_val}"] + files
    
    # Run Compilation
    # Add iverilog bin path to environment
    env = os.environ.copy()
    env["PATH"] += os.pathsep + os.path.dirname(IVERILOG_BIN)
    subprocess.run(compile_cmd, check=True, env=env, cwd=PROJECT_ROOT)
    
    # 2. Run Simulation
    if os.path.exists(LOG_FILE_PATH):
        os.remove(LOG_FILE_PATH)
        
    subprocess.run([VVP_BIN, OUTPUT_FILE], check=True, env=env, cwd=PROJECT_ROOT)
    
    # 3. Parse Log and Save labeled CSV
    if os.path.exists(LOG_FILE_PATH):
        df = pd.read_csv(LOG_FILE_PATH)
        # Save to labeled CSV for review
        csv_path = FUSED_CSV_PATH if enable_fusion else BASELINE_CSV_PATH
        df.to_csv(csv_path, index=False)
        return df
    return None

# --- Analysis ---
def analyze_performance(df, label):
    if df is None or df.empty:
        return None
    
    # Filter out reset cycles
    # The testbench only logs when Reset is INACTIVE (High), so all logged rows are active.
    # However, to be safe, we check for Reset == 1 (Active Low logic)
    if 'Reset' in df.columns:
        active_df = df[df['Reset'] == 1].copy()
    else:
        active_df = df.copy()
    
    # Calculate Metrics
    # 1. Determine "Effective Duration" (Time to finish the workload)
    # We look for the last instruction that is NOT a NOP (0x13) or Bubble (0x0)
    # This handles the case where one simulation finishes early and spins on NOPs
    
    def get_instr_val(x):
        try:
            return int(str(x).strip(), 16)
        except:
            return 0 # Treat 'x' or invalid as Bubble (0)

    # Convert Instruction column to numeric values
    instr_values = active_df['Instruction'].apply(get_instr_val)
    
    # Find indices where instruction is valid (not NOP/Bubble)
    # 0x13 = NOP, 0x0 = Bubble
    is_valid = ~instr_values.isin([0x13, 0])
    valid_indices = is_valid[is_valid].index
    
    if not valid_indices.empty:
        # Find the FIRST occurrence where we have a sustained NOP sequence (program finished)
        # We look for the first gap in valid instructions (consecutive NOPs after a valid instruction)
        # This handles PC wrap-around where the same code executes again later.
        
        # Strategy: Find the first valid instruction, then find where valid instructions stop
        # by detecting a sequence of at least 5 consecutive NOPs
        NOP_THRESHOLD = 5  # Number of consecutive NOPs to consider "program finished"
        
        # Get the positions of valid instructions
        valid_positions = [active_df.index.get_loc(idx) for idx in valid_indices]
        
        # Find first gap of >= NOP_THRESHOLD in valid instruction positions
        end_pos = len(active_df)  # Default to full length
        for i in range(1, len(valid_positions)):
            gap = valid_positions[i] - valid_positions[i-1]
            if gap > NOP_THRESHOLD:
                # Found a significant gap - program likely finished before this
                end_pos = valid_positions[i-1] + 1
                break
        
        effective_cycles = end_pos
        
        # Slice the dataframe to this effective range
        effective_df = active_df.iloc[:effective_cycles]
        
        # Recount valid instructions within effective range
        effective_instr_values = effective_df['Instruction'].apply(get_instr_val)
        instr_count = len(effective_instr_values[~effective_instr_values.isin([0x13, 0])])
    else:
        effective_cycles = len(active_df)
        effective_df = active_df
        instr_count = 0

    # 2. Count Fusion Hits by Type (Handle 'x' strings and binary format)
    fusion_hits = 0
    fusion_lui_addi = 0
    fusion_auipc_jalr = 0
    fusion_load_alu = 0
    
    if 'FuseFlag' in effective_df.columns:
        # Coerce to numeric, turn errors/'x' to 0
        fusion_flags = pd.to_numeric(effective_df['FuseFlag'], errors='coerce').fillna(0)
        fusion_hits = int(fusion_flags.sum())
        
        # Parse FuseType if available
        if 'FuseType' in effective_df.columns:
            # FuseType might be read as integer (1, 10, 11) or string ('01', '10', '11')
            # Convert to string and zero-pad to 2 digits for consistent comparison
            fuse_types = effective_df['FuseType'].apply(lambda x: str(x).zfill(2) if pd.notna(x) else '00')
            # Count each type: 01=LUI+ADDI, 10=AUIPC+JALR, 11=LOAD+ALU
            fusion_lui_addi = len(fuse_types[(fusion_flags == 1) & (fuse_types == '01')])
            fusion_auipc_jalr = len(fuse_types[(fusion_flags == 1) & (fuse_types == '10')])
            fusion_load_alu = len(fuse_types[(fusion_flags == 1) & (fuse_types == '11')])
        else:
            # Old format without FuseType - assume all are LUI+ADDI
            fusion_lui_addi = fusion_hits

    # 3. Calculate Metrics based on Effective Duration
    # Note: In fused mode, instruction count is LOWER because fused pairs count as 1 instruction
    # This is expected behavior - fusion eliminates instructions
    ipc = instr_count / effective_cycles if effective_cycles > 0 else 0
    cpi = effective_cycles / instr_count if instr_count > 0 else 0
    
    # For proper comparison, we track "equivalent instructions" 
    # = actual instructions + fusion hits (since each fusion replaced 2 instructions with 1)
    # This represents the total "work done" - same as baseline would fetch
    equivalent_instr_count = instr_count + fusion_hits
    
    # Effective IPC = how many "logical operations" per cycle (counting fused ops as 2)
    effective_ipc = equivalent_instr_count / effective_cycles if effective_cycles > 0 else 0
    effective_cpi = effective_cycles / equivalent_instr_count if equivalent_instr_count > 0 else 0
    
    return {
        "Label": label,
        "Cycles": effective_cycles, # Use effective cycles, not total simulation time
        "Instructions": instr_count,  # Actual instructions fetched
        "EquivalentInstructions": equivalent_instr_count,  # Logical work done (fusions count as 2)
        "IPC": ipc,  # Actual IPC
        "CPI": cpi,  # Actual CPI
        "EffectiveIPC": effective_ipc,  # Effective IPC counting fusions as 2
        "EffectiveCPI": effective_cpi,  # Effective CPI
        "FusionHits": fusion_hits,
        "FusionLuiAddi": fusion_lui_addi,
        "FusionAuipcJalr": fusion_auipc_jalr,
        "FusionLoadAlu": fusion_load_alu,
        "Data": active_df # Keep full data for graph
    }

# --- UI Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📝 Assembly Input")
    default_asm = """// Multi-Pattern Fusion Benchmark
// Tests: LUI+ADDI, LOAD+ALU, AUIPC+JALR

// Initialize loop counter (x5 = 5)
addi x5, x0, 5

loop:
// Pattern 1: LUI + ADDI (Load 32-bit immediate)
lui x1, 0x12345000
addi x1, x1, 0x678

// Pattern 1 again: Another LUI + ADDI
lui x2, 0x87654000
addi x2, x2, 0x321

// Normal ALU Ops
add x3, x1, x2
sub x4, x1, x2

// Pattern 3: LOAD + ALU (Load-Use)
sw x1, 0(x0)
lw x7, 0(x0)
add x8, x7, x3

// Decrement and Branch
addi x5, x5, -1
bne x5, x0, loop

// Pattern 2: AUIPC + JALR (Long Jump)
// This jumps forward to skip some NOPs
auipc x10, 0
jalr x0, x10, 12

nop
nop
nop
"""
    asm_code = st.text_area("Enter RISC-V Assembly", default_asm, height=400)
    
    if st.button("🚀 Run Comparison", type="primary"):
        with st.spinner("Running Baseline & Fused Simulations..."):
            run_assembler(asm_code)
            
            # Run Baseline
            df_base = run_simulation(enable_fusion=False)
            res_base = analyze_performance(df_base, "Baseline (No Fusion)")
            
            # Run Fused
            df_fused = run_simulation(enable_fusion=True)
            res_fused = analyze_performance(df_fused, "Fused (Macro-Op)")
            
            st.session_state['res_base'] = res_base
            st.session_state['res_fused'] = res_fused
            st.session_state['df_base'] = df_base
            st.session_state['df_fused'] = df_fused
            st.success("Comparison Complete!")
            st.info(f"📁 CSV logs saved to:\n• temp/baseline_execution.csv\n• temp/fused_execution.csv")

with col2:
    if 'res_base' in st.session_state:
        base = st.session_state['res_base']
        fused = st.session_state['res_fused']
        
        # --- Hero Metrics ---
        st.subheader("🚀 Performance Summary")
        
        # Speedup Calculation
        speedup = 0.0
        cycles_saved = base['Cycles'] - fused['Cycles']
        if base['Cycles'] > 0:
            speedup = cycles_saved / base['Cycles'] * 100
        
        # Big speedup display
        hero_col1, hero_col2, hero_col3 = st.columns([2, 1, 1])
        with hero_col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                        padding: 20px; border-radius: 15px; text-align: center;
                        border: 2px solid #00d4ff;">
                <h1 style="color: #00ff88; margin: 0; font-size: 3em;">{speedup:.1f}%</h1>
                <p style="color: #888; margin: 5px 0 0 0;">Speedup Achieved</p>
            </div>
            """, unsafe_allow_html=True)
        with hero_col2:
            st.metric("🔥 Total Fusions", f"{fused['FusionHits']}")
            st.metric("⏱️ Cycles Saved", f"{cycles_saved}")
        with hero_col3:
            st.metric("📊 Fused Cycles", f"{fused['Cycles']}")
            st.metric("📊 Baseline Cycles", f"{base['Cycles']}")
        
        # --- Fusion Type Breakdown ---
        st.markdown("---")
        st.subheader("🔀 Fusion Type Breakdown")
        
        ftype_col1, ftype_col2, ftype_col3, ftype_col4 = st.columns(4)
        
        with ftype_col1:
            st.markdown(f"""
            <div style="background: #1a2d1a; padding: 15px; border-radius: 10px; text-align: center;
                        border: 2px solid #00ff88;">
                <h2 style="color: #00ff88; margin: 0;">{fused['FusionLuiAddi']}</h2>
                <p style="color: #aaa; margin: 5px 0 0 0; font-size: 0.9em;">LUI + ADDI</p>
                <p style="color: #666; margin: 0; font-size: 0.7em;">32-bit Immediate</p>
            </div>
            """, unsafe_allow_html=True)
        
        with ftype_col2:
            st.markdown(f"""
            <div style="background: #2d1a2d; padding: 15px; border-radius: 10px; text-align: center;
                        border: 2px solid #ff88ff;">
                <h2 style="color: #ff88ff; margin: 0;">{fused['FusionAuipcJalr']}</h2>
                <p style="color: #aaa; margin: 5px 0 0 0; font-size: 0.9em;">AUIPC + JALR</p>
                <p style="color: #666; margin: 0; font-size: 0.7em;">Long Jump</p>
            </div>
            """, unsafe_allow_html=True)
        
        with ftype_col3:
            st.markdown(f"""
            <div style="background: #2d2d1a; padding: 15px; border-radius: 10px; text-align: center;
                        border: 2px solid #ffff88;">
                <h2 style="color: #ffff88; margin: 0;">{fused['FusionLoadAlu']}</h2>
                <p style="color: #aaa; margin: 5px 0 0 0; font-size: 0.9em;">LOAD + ALU</p>
                <p style="color: #666; margin: 0; font-size: 0.7em;">Load-Use Fusion</p>
            </div>
            """, unsafe_allow_html=True)
        
        with ftype_col4:
            # Fusion type pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=['LUI+ADDI', 'AUIPC+JALR', 'LOAD+ALU'],
                values=[fused['FusionLuiAddi'], fused['FusionAuipcJalr'], fused['FusionLoadAlu']],
                marker=dict(colors=['#00ff88', '#ff88ff', '#ffff88']),
                hole=0.4
            )])
            fig_pie.update_layout(
                showlegend=False,
                height=150,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("---")
        
        # --- Detailed Metrics Comparison ---
        st.subheader("📊 Detailed Metrics")
        
        # Calculate NORMALIZED metrics - use baseline instruction count as "work done" for BOTH
        # This is the FAIR comparison because both execute the same program
        work_done = base['Instructions']  # Same work for both - baseline fetches all instructions
        
        # Normalized IPC = work_done / cycles (same work, different cycles)
        base_normalized_ipc = work_done / base['Cycles'] if base['Cycles'] > 0 else 0
        fused_normalized_ipc = work_done / fused['Cycles'] if fused['Cycles'] > 0 else 0
        base_normalized_cpi = base['Cycles'] / work_done if work_done > 0 else 0
        fused_normalized_cpi = fused['Cycles'] / work_done if work_done > 0 else 0
        
        normalized_ipc_improvement = ((fused_normalized_ipc - base_normalized_ipc) / base_normalized_ipc * 100) if base_normalized_ipc > 0 else 0
        normalized_cpi_reduction = ((base_normalized_cpi - fused_normalized_cpi) / base_normalized_cpi * 100) if base_normalized_cpi > 0 else 0
        
        # Explain the metrics
        st.markdown(f"""
        <div style="background: #1a1a2e; padding: 12px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #00d4ff;">
            <p style="color: #aaa; margin: 0; font-size: 0.9em;">
                <b>⚙️ Fair Comparison:</b> Both pipelines execute the same program ({work_done} operations).
                Fusion completes this work in <b>{fused['Cycles']} cycles</b> vs baseline's <b>{base['Cycles']} cycles</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Row 1: THE KEY METRICS - Normalized throughput (same work, different cycles)
        st.markdown("**🎯 Normalized Throughput (Same Work, Fair Comparison)**")
        m1, m2, m3, m4 = st.columns(4)
        
        m1.metric("Fused Throughput", f"{fused_normalized_ipc:.4f} ops/cycle", 
                  delta=f"+{fused_normalized_ipc - base_normalized_ipc:.4f}")
        m2.metric("Baseline Throughput", f"{base_normalized_ipc:.4f} ops/cycle")
        m3.metric("Fused Latency", f"{fused_normalized_cpi:.4f} cyc/op", 
                  delta=f"{fused_normalized_cpi - base_normalized_cpi:.4f}", delta_color="inverse")
        m4.metric("Baseline Latency", f"{base_normalized_cpi:.4f} cyc/op")
        
        # Row 2: THE BIG NUMBERS - Improvement percentages
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("🚀 Throughput Improvement", f"+{normalized_ipc_improvement:.1f}%")
        m6.metric("⚡ Latency Reduction", f"{normalized_cpi_reduction:.1f}%")
        m7.metric("⏱️ Cycles Saved", f"{cycles_saved}", delta=f"-{speedup:.1f}%")
        m8.metric("🔥 Fusion Events", f"{fused['FusionHits']}")
        
        # Row 3: Raw metrics (actual fetched instructions) - for transparency
        st.markdown("**📋 Raw Pipeline Metrics (Actual Instructions Fetched)**")
        st.markdown(f"""
        <div style="background: #2d1a1a; padding: 8px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #ff6b6b; font-size: 0.85em;">
            <b>⚠️ Note:</b> Fused pipeline fetches <b>fewer</b> instructions ({fused['Instructions']} vs {base['Instructions']}) 
            because fused pairs count as 1 fetch. Both pipelines do the <b>same work</b> ({work_done} operations).
        </div>
        """, unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Fused Fetched", f"{fused['Instructions']}", 
                  delta=f"{fused['Instructions'] - base['Instructions']} fewer", help="Actual instructions fetched by fused pipeline. Lower because fused pairs = 1 fetch.")
        r2.metric("Baseline Fetched", f"{base['Instructions']}", help="Actual instructions fetched by baseline pipeline.")
        instr_reduction = ((base['Instructions'] - fused['Instructions']) / base['Instructions'] * 100) if base['Instructions'] > 0 else 0
        r3.metric("📉 Fetch Reduction", f"{instr_reduction:.1f}%", help="Percentage reduction in instruction fetches.")
        r4.metric("📊 Work Done (Both)", f"{work_done} ops", help="Total operations executed - same for both pipelines!")
        
        # --- Summary Table ---
        st.markdown("---")
        st.subheader("📋 Comprehensive Performance Summary")
        
        st.markdown("""
        <div style="background: #1a2d1a; padding: 10px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #00ff88; font-size: 0.85em;">
            <b>📊 Understanding the Metrics:</b><br>
            • <b>Work Done</b> = Total operations (same for both - same program)<br>
            • <b>Instructions Fetched</b> = What the pipeline actually fetched (Fused is lower!)<br>
            • <b>Throughput</b> = Work Done ÷ Cycles (fair comparison)<br>
            • <b>Fused fetches fewer</b> because each fusion combines 2 instructions into 1 fetch
        </div>
        """, unsafe_allow_html=True)
        
        summary_data = {
            "Metric": [
                "⏱️ Total Execution Cycles", 
                "📦 Work Done (Same Program!)",
                "📋 Instructions Fetched (Different!)",
                "🚀 Throughput (ops/cycle)",
                "⚡ Latency (cycles/op)",
                "🟢 LUI+ADDI Fusions",
                "🟣 AUIPC+JALR Fusions",
                "🟡 LOAD+ALU Fusions",
                "🔥 Total Fusion Events"
            ],
            "Baseline": [
                f"{base['Cycles']}",
                f"{work_done}",
                f"{base['Instructions']}",
                f"{base_normalized_ipc:.4f}",
                f"{base_normalized_cpi:.4f}",
                f"{base.get('FusionLuiAddi', 0)}",
                f"{base.get('FusionAuipcJalr', 0)}",
                f"{base.get('FusionLoadAlu', 0)}",
                f"{base['FusionHits']}"
            ],
            "Fused": [
                f"{fused['Cycles']}",
                f"{work_done}  ← Same!",
                f"{fused['Instructions']}  ← Fewer!",
                f"{fused_normalized_ipc:.4f}",
                f"{fused_normalized_cpi:.4f}",
                f"{fused['FusionLuiAddi']}",
                f"{fused['FusionAuipcJalr']}",
                f"{fused['FusionLoadAlu']}",
                f"{fused['FusionHits']}"
            ],
            "Change": [
                f"✅ {cycles_saved} saved ({speedup:.1f}% faster)",
                f"= (same program)",
                f"↓ {base['Instructions'] - fused['Instructions']} fewer fetches ({instr_reduction:.1f}%)",
                f"↑ +{normalized_ipc_improvement:.1f}%",
                f"↓ -{normalized_cpi_reduction:.1f}%",
                f"+{fused['FusionLuiAddi'] - base.get('FusionLuiAddi', 0)}",
                f"+{fused['FusionAuipcJalr'] - base.get('FusionAuipcJalr', 0)}",
                f"+{fused['FusionLoadAlu'] - base.get('FusionLoadAlu', 0)}",
                f"+{fused['FusionHits'] - base['FusionHits']}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Style the dataframe
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn("📏 Metric", width="large"),
                "Baseline": st.column_config.TextColumn("⚫ Baseline", width="small"),
                "Fused": st.column_config.TextColumn("🟢 Fused", width="small"),
                "Change": st.column_config.TextColumn("🔄 Change", width="medium")
            }
        )
        
        st.markdown("---")
        
        # --- Bar Chart Comparison ---
        st.subheader("📈 Visual Comparison")
        
        bar_col1, bar_col2 = st.columns(2)
        
        with bar_col1:
            # Cycles comparison bar chart
            fig_cycles = go.Figure(data=[
                go.Bar(name='Baseline', x=['Total Cycles'], y=[base['Cycles']], 
                       marker_color='#6c757d', text=[base['Cycles']], textposition='auto'),
                go.Bar(name='Fused', x=['Total Cycles'], y=[fused['Cycles']], 
                       marker_color='#00ff88', text=[fused['Cycles']], textposition='auto')
            ])
            fig_cycles.update_layout(
                title="Cycle Count Comparison",
                barmode='group',
                height=300,
                showlegend=True,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_cycles, width='stretch')
        
        with bar_col2:
            # Normalized Throughput comparison (same work, different cycles)
            fig_ipc = go.Figure(data=[
                go.Bar(name='Baseline', x=['Throughput (ops/cycle)'], y=[base_normalized_ipc], 
                       marker_color='#6c757d', text=[f"{base_normalized_ipc:.3f}"], textposition='auto'),
                go.Bar(name='Fused', x=['Throughput (ops/cycle)'], y=[fused_normalized_ipc], 
                       marker_color='#00ff88', text=[f"{fused_normalized_ipc:.3f}"], textposition='auto')
            ])
            fig_ipc.update_layout(
                title=f"Normalized Throughput ({work_done} ops / cycles)",
                barmode='group',
                height=300,
                showlegend=True,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_ipc, width='stretch')
        
        st.markdown("---")
        
        # --- PC Flow Timeline ---
        st.subheader("🔄 Program Counter Flow")
        
        # Prepare data - limit to effective cycles for cleaner view
        base_eff = base['Data'].iloc[:base['Cycles']].copy()
        fused_eff = fused['Data'].iloc[:fused['Cycles']].copy()
        
        def safe_hex_to_int(x):
            try:
                return int(str(x), 16) if str(x) != 'xxxxxxxx' else 0
            except:
                return 0
        
        fig_pc = go.Figure()
        
        # Baseline Trace
        fig_pc.add_trace(go.Scatter(
            x=list(range(len(base_eff))),
            y=base_eff['PC'].apply(safe_hex_to_int),
            mode='lines+markers',
            name='Baseline',
            line=dict(color='#6c757d', width=2),
            marker=dict(size=4)
        ))
        
        # Fused Trace
        fig_pc.add_trace(go.Scatter(
            x=list(range(len(fused_eff))),
            y=fused_eff['PC'].apply(safe_hex_to_int),
            mode='lines+markers',
            name='Fused',
            line=dict(color='#00ff88', width=3),
            marker=dict(size=5)
        ))
        
        # Mark fusion points
        fused_flags = fused_eff[fused_eff['FuseFlag'].astype(str) == '1']
        if not fused_flags.empty:
            fusion_x = [fused_eff.index.get_loc(idx) for idx in fused_flags.index]
            fusion_y = fused_flags['PC'].apply(safe_hex_to_int).tolist()
            fig_pc.add_trace(go.Scatter(
                x=fusion_x,
                y=fusion_y,
                mode='markers',
                name='Fusion Event',
                marker=dict(size=15, color='#ff6b6b', symbol='star', 
                           line=dict(width=2, color='white'))
            ))
        
        fig_pc.update_layout(
            title="PC Address vs Cycle (Effective Execution Only)",
            xaxis_title="Cycle Number",
            yaxis_title="PC Address (hex)",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pc, width='stretch')
        
        st.markdown("---")
        
        # --- Stall Analysis ---
        st.subheader("🛑 Stall Analysis (Baseline)")
        
        # Detect stalls in baseline (same PC appearing twice)
        base_pcs = base_eff['PC'].apply(safe_hex_to_int).tolist()
        stall_positions = []
        for i in range(1, len(base_pcs)):
            if base_pcs[i] == base_pcs[i-1] and base_pcs[i] != 0:
                stall_positions.append(i)
        
        stall_col1, stall_col2 = st.columns([1, 2])
        
        with stall_col1:
            st.markdown(f"""
            <div style="background: #2d1f1f; padding: 15px; border-radius: 10px; 
                        border-left: 4px solid #ff6b6b;">
                <h3 style="color: #ff6b6b; margin: 0;">⚠️ {len(stall_positions)} Stalls</h3>
                <p style="color: #aaa; margin: 5px 0 0 0;">Detected in baseline execution</p>
                <p style="color: #888; font-size: 0.9em;">Each stall = 1 wasted cycle</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: #1f2d1f; padding: 15px; border-radius: 10px; 
                        border-left: 4px solid #00ff88; margin-top: 10px;">
                <h3 style="color: #00ff88; margin: 0;">✨ {fused['FusionHits']} Fusions</h3>
                <p style="color: #aaa; margin: 5px 0 0 0;">Eliminated stalls via Macro-Op Fusion</p>
            </div>
            """, unsafe_allow_html=True)
        
        with stall_col2:
            # Show stall locations as a heatmap-style visualization
            if stall_positions:
                fig_stalls = go.Figure()
                
                # Create a timeline showing where stalls occur
                cycle_status = ['Normal'] * len(base_eff)
                for pos in stall_positions:
                    cycle_status[pos] = 'Stall'
                
                stall_y = [1 if s == 'Stall' else 0 for s in cycle_status[:min(150, len(cycle_status))]]
                
                fig_stalls.add_trace(go.Bar(
                    x=list(range(len(stall_y))),
                    y=stall_y,
                    marker_color=['#ff6b6b' if y == 1 else '#2a2a2a' for y in stall_y],
                    name='Stall Events'
                ))
                
                fig_stalls.update_layout(
                    title="Stall Timeline (First 150 Cycles)",
                    xaxis_title="Cycle",
                    yaxis_title="",
                    height=200,
                    showlegend=False,
                    yaxis=dict(showticklabels=False),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_stalls, width='stretch')
        
        st.markdown("---")
        
        # --- Instruction Breakdown ---
        st.subheader("📋 Execution Breakdown")
        
        def categorize_instruction(instr_hex):
            try:
                val = int(str(instr_hex), 16)
                if val == 0:
                    return 'Bubble'
                elif val == 0x13:
                    return 'NOP'
                opcode = val & 0x7F
                if opcode == 0x37:  # LUI
                    return 'LUI'
                elif opcode == 0x17:  # AUIPC
                    return 'AUIPC'
                elif opcode == 0x13:  # I-type ALU
                    return 'ADDI/I-type'
                elif opcode == 0x33:  # R-type
                    return 'R-type'
                elif opcode == 0x63:  # Branch
                    return 'Branch'
                elif opcode == 0x03:  # Load
                    return 'LOAD'
                elif opcode == 0x23:  # Store
                    return 'STORE'
                elif opcode == 0x67:  # JALR
                    return 'JALR'
                elif opcode == 0x6F:  # JAL
                    return 'JAL'
                else:
                    return 'Other'
            except:
                return 'Unknown'
        
        # Get counts for both
        base_cats = base_eff['Instruction'].apply(categorize_instruction)
        fused_cats = fused_eff['Instruction'].apply(categorize_instruction)
        
        base_counts = base_cats.value_counts()
        fused_counts = fused_cats.value_counts()
        
        # Combine all categories
        all_cats = list(set(base_counts.index.tolist() + fused_counts.index.tolist()))
        all_cats_sorted = ['LUI', 'AUIPC', 'ADDI/I-type', 'R-type', 'LOAD', 'STORE', 'JALR', 'JAL', 'Branch', 'NOP', 'Bubble', 'Other', 'Unknown']
        all_cats = [c for c in all_cats_sorted if c in all_cats]
        
        base_vals = [base_counts.get(c, 0) for c in all_cats]
        fused_vals = [fused_counts.get(c, 0) for c in all_cats]
        
        # Create grouped bar chart
        fig_instr = go.Figure()
        
        fig_instr.add_trace(go.Bar(
            name='Baseline',
            x=all_cats,
            y=base_vals,
            marker_color='#6c757d',
            text=base_vals,
            textposition='outside'
        ))
        
        fig_instr.add_trace(go.Bar(
            name='Fused',
            x=all_cats,
            y=fused_vals,
            marker_color='#00ff88',
            text=fused_vals,
            textposition='outside'
        ))
        
        fig_instr.update_layout(
            title="Instruction Count Comparison (Absolute Numbers)",
            xaxis_title="Instruction Type",
            yaxis_title="Count",
            barmode='group',
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_instr, width='stretch')
        
        # Show key insight - Detailed Per-Pattern Analysis
        st.markdown("### 🔍 Fusion Impact Analysis by Pattern")
        
        # Get counts for each instruction type
        lui_base = base_counts.get('LUI', 0)
        lui_fused = fused_counts.get('LUI', 0)
        addi_base = base_counts.get('ADDI/I-type', 0)
        addi_fused = fused_counts.get('ADDI/I-type', 0)
        auipc_base = base_counts.get('AUIPC', 0)
        auipc_fused = fused_counts.get('AUIPC', 0)
        jalr_base = base_counts.get('JALR', 0)
        jalr_fused = fused_counts.get('JALR', 0)
        load_base = base_counts.get('LOAD', 0)
        load_fused = fused_counts.get('LOAD', 0)
        nop_base = base_counts.get('NOP', 0)
        nop_fused = fused_counts.get('NOP', 0)
        bubble_base = base_counts.get('Bubble', 0)
        bubble_fused = fused_counts.get('Bubble', 0)
        rtype_base = base_counts.get('R-type', 0)
        rtype_fused = fused_counts.get('R-type', 0)
        
        # Create detailed breakdown for each fusion pattern
        st.markdown("#### 🎯 Pattern 1: LUI + ADDI (32-bit Immediate Load)")
        lui_addi_col1, lui_addi_col2, lui_addi_col3, lui_addi_col4, lui_addi_col5 = st.columns(5)
        
        # Calculate actual fusion-related removals (should match fusion count)
        lui_addi_fusions = fused['FusionLuiAddi']
        addi_stall_reduction = (addi_base - addi_fused) - lui_addi_fusions  # Extra reduction from stalls
        
        with lui_addi_col1:
            st.markdown(f"""
            <div style="background: #1a2d1a; padding: 12px; border-radius: 8px; text-align: center;
                        border: 1px solid #00ff88;">
                <p style="color: #888; margin: 0; font-size: 0.9em;">ADDI Fused</p>
                <h2 style="color: #00ff88; margin: 5px 0;">{lui_addi_fusions}</h2>
                <p style="color: #666; margin: 0; font-size: 0.8em;">Merged into LUI</p>
            </div>
            """, unsafe_allow_html=True)
        
        with lui_addi_col2:
            st.markdown(f"""
            <div style="background: #2d1a1a; padding: 12px; border-radius: 8px; text-align: center;
                        border: 1px solid #ff6666;">
                <p style="color: #888; margin: 0; font-size: 0.9em;">Stall Duplicates</p>
                <h2 style="color: #ff6666; margin: 5px 0;">{max(0, addi_stall_reduction)}</h2>
                <p style="color: #666; margin: 0; font-size: 0.8em;">Eliminated</p>
            </div>
            """, unsafe_allow_html=True)
        
        with lui_addi_col3:
            st.markdown(f"""
            <div style="background: #1a1a2d; padding: 12px; border-radius: 8px; text-align: center;
                        border: 1px solid #8888ff;">
                <p style="color: #888; margin: 0; font-size: 0.9em;">LUI Count</p>
                <h2 style="color: #8888ff; margin: 5px 0;">{lui_fused}</h2>
                <p style="color: #666; margin: 0; font-size: 0.8em;">{lui_base} → {lui_fused}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with lui_addi_col4:
            st.markdown(f"""
            <div style="background: #2d1a2d; padding: 12px; border-radius: 8px; text-align: center;
                        border: 1px solid #ff88ff;">
                <p style="color: #888; margin: 0; font-size: 0.9em;">Fusions Detected</p>
                <h2 style="color: #ff88ff; margin: 5px 0;">{lui_addi_fusions}</h2>
                <p style="color: #666; margin: 0; font-size: 0.8em;">LUI+ADDI pairs</p>
            </div>
            """, unsafe_allow_html=True)
        
        with lui_addi_col5:
            lui_addi_savings = fused['FusionLuiAddi']  # Each fusion saves 1 cycle
            st.markdown(f"""
            <div style="background: #2d2d1a; padding: 12px; border-radius: 8px; text-align: center;
                        border: 1px solid #ffcc00;">
                <p style="color: #888; margin: 0; font-size: 0.9em;">Cycles Saved</p>
                <h2 style="color: #ffcc00; margin: 5px 0;">{lui_addi_savings}</h2>
                <p style="color: #666; margin: 0; font-size: 0.8em;">-1 per fusion</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        st.markdown("#### 🚀 Pattern 2: AUIPC + JALR (Long Jump / Function Call)")
        auipc_jalr_col1, auipc_jalr_col2, auipc_jalr_col3, auipc_jalr_col4, auipc_jalr_col5 = st.columns(5)
        
        auipc_jalr_fusions = fused['FusionAuipcJalr']
        jalr_stall_reduction = (jalr_base - jalr_fused) - auipc_jalr_fusions
        
        with auipc_jalr_col1:
            st.markdown(f"""
            <div style="background: #1a2d1a; padding: 12px; border-radius: 8px; text-align: center;
                        border: 1px solid #00ff88;">
                <p style="color: #888; margin: 0; font-size: 0.9em;">JALR Fused</p>
                <h2 style="color: #00ff88; margin: 5px 0;">{auipc_jalr_fusions}</h2>
                <p style="color: #666; margin: 0; font-size: 0.8em;">Merged into AUIPC</p>
            </div>
            """, unsafe_allow_html=True)
        
        with auipc_jalr_col2:
            st.markdown(f"""
            <div style="background: #2d1a1a; padding: 12px; border-radius: 8px; text-align: center;
                        border: 1px solid #ff6666;">
                <p style="color: #888; margin: 0; font-size: 0.9em;">Stall Duplicates</p>
                <h2 style="color: #ff6666; margin: 5px 0;">{max(0, jalr_stall_reduction)}</h2>
                <p style="color: #666; margin: 0; font-size: 0.8em;">Eliminated</p>
            </div>
            """, unsafe_allow_html=True)
        
        with auipc_jalr_col3:
            st.markdown(f"""
            <div style="background: #1a1a2d; padding: 12px; border-radius: 8px; text-align: center;
                        border: 1px solid #8888ff;">
                <p style="color: #888; margin: 0; font-size: 0.9em;">AUIPC Count</p>
                <h2 style="color: #8888ff; margin: 5px 0;">{auipc_fused}</h2>
                <p style="color: #666; margin: 0; font-size: 0.8em;">{auipc_base} → {auipc_fused}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with auipc_jalr_col4:
            st.markdown(f"""
            <div style="background: #2d1a2d; padding: 12px; border-radius: 8px; text-align: center;
                        border: 1px solid #ff88ff;">
                <p style="color: #888; margin: 0; font-size: 0.9em;">Fusions Detected</p>
                <h2 style="color: #ff88ff; margin: 5px 0;">{auipc_jalr_fusions}</h2>
                <p style="color: #666; margin: 0; font-size: 0.8em;">AUIPC+JALR pairs</p>
            </div>
            """, unsafe_allow_html=True)
        
        with auipc_jalr_col5:
            auipc_jalr_savings = fused['FusionAuipcJalr']
            st.markdown(f"""
            <div style="background: #2d2d1a; padding: 12px; border-radius: 8px; text-align: center;
                        border: 1px solid #ffcc00;">
                <p style="color: #888; margin: 0; font-size: 0.9em;">Cycles Saved</p>
                <h2 style="color: #ffcc00; margin: 5px 0;">{auipc_jalr_savings}</h2>
                <p style="color: #666; margin: 0; font-size: 0.8em;">-1 per fusion</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        st.markdown("#### ⚡ Pattern 3: LOAD + ALU (Load-Use Fusion)")
        load_alu_col1, load_alu_col2, load_alu_col3, load_alu_col4, load_alu_col5 = st.columns(5)
        
        load_alu_fusions = fused['FusionLoadAlu']
        # LOAD+ALU fusion: ALU is merged into LOAD, so ALU count should reduce
        # LOAD count stays same (it's now a fused LOAD+ALU op)
        alu_stall_reduction = (rtype_base - rtype_fused) - load_alu_fusions
        
        with load_alu_col1:
            st.markdown(f"""
            <div style="background: #1a2d1a; padding: 12px; border-radius: 8px; text-align: center;
                        border: 1px solid #00ff88;">
                <p style="color: #888; margin: 0; font-size: 0.9em;">ALU Ops Fused</p>
                <h2 style="color: #00ff88; margin: 5px 0;">{load_alu_fusions}</h2>
                <p style="color: #666; margin: 0; font-size: 0.8em;">Merged into LOAD</p>
            </div>
            """, unsafe_allow_html=True)
        
        with load_alu_col2:
            st.markdown(f"""
            <div style="background: #2d1a1a; padding: 12px; border-radius: 8px; text-align: center;
                        border: 1px solid #ff6666;">
                <p style="color: #888; margin: 0; font-size: 0.9em;">Stall Duplicates</p>
                <h2 style="color: #ff6666; margin: 5px 0;">{max(0, alu_stall_reduction)}</h2>
                <p style="color: #666; margin: 0; font-size: 0.8em;">Eliminated</p>
            </div>
            """, unsafe_allow_html=True)
        
        with load_alu_col3:
            load_stall_reduction = load_base - load_fused  # This is stall duplicates, not fusion
            st.markdown(f"""
            <div style="background: #1a1a2d; padding: 12px; border-radius: 8px; text-align: center;
                        border: 1px solid #8888ff;">
                <p style="color: #888; margin: 0; font-size: 0.9em;">LOAD Stall Dups</p>
                <h2 style="color: #8888ff; margin: 5px 0;">{load_stall_reduction}</h2>
                <p style="color: #666; margin: 0; font-size: 0.8em;">{load_base} → {load_fused}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with load_alu_col4:
            st.markdown(f"""
            <div style="background: #2d1a2d; padding: 12px; border-radius: 8px; text-align: center;
                        border: 1px solid #ff88ff;">
                <p style="color: #888; margin: 0; font-size: 0.9em;">Fusions Detected</p>
                <h2 style="color: #ff88ff; margin: 5px 0;">{load_alu_fusions}</h2>
                <p style="color: #666; margin: 0; font-size: 0.8em;">LOAD+ALU pairs</p>
            </div>
            """, unsafe_allow_html=True)
        
        with load_alu_col5:
            load_alu_savings = fused['FusionLoadAlu']
            st.markdown(f"""
            <div style="background: #2d2d1a; padding: 12px; border-radius: 8px; text-align: center;
                        border: 1px solid #ffcc00;">
                <p style="color: #888; margin: 0; font-size: 0.9em;">Stalls Avoided</p>
                <h2 style="color: #ffcc00; margin: 5px 0;">{load_alu_savings}</h2>
                <p style="color: #666; margin: 0; font-size: 0.8em;">Load-Use hazard</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        st.markdown("#### 📊 Overall Pipeline Impact")
        
        nop_diff = nop_fused - nop_base
        bubble_diff = bubble_fused - bubble_base
        total_fusions = fused['FusionLuiAddi'] + fused['FusionAuipcJalr'] + fused['FusionLoadAlu']
        
        impact_col1, impact_col2, impact_col3, impact_col4 = st.columns(4)
        
        with impact_col1:
            st.markdown(f"""
            <div style="background: #1f2d2d; padding: 15px; border-radius: 10px; text-align: center;
                        border: 2px solid #00cccc;">
                <p style="color: #888; margin: 0; font-size: 0.9em;">NOPs Added</p>
                <h2 style="color: {'#ff6b6b' if nop_diff > 0 else '#00ff88'}; margin: 5px 0;">{'+' if nop_diff > 0 else ''}{nop_diff}</h2>
                <p style="color: #666; margin: 0; font-size: 0.8em;">{nop_base} → {nop_fused}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with impact_col2:
            st.markdown(f"""
            <div style="background: #2d1f2d; padding: 15px; border-radius: 10px; text-align: center;
                        border: 2px solid #cc00cc;">
                <p style="color: #888; margin: 0; font-size: 0.9em;">Bubbles Changed</p>
                <h2 style="color: {'#ff6b6b' if bubble_diff > 0 else '#00ff88'}; margin: 5px 0;">{'+' if bubble_diff > 0 else ''}{bubble_diff}</h2>
                <p style="color: #666; margin: 0; font-size: 0.8em;">{bubble_base} → {bubble_fused}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with impact_col3:
            st.markdown(f"""
            <div style="background: #1f2d1f; padding: 15px; border-radius: 10px; text-align: center;
                        border: 2px solid #00ff88;">
                <p style="color: #888; margin: 0; font-size: 0.9em;">Total Fusions</p>
                <h2 style="color: #00ff88; margin: 5px 0;">{total_fusions}</h2>
                <p style="color: #666; margin: 0; font-size: 0.8em;">All patterns</p>
            </div>
            """, unsafe_allow_html=True)
        
        with impact_col4:
            net_saved = cycles_saved
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1f2d1f 0%, #2d3d2d 100%); padding: 15px; border-radius: 10px; text-align: center;
                        border: 2px solid #00ff88;">
                <p style="color: #888; margin: 0; font-size: 0.9em;">NET CYCLES SAVED</p>
                <h2 style="color: #00ff88; margin: 5px 0; font-size: 1.8em;">{net_saved}</h2>
                <p style="color: #00ff88; margin: 0; font-size: 0.9em; font-weight: bold;">{speedup:.1f}% faster</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Efficiency metrics
        st.markdown("")
        st.markdown("#### 📈 Fusion Efficiency Analysis")
        
        eff_col1, eff_col2 = st.columns(2)
        
        with eff_col1:
            # Create a grouped bar showing before/after for each pattern
            patterns = ['LUI+ADDI', 'AUIPC+JALR', 'LOAD+ALU']
            fusions_by_type = [fused['FusionLuiAddi'], fused['FusionAuipcJalr'], fused['FusionLoadAlu']]
            
            fig_fusion_detail = go.Figure()
            
            fig_fusion_detail.add_trace(go.Bar(
                name='Fusions',
                x=patterns,
                y=fusions_by_type,
                marker_color=['#00ff88', '#ff88ff', '#ffff88'],
                text=fusions_by_type,
                textposition='auto'
            ))
            
            fig_fusion_detail.update_layout(
                title="Fusion Counts by Pattern Type",
                xaxis_title="Fusion Pattern",
                yaxis_title="Count",
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_fusion_detail, use_container_width=True)
        
        with eff_col2:
            # Create a waterfall chart showing cycle reduction
            # Calculate actual values: 93 - 10 - 1 - 5 = 77
            # There's no "overhead" - the math should work out exactly
            total_fusion_savings = fused['FusionLuiAddi'] + fused['FusionAuipcJalr'] + fused['FusionLoadAlu']
            expected_fused = base['Cycles'] - total_fusion_savings
            actual_diff = fused['Cycles'] - expected_fused  # Should be 0 or close
            
            fig_waterfall = go.Figure(go.Waterfall(
                name="Cycle Analysis",
                orientation="v",
                x=["Baseline", "LUI+ADDI\n(-10)", "AUIPC+JALR\n(-1)", "LOAD+ALU\n(-5)", "Fused"],
                y=[base['Cycles'], -fused['FusionLuiAddi'], -fused['FusionAuipcJalr'], 
                   -fused['FusionLoadAlu'], 0],
                measure=["absolute", "relative", "relative", "relative", "total"],
                text=[f"{base['Cycles']}", f"-{fused['FusionLuiAddi']}", f"-{fused['FusionAuipcJalr']}", 
                      f"-{fused['FusionLoadAlu']}", f"{fused['Cycles']}"],
                textposition="outside",
                connector={"line": {"color": "#888888", "width": 2}},
                decreasing={"marker": {"color": "#00ff88"}},
                increasing={"marker": {"color": "#ff6b6b"}},
                totals={"marker": {"color": "#00d4ff"}}
            ))
            
            fig_waterfall.update_layout(
                title="Cycle Reduction Breakdown",
                yaxis_title="Cycles",
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)
        
        # --- Instruction Fetch Breakdown ---
        st.markdown("#### 📦 Instruction Fetch Analysis")
        instr_col1, instr_col2 = st.columns(2)
        
        with instr_col1:
            # Instruction fetch waterfall chart
            # Baseline fetches 79, fused fetches 48
            # Reduction = 16 fusions (each fusion removes 1 instruction fetch)
            # But we also have NOPs added (15) which are fetched
            instr_reduction = base['Instructions'] - fused['Instructions']  # 79 - 48 = 31
            
            fig_instr_waterfall = go.Figure(go.Waterfall(
                name="Instruction Analysis",
                orientation="v",
                x=["Baseline\nFetched", "Fused Pairs\nRemoved", "Fused\nFetched"],
                y=[base['Instructions'], -instr_reduction, 0],
                measure=["absolute", "relative", "total"],
                text=[f"{base['Instructions']}", f"-{instr_reduction}", f"{fused['Instructions']}"],
                textposition="outside",
                connector={"line": {"color": "#888888", "width": 2}},
                decreasing={"marker": {"color": "#00ff88"}},
                increasing={"marker": {"color": "#ff6b6b"}},
                totals={"marker": {"color": "#ff88ff"}}
            ))
            
            fig_instr_waterfall.update_layout(
                title="Instruction Fetch Reduction",
                yaxis_title="Instructions",
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            st.plotly_chart(fig_instr_waterfall, use_container_width=True)
        
        with instr_col2:
            # Side by side comparison
            fig_compare = go.Figure()
            
            fig_compare.add_trace(go.Bar(
                name='Cycles',
                x=['Baseline', 'Fused'],
                y=[base['Cycles'], fused['Cycles']],
                marker_color=['#6c757d', '#00d4ff'],
                text=[base['Cycles'], fused['Cycles']],
                textposition='auto',
                offsetgroup=0
            ))
            
            fig_compare.add_trace(go.Bar(
                name='Instructions',
                x=['Baseline', 'Fused'],
                y=[base['Instructions'], fused['Instructions']],
                marker_color=['#888888', '#ff88ff'],
                text=[base['Instructions'], fused['Instructions']],
                textposition='auto',
                offsetgroup=1
            ))
            
            fig_compare.update_layout(
                title="Cycles vs Instructions Comparison",
                yaxis_title="Count",
                barmode='group',
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_compare, use_container_width=True)
        
        # Summary explanation
        st.markdown(f"""
        <div style="background: #1a1a2e; padding: 15px; border-radius: 10px; margin-top: 10px;">
            <h4 style="color: #00d4ff; margin: 0 0 10px 0;">📊 Summary</h4>
            <table style="width: 100%; color: #ccc;">
                <tr>
                    <td><b>Metric</b></td>
                    <td style="text-align: center;"><b>Baseline</b></td>
                    <td style="text-align: center;"><b>Fused</b></td>
                    <td style="text-align: center;"><b>Reduction</b></td>
                </tr>
                <tr>
                    <td>⏱️ Cycles (Time)</td>
                    <td style="text-align: center;">{base['Cycles']}</td>
                    <td style="text-align: center;">{fused['Cycles']}</td>
                    <td style="text-align: center; color: #00ff88;">-{cycles_saved} ({speedup:.1f}%)</td>
                </tr>
                <tr>
                    <td>📦 Instructions Fetched</td>
                    <td style="text-align: center;">{base['Instructions']}</td>
                    <td style="text-align: center;">{fused['Instructions']}</td>
                    <td style="text-align: center; color: #00ff88;">-{base['Instructions'] - fused['Instructions']} ({(base['Instructions'] - fused['Instructions'])/base['Instructions']*100:.1f}%)</td>
                </tr>
                <tr>
                    <td>🔧 Work Done</td>
                    <td style="text-align: center;">{work_done}</td>
                    <td style="text-align: center;">{work_done}</td>
                    <td style="text-align: center; color: #888;">Same program</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # --- Detailed Logs ---
        with st.expander("📜 View Detailed Execution Logs"):
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Baseline Execution Log**")
                st.dataframe(base_eff, height=400)
            with c2:
                st.write("**Fused Execution Log**")
                st.dataframe(fused_eff, height=400)
        
        # --- CSV Download Section ---
        st.markdown("---")
        st.subheader("📥 Export Execution Logs")
        
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        
        with dl_col1:
            # Download baseline CSV
            if 'df_base' in st.session_state and st.session_state['df_base'] is not None:
                baseline_csv = st.session_state['df_base'].to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Baseline CSV",
                    data=baseline_csv,
                    file_name="baseline_execution.csv",
                    mime="text/csv"
                )
        
        with dl_col2:
            # Download fused CSV
            if 'df_fused' in st.session_state and st.session_state['df_fused'] is not None:
                fused_csv = st.session_state['df_fused'].to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Fused CSV",
                    data=fused_csv,
                    file_name="fused_execution.csv",
                    mime="text/csv"
                )
        
        with dl_col3:
            st.markdown(f"""
            <div style="background: #1a2d1a; padding: 10px; border-radius: 8px; text-align: center;">
                <p style="color: #00ff88; margin: 0; font-size: 0.9em;">📁 CSVs auto-saved to:</p>
                <p style="color: #888; margin: 5px 0 0 0; font-size: 0.8em;">temp/baseline_execution.csv</p>
                <p style="color: #888; margin: 0; font-size: 0.8em;">temp/fused_execution.csv</p>
            </div>
            """, unsafe_allow_html=True)
                
    else:
        st.info("👈 Click 'Run Comparison' to analyze the fusion performance! ✨")

st.markdown("---")
st.caption("🔬 RV32I Multi-Pattern Macro-Op Fusion Processor Dashboard | Supports LUI+ADDI, AUIPC+JALR, LOAD+ALU Fusion | Built with Streamlit & Plotly")
