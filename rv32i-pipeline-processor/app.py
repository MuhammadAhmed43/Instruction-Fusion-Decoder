import streamlit as st
import pandas as pd
import subprocess
import os
import plotly.graph_objects as go

# --- Configuration ---
PROJECT_ROOT = os.getcwd()
INSTR_MEM_PATH = os.path.join(PROJECT_ROOT, "tb", "instr.mem")
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "temp", "simulation.log")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "temp", "microprocessor.output")

st.set_page_config(page_title="RV32I Fusion Analytics", layout="wide")

st.title("⚡ RV32I Macro-Op Fusion Analytics")
st.markdown("""
This dashboard compares the performance of the **Standard RV32I Pipeline** vs. the **Fused Pipeline**.
It runs two separate simulations to demonstrate the impact of fusing `LUI` + `ADDI` sequences.
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

    # Read flist
    with open("flist", "r") as f:
        files = [l.strip().replace("${CORE_ROOT}", ".") for l in f.readlines() if l.strip()]
    
    # Use -D to define the macro for the parameter default value
    compile_cmd = [IVERILOG_BIN, "-o", OUTPUT_FILE, f"-DFUSION_ENABLE_VAL={param_val}"] + files
    
    # Run Compilation
    # Add iverilog bin path to environment
    env = os.environ.copy()
    env["PATH"] += os.pathsep + os.path.dirname(IVERILOG_BIN)
    subprocess.run(compile_cmd, check=True, env=env)
    
    # 2. Run Simulation
    if os.path.exists(LOG_FILE_PATH):
        os.remove(LOG_FILE_PATH)
        
    subprocess.run([VVP_BIN, OUTPUT_FILE], check=True, env=env)
    
    # 3. Parse Log
    if os.path.exists(LOG_FILE_PATH):
        return pd.read_csv(LOG_FILE_PATH)
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

    # 2. Fix Fusion Hits (Handle 'x' strings)
    # Count fusion hits only within effective execution window (not the full simulation with wrapping)
    fusion_hits = 0
    if 'FuseFlag' in effective_df.columns:
        # Coerce to numeric, turn errors/'x' to 0
        fusion_hits = pd.to_numeric(effective_df['FuseFlag'], errors='coerce').fillna(0).sum()

    # 3. Calculate Metrics based on Effective Duration
    ipc = instr_count / effective_cycles if effective_cycles > 0 else 0
    cpi = effective_cycles / instr_count if instr_count > 0 else 0
    
    return {
        "Label": label,
        "Cycles": effective_cycles, # Use effective cycles, not total simulation time
        "Instructions": instr_count,
        "IPC": ipc,
        "CPI": cpi,
        "FusionHits": int(fusion_hits),
        "Data": active_df # Keep full data for graph
    }

# --- UI Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📝 Assembly Input")
    default_asm = """// Fusion Benchmark Loop
// Initialize loop counter (x5 = 10)
addi x5, x0, 10

loop:
// Pair 1: LUI + ADDI (Should Fuse)
lui x1, 0x12345000
addi x1, x1, 0x678

// Pair 2: LUI + ADDI (Should Fuse)
lui x2, 0x87654000
addi x2, x2, 0x321

// Normal ALU Ops
add x3, x1, x2
sub x4, x1, x2

// Decrement and Branch
addi x5, x5, -1
bne x5, x0, loop

// End
nop
nop
"""
    asm_code = st.text_area("Enter RISC-V Assembly", default_asm, height=300)
    
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
            st.success("Comparison Complete!")

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
            st.metric("🔥 Fusion Hits", f"{fused['FusionHits']}")
            st.metric("⏱️ Cycles Saved", f"{cycles_saved}")
        with hero_col3:
            st.metric("📊 Fused Cycles", f"{fused['Cycles']}")
            st.metric("📊 Baseline Cycles", f"{base['Cycles']}")
        
        st.markdown("---")
        
        # --- Detailed Metrics Comparison ---
        st.subheader("📊 Detailed Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fused IPC", f"{fused['IPC']:.3f}", delta=f"{fused['IPC'] - base['IPC']:.3f}")
        m2.metric("Baseline IPC", f"{base['IPC']:.3f}")
        m3.metric("Fused CPI", f"{fused['CPI']:.2f}", delta=f"{fused['CPI'] - base['CPI']:.2f}", delta_color="inverse")
        m4.metric("Baseline CPI", f"{base['CPI']:.2f}")
        
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
            # IPC comparison
            fig_ipc = go.Figure(data=[
                go.Bar(name='Baseline', x=['IPC'], y=[base['IPC']], 
                       marker_color='#6c757d', text=[f"{base['IPC']:.3f}"], textposition='auto'),
                go.Bar(name='Fused', x=['IPC'], y=[fused['IPC']], 
                       marker_color='#00ff88', text=[f"{fused['IPC']:.3f}"], textposition='auto')
            ])
            fig_ipc.update_layout(
                title="Instructions Per Cycle (IPC)",
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
                elif opcode == 0x13:  # I-type
                    return 'ADDI/I-type'
                elif opcode == 0x33:  # R-type
                    return 'R-type'
                elif opcode == 0x63:  # Branch
                    return 'Branch'
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
        all_cats_sorted = ['LUI', 'ADDI/I-type', 'R-type', 'Branch', 'NOP', 'Bubble', 'Other', 'Unknown']
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
        
        # Show key insight
        addi_base = base_counts.get('ADDI/I-type', 0)
        addi_fused = fused_counts.get('ADDI/I-type', 0)
        nop_base = base_counts.get('NOP', 0)
        nop_fused = fused_counts.get('NOP', 0)
        
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        with insight_col1:
            st.markdown(f"""
            <div style="background: #2d2d1f; padding: 15px; border-radius: 10px; text-align: center;
                        border: 1px solid #ffcc00;">
                <h3 style="color: #ffcc00; margin: 0;">ADDI Instructions</h3>
                <p style="color: #888; margin: 5px 0;">Baseline: <b>{addi_base}</b> → Fused: <b>{addi_fused}</b></p>
                <p style="color: #00ff88; font-size: 1.2em; margin: 0;">↓ {addi_base - addi_fused} eliminated</p>
            </div>
            """, unsafe_allow_html=True)
        with insight_col2:
            st.markdown(f"""
            <div style="background: #1f2d2d; padding: 15px; border-radius: 10px; text-align: center;
                        border: 1px solid #00cccc;">
                <h3 style="color: #00cccc; margin: 0;">NOP (Flush Bubbles)</h3>
                <p style="color: #888; margin: 5px 0;">Baseline: <b>{nop_base}</b> → Fused: <b>{nop_fused}</b></p>
                <p style="color: #ff6b6b; font-size: 1.2em; margin: 0;">↑ {nop_fused - nop_base} added</p>
            </div>
            """, unsafe_allow_html=True)
        with insight_col3:
            net_saved = (addi_base - addi_fused) - (nop_fused - nop_base)
            st.markdown(f"""
            <div style="background: #1f2d1f; padding: 15px; border-radius: 10px; text-align: center;
                        border: 1px solid #00ff88;">
                <h3 style="color: #00ff88; margin: 0;">Net Cycles Saved</h3>
                <p style="color: #888; margin: 5px 0;">ADDI removed - NOPs added</p>
                <p style="color: #00ff88; font-size: 1.5em; margin: 0; font-weight: bold;">{net_saved} cycles</p>
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
                
    else:
        st.info("👈 Click 'Run Comparison' to analyze the fusion performance! ✨")

st.markdown("---")
st.caption("🔬 RV32I Macro-Op Fusion Processor Dashboard | Built with Streamlit & Plotly")
