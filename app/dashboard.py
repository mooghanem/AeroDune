import sys
import os

# Ensure core module is in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from core.dem_handler import load_and_mosaic_dem
from core.wind_parser import parse_nasa_wind_csv
from core.physics_engine import SandSimulation2D
from core.infrastructure import fetch_and_rasterize_highways
from core.economic_engine import SilicaSandEconomicEngine


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AeroDune Framework | North Sinai",
    page_icon="🏜️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏜️ AeroDune: Physics-Based Sand Simulation & Economic Framework")
st.markdown("""
*Simulating 2D wind-blown sand migration over North Sinai infrastructure (Qantara–El Arish Highway)
and analyzing macroeconomic GDP value-addition of Egypt's high-purity silica sand.*
""")

# -----------------------------------------------------------------------------
# CACHED DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_framework_data():
    dem_matrix, transform = load_and_mosaic_dem()
    wind_df = parse_nasa_wind_csv()
    road_mask = fetch_and_rasterize_highways(dem_matrix.shape)
    return dem_matrix, transform, wind_df, road_mask

dem_matrix, transform, wind_df, road_mask = load_framework_data()
econ_engine = SilicaSandEconomicEngine()

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Simulation Controls")

sim_days = st.sidebar.slider("Simulation Horizon (Days)", min_value=1, max_value=365, value=60, step=5)
wind_mult = st.sidebar.slider("Wind Intensity Multiplier", min_value=0.5, max_value=2.5, value=1.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Windbreak Barrier Config")
enable_barrier = st.sidebar.checkbox("Deploy Virtual Windbreak Barrier", value=True)
barrier_row = st.sidebar.slider("Barrier Row Grid Position", 0, dem_matrix.shape[0]-1, int(dem_matrix.shape[0]*0.45))
barrier_length = st.sidebar.slider("Barrier Length (Cells)", 10, dem_matrix.shape[1], int(dem_matrix.shape[1]*0.6))
barrier_porosity = st.sidebar.slider("Barrier Porosity (P)", 0.0, 1.0, 0.3, step=0.05)

# Create Barrier Mask Matrix
barrier_mask = np.zeros_like(dem_matrix, dtype=np.uint8)
if enable_barrier:
    start_col = (dem_matrix.shape[1] - barrier_length) // 2
    barrier_mask[barrier_row, start_col:start_col+barrier_length] = 1

# -----------------------------------------------------------------------------
# DASHBOARD TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Topography & Infrastructure",
    "🌪️ Live Sand Migration Engine",
    "🚨 Hazard Alarm & Windbreak",
    "📈 Economic & GDP Analytics"
])

# -----------------------------------------------------------------------------
# TAB 1: TOPOGRAPHY & INFRASTRUCTURE MAP
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("North Sinai Digital Elevation Model (DEM) & Highway Network")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(dem_matrix, cmap='terrain', origin='upper')
        plt.colorbar(im, ax=ax, label="Elevation (Meters above Sea Level)")
        
        # Overlay Road Network in Red
        road_y, road_x = np.where(road_mask > 0)
        ax.scatter(road_x, road_y, c='red', s=3, alpha=0.7, label='Qantara–El Arish Highway')
        
        # Overlay Barrier in Cyan
        if enable_barrier:
            b_y, b_x = np.where(barrier_mask > 0)
            ax.scatter(b_x, b_y, c='cyan', s=10, marker='s', label='Windbreak Barrier')

        ax.set_title("SRTM Elevation Mosaic with Overlaid Infrastructure")
        ax.set_xlabel("Grid Columns (30m cell resolution)")
        ax.set_ylabel("Grid Rows")
        ax.legend(loc='lower right')
        st.pyplot(fig)
        
    with col2:
        st.metric("DEM Grid Dimensions", f"{dem_matrix.shape[0]} × {dem_matrix.shape[1]} cells")
        st.metric("Grid Spatial Resolution", "30 Meters / Cell")
        st.metric("Max Terrain Elevation", f"{np.max(dem_matrix):.1f} m")
        st.metric("Min Terrain Elevation", f"{np.min(dem_matrix):.1f} m")
        st.info("The map integrates 5 USGS SRTM GeoTIFF tiles covering North Sinai.")

# -----------------------------------------------------------------------------
# TAB 2: LIVE SIMULATION ENGINE
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("2D Bagnold-Exner Spatiotemporal Sand Migration Solver")
    
    run_sim = st.button("🚀 Run Physics Simulation", type="primary")
    
    if run_sim or "sim_accumulation" in st.session_state:
        if run_sim:
            sim_engine = SandSimulation2D(dem_matrix)
            progress_bar = st.progress(0)
            
            # Step through daily wind records
            for d in range(sim_days):
                wind_row = wind_df.iloc[d % len(wind_df)]
                ws = wind_row['WS10M']
                wd = wind_row['WD10M']
                
                sim_engine.step(ws, wd, barrier_mask=barrier_mask if enable_barrier else None, speed_multiplier=wind_mult)
                progress_bar.progress((d + 1) / sim_days)
                
            accumulation = sim_engine.get_accumulation_map()
            st.session_state["sim_accumulation"] = accumulation
            st.session_state["final_Z"] = sim_engine.Z
        else:
            accumulation = st.session_state["sim_accumulation"]
            
        fig_acc, ax_acc = plt.subplots(figsize=(10, 5))
        c = ax_acc.imshow(accumulation, cmap='seismic', vmin=-1.5, vmax=1.5)
        plt.colorbar(c, ax=ax_acc, label="Sand Accumulation (+) / Erosion (-) [Meters]")
        
        # Overlay roads
        ax_acc.scatter(road_x, road_y, c='black', s=2, alpha=0.5, label='Highway Path')
        
        ax_acc.set_title(f"Cumulative Bed Height Elevation Change after {sim_days} Days")
        st.pyplot(fig_acc)

# -----------------------------------------------------------------------------
# TAB 3: HAZARD ALARM & WINDBREAK OPTIMIZATION
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Infrastructure Hazard Detection & Windbreak Defense Matrix")
    
    if "sim_accumulation" in st.session_state:
        accumulation = st.session_state["sim_accumulation"]
        road_sand_depth = accumulation[road_mask > 0]
        max_road_sand = np.max(road_sand_depth) if len(road_sand_depth) > 0 else 0.0
        
        hazard_threshold = 0.5 # 0.5 meters of sand over road triggers alarm
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Max Sand Depth on Road", f"{max_road_sand:.2f} Meters")
        c2.metric("Hazard Threshold", f"{hazard_threshold} Meters")
        
        if max_road_sand > hazard_threshold:
            c3.error("🚨 CRITICAL HAZARD ALARM: Road Encroachment Detected!")
            st.error(f"WARNING: Sand accumulation on the Qantara-El Arish Highway has exceeded {hazard_threshold}m! Clear-out required.")
        else:
            c3.success("✅ ROAD CLEAR: No Severe Sand Obstruction Detected.")
            st.success("Infrastructure remains protected under current barrier parameters.")

        # Comparison Plot: Sand Depth Along Road Profile
        fig_prof, ax_prof = plt.subplots(figsize=(10, 3.5))
        ax_prof.plot(road_sand_depth, color='crimson', label='Sand Accumulation along Highway Cells')
        ax_prof.axhline(hazard_threshold, color='black', linestyle='--', label='Critical Hazard Level (0.5m)')
        ax_prof.set_ylabel("Depth (m)")
        ax_prof.set_xlabel("Highway Cell Index")
        ax_prof.set_title("Sand Accumulation Profile along Highway")
        ax_prof.legend()
        st.pyplot(fig_prof)
    else:
        st.info("Please click 'Run Physics Simulation' in Tab 2 to calculate road hazard metrics.")

# -----------------------------------------------------------------------------
# TAB 4: MACROECONOMIC & GDP ANALYTICS
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Egyptian Silica Sand Industrialization & Macroeconomic Impact")
    st.markdown("""
    *Evaluating the economic impact of industrializing Sinai's raw silica sand into high-value manufactured **Float & Solar Panel Glass**.*
    """)
    
    st.markdown("---")
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        sand_tonnage = st.number_input("Annual Silica Sand Extraction (Tons)", min_value=500000, max_value=20000000, value=3500000, step=250000)
        raw_price = st.slider("Raw Sand Export Price ($/Ton)", 10.0, 60.0, 25.0)
        
    with col_e2:
        glass_price = st.slider("Manufactured Float/Solar Glass Price ($/Ton)", 250.0, 800.0, 450.0)
        conversion_yield = st.slider("Industrial Conversion Yield (%)", 50, 95, 80) / 100.0

    econ_results = econ_engine.calculate_value_add(
        raw_sand_tonnage=sand_tonnage,
        raw_price_per_ton=raw_price,
        glass_yield_ratio=conversion_yield,
        glass_price_per_ton=glass_price
    )
    
    # Financial KPI Metrics Cards
    st.markdown("### 📊 Annual Financial Return Metrics")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Raw Sand Revenue", f"${econ_results['raw_revenue_usd']/1e6:.1f} Million")
    kpi2.metric("Manufactured Glass Revenue", f"${econ_results['glass_revenue_usd']/1e6:.1f} Million")
    kpi3.metric("Net Annual Value-Add", f"+${econ_results['net_value_add_usd']/1e6:.1f} Million")
    kpi4.metric("Value Addition Multiplier", f"{econ_results['value_multiplier']:.1f}x")

    st.success(f"💡 **Direct GDP Impact:** Industrializing this silica sand volume adds **+{econ_results['gdp_boost_percentage']:.3f}% directly to Egypt's Total GDP** ($380 Billion USD baseline) every year!")

    # 10-Year Projection Chart
    st.markdown("### 📈 10-Year Revenue Trajectory (Raw Export vs Industrial Glass)")
    df_proj = econ_engine.generate_10year_projection(initial_tonnage=sand_tonnage)
    
    fig_plotly = px.bar(
        df_proj, 
        x='Year', 
        y=['Raw Sand Export ($M)', 'Net Value Addition ($M)'],
        title="10-Year Projected Revenue Comparison for Egypt ($ Millions USD)",
        barmode='stack',
        color_discrete_sequence=['#ef553b', '#00cc96']
    )
    st.plotly_chart(fig_plotly, use_container_width=True)
