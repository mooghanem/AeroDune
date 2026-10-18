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
# PAGE CONFIGURATION & CLEAN ORGANIC DESIGN SYSTEM
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AeroDune | Physics-Based Sand Simulation & Economic Framework",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Injection: Organic / Natural Design Tokens & Clean Controls
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700;800&family=Nunito:wght@400;600;700;800&display=swap');

    /* Design Tokens */
    :root {
        --bg-paper: #FDFCF8;
        --fg-loam: #2C2C24;
        --primary-moss: #5D7052;
        --primary-mist: #F3F4F1;
        --secondary-terracotta: #C18C5D;
        --accent-sand: #E6DCCD;
        --muted-stone: #F0EBE5;
        --muted-grass: #78786C;
        --border-timber: #DED8CF;
        --card-bg: #FEFEFA;
        --shadow-moss: 0 4px 16px -2px rgba(93, 112, 82, 0.12);
        --shadow-clay: 0 8px 24px -4px rgba(193, 140, 93, 0.18);
    }

    /* Global Typography & Background */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg-paper) !important;
        color: var(--fg-loam) !important;
        font-family: 'Nunito', sans-serif !important;
    }

    /* Subtle Grain Texture Overlay */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image: radial-gradient(#2C2C24 0.5px, transparent 0.5px);
        background-size: 28px 28px;
        opacity: 0.025;
        pointer-events: none;
        z-index: 1;
    }

    /* Headings (Fraunces Soft Serif) */
    h1, h2, h3, h4 {
        font-family: 'Fraunces', serif !important;
        color: var(--fg-loam) !important;
        font-weight: 700 !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: var(--muted-stone) !important;
        border-right: 1px solid var(--border-timber) !important;
        padding-top: 1.5rem !important;
    }

    /* Tab Navigation Pills */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem !important;
        background-color: var(--accent-sand) !important;
        padding: 0.4rem !important;
        border-radius: 9999px !important;
        border: 1px solid var(--border-timber) !important;
    }

    .stTabs [data-baseweb="tab"] {
        height: 2.5rem !important;
        border-radius: 9999px !important;
        color: var(--fg-loam) !important;
        font-family: 'Nunito', sans-serif !important;
        font-weight: 700 !important;
        padding: 0 1.25rem !important;
        border: none !important;
        transition: all 0.25s ease !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--primary-moss) !important;
        color: var(--primary-mist) !important;
        box-shadow: var(--shadow-moss) !important;
    }

    /* Custom Organic Buttons */
    .stButton>button {
        background-color: var(--primary-moss) !important;
        color: var(--primary-mist) !important;
        font-family: 'Nunito', sans-serif !important;
        font-weight: 700 !important;
        border-radius: 9999px !important;
        padding: 0.6rem 2rem !important;
        border: none !important;
        box-shadow: var(--shadow-moss) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-clay) !important;
        background-color: #4f6045 !important;
    }

    /* Metrics Cards */
    [data-testid="stMetric"] {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-timber) !important;
        border-radius: 1.25rem !important;
        padding: 1rem 1.25rem !important;
        box-shadow: var(--shadow-moss) !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Fraunces', serif !important;
        color: var(--primary-moss) !important;
        font-weight: 800 !important;
    }

    /* Clean Streamlit Slider Overrides */
    div[data-baseweb="slider"] {
        margin-bottom: 0.5rem !important;
    }

    /* Clean Docked Footer */
    .clean-footer {
        margin-top: 4rem;
        padding: 2rem 1rem;
        border-top: 1px solid var(--border-timber);
        text-align: center;
        background-color: transparent;
    }

    .clean-footer .footer-brand {
        font-family: 'Fraunces', serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--fg-loam);
        margin-bottom: 0.2rem;
    }

    .clean-footer .footer-subtitle {
        font-family: 'Nunito', sans-serif;
        font-size: 0.85rem;
        color: var(--muted-grass);
        max-width: 800px;
        margin: 0 auto 0.75rem auto;
        line-height: 1.4;
    }

    .clean-footer .footer-dev {
        font-family: 'Nunito', sans-serif;
        font-size: 0.9rem;
        font-weight: 700;
        color: var(--primary-moss);
        margin-bottom: 0.85rem;
    }

    .clean-footer .footer-links-row {
        display: flex;
        justify-content: center;
        gap: 1rem;
    }

    .clean-footer .footer-btn {
        display: inline-block;
        background-color: var(--card-bg);
        color: var(--primary-moss);
        padding: 0.4rem 1.2rem;
        border-radius: 9999px;
        border: 1px solid var(--border-timber);
        font-family: 'Nunito', sans-serif;
        font-size: 0.85rem;
        font-weight: 700;
        text-decoration: none;
        transition: all 0.2s ease;
    }

    .clean-footer .footer-btn:hover {
        background-color: var(--primary-moss);
        color: var(--primary-mist);
        border-color: var(--primary-moss);
    }
</style>
""", unsafe_allow_html=True)


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
# HEADER & BRANDING SECTION
# -----------------------------------------------------------------------------
st.markdown('<h1 style="font-size: 2.8rem; margin-bottom: 0.2rem;">AeroDune</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="color: #5D7052; font-weight: 600; margin-top: 0; font-size: 1.15rem;">A Physics-Based Spatiotemporal Simulation Framework for Wind-Blown Sand Encroachment, Windbreak Optimization, and Desert Resource Economics</h3>', unsafe_allow_html=True)
st.markdown("""
*Simulating 2D wind-blown sand migration over North Sinai infrastructure (Qantara–El Arish Highway)
and analyzing macroeconomic GDP value-addition of Egypt's high-purity silica sand.*
""")
st.markdown("<hr style='border: none; border-top: 1px solid #DED8CF; margin: 1.25rem 0;'>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.markdown("<h2 style='font-size: 1.3rem; margin-bottom: 1rem;'>Simulation Controls</h2>", unsafe_allow_html=True)

sim_days = st.sidebar.slider("Simulation Horizon (Days)", min_value=1, max_value=365, value=60, step=5)
wind_mult = st.sidebar.slider("Wind Intensity Multiplier", min_value=0.5, max_value=2.5, value=1.0, step=0.1)

st.sidebar.markdown("<hr style='border: none; border-top: 1px solid #DED8CF; margin: 1.25rem 0;'>", unsafe_allow_html=True)
st.sidebar.markdown("<h2 style='font-size: 1.3rem; margin-bottom: 1rem;'>Windbreak Barrier Config</h2>", unsafe_allow_html=True)

enable_barrier = st.sidebar.checkbox("Deploy Virtual Windbreak Barrier", value=True)
barrier_row = st.sidebar.slider("Barrier Row Grid Position", 0, dem_matrix.shape[0]-1, int(dem_matrix.shape[0]*0.35))
barrier_length = st.sidebar.slider("Barrier Length (Cells)", 10, dem_matrix.shape[1], int(dem_matrix.shape[1]*0.8))
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
    "Topography & Infrastructure",
    "Live Sand Migration Engine",
    "Hazard Alarm & Windbreak",
    "Economic & GDP Analytics"
])

# -----------------------------------------------------------------------------
# TAB 1: TOPOGRAPHY & INFRASTRUCTURE MAP
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("North Sinai Digital Elevation Model (DEM) & Highway Network")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#FDFCF8')
        ax.set_facecolor('#FDFCF8')
        im = ax.imshow(dem_matrix, cmap='terrain', origin='upper')
        cbar = plt.colorbar(im, ax=ax, label="Elevation (Meters above Sea Level)")
        cbar.ax.yaxis.label.set_color('#2C2C24')
        cbar.ax.tick_params(colors='#2C2C24')
        
        road_y, road_x = np.where(road_mask > 0)
        ax.scatter(road_x, road_y, c='#A85448', s=4, alpha=0.8, label='Qantara–El Arish Highway')
        
        if enable_barrier:
            b_y, b_x = np.where(barrier_mask > 0)
            ax.scatter(b_x, b_y, c='#5D7052', s=12, marker='s', label='Windbreak Barrier')

        ax.set_title("SRTM Elevation Mosaic with Overlaid Infrastructure", color='#2C2C24', fontsize=12, fontweight='bold')
        ax.set_xlabel("Grid Columns (30m cell resolution)", color='#2C2C24')
        ax.set_ylabel("Grid Rows", color='#2C2C24')
        ax.tick_params(colors='#2C2C24')
        ax.legend(loc='lower right', facecolor='#FEFEFA', edgecolor='#DED8CF')
        st.pyplot(fig)
        
    with col2:
        st.metric("DEM Grid Dimensions", f"{dem_matrix.shape[0]} x {dem_matrix.shape[1]}")
        st.metric("Spatial Resolution", "30 Meters / Cell")
        st.metric("Max Elevation", f"{np.max(dem_matrix):.1f} m")
        st.metric("Min Elevation", f"{np.min(dem_matrix):.1f} m")

# -----------------------------------------------------------------------------
# TAB 2: LIVE SIMULATION ENGINE
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("2D Bagnold-Exner Spatiotemporal Sand Migration Solver")
    
    run_sim = st.button("Run Physics Simulation")
    
    if run_sim or "sim_accumulation" in st.session_state:
        if run_sim:
            sim_engine = SandSimulation2D(dem_matrix)
            progress_bar = st.progress(0)
            
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
            
        fig_acc, ax_acc = plt.subplots(figsize=(10, 5), facecolor='#FDFCF8')
        ax_acc.set_facecolor('#FDFCF8')
        c = ax_acc.imshow(accumulation, cmap='seismic', vmin=-1.5, vmax=1.5)
        cbar = plt.colorbar(c, ax=ax_acc, label="Sand Accumulation (+) / Erosion (-) [Meters]")
        cbar.ax.yaxis.label.set_color('#2C2C24')
        cbar.ax.tick_params(colors='#2C2C24')
        
        ax_acc.scatter(road_x, road_y, c='#2C2C24', s=2, alpha=0.5, label='Highway Path')
        ax_acc.set_title(f"Cumulative Bed Height Elevation Change after {sim_days} Days", color='#2C2C24', fontweight='bold')
        ax_acc.tick_params(colors='#2C2C24')
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
        
        hazard_threshold = 0.5
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Max Sand Depth on Road", f"{max_road_sand:.2f} Meters")
        c2.metric("Hazard Threshold", f"{hazard_threshold} Meters")
        
        if max_road_sand > hazard_threshold:
            c3.error("CRITICAL HAZARD ALARM: Road Encroachment Detected!")
        else:
            c3.success("ROAD CLEAR: Highway Protected!")

        fig_prof, ax_prof = plt.subplots(figsize=(10, 3.5), facecolor='#FDFCF8')
        ax_prof.set_facecolor('#FDFCF8')
        ax_prof.plot(road_sand_depth, color='#A85448', linewidth=2, label='Sand Accumulation along Highway Cells')
        ax_prof.axhline(hazard_threshold, color='#2C2C24', linestyle='--', label='Critical Hazard Level (0.5m)')
        ax_prof.set_ylabel("Depth (m)", color='#2C2C24')
        ax_prof.set_xlabel("Highway Cell Index", color='#2C2C24')
        ax_prof.set_title("Sand Accumulation Profile along Highway", color='#2C2C24', fontweight='bold')
        ax_prof.tick_params(colors='#2C2C24')
        ax_prof.legend(facecolor='#FEFEFA', edgecolor='#DED8CF')
        st.pyplot(fig_prof)
    else:
        st.info("Please click 'Run Physics Simulation' in Tab 2 to run the hazard evaluation.")

# -----------------------------------------------------------------------------
# TAB 4: MACROECONOMIC & GDP ANALYTICS
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Egyptian Silica Sand Industrialization & Macroeconomic Impact")
    st.markdown("""
    *Evaluating the economic impact of industrializing Sinai's raw silica sand into high-value manufactured **Float & Solar Panel Glass**.*
    """)
    
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
    
    st.markdown("### Annual Financial Return Metrics")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Raw Sand Revenue", f"${econ_results['raw_revenue_usd']/1e6:.1f} Million")
    kpi2.metric("Manufactured Glass Revenue", f"${econ_results['glass_revenue_usd']/1e6:.1f} Million")
    kpi3.metric("Net Annual Value-Add", f"+${econ_results['net_value_add_usd']/1e6:.1f} Million")
    kpi4.metric("Value Addition Multiplier", f"{econ_results['value_multiplier']:.1f}x")

    st.success(f"Direct GDP Impact: Industrializing this silica sand volume adds +{econ_results['gdp_boost_percentage']:.3f}% directly to Egypt's Total GDP ($380 Billion USD baseline) every year!")

    df_proj = econ_engine.generate_10year_projection(initial_tonnage=sand_tonnage)
    
    fig_plotly = px.bar(
        df_proj, 
        x='Year', 
        y=['Raw Sand Export ($M)', 'Net Value Addition ($M)'],
        title="10-Year Projected Revenue Comparison for Egypt ($ Millions USD)",
        barmode='stack',
        color_discrete_sequence=['#C18C5D', '#5D7052']
    )
    fig_plotly.update_layout(
        paper_bgcolor='#FDFCF8',
        plot_bgcolor='#FDFCF8',
        font_family='Nunito',
        font_color='#2C2C24'
    )
    st.plotly_chart(fig_plotly, use_container_width=True)

# -----------------------------------------------------------------------------
# DOCKED FOOTER (COPYRIGHT & DEVELOPER LINKS)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="clean-footer">
    <div class="footer-brand">© 2026 AeroDune</div>
    <div class="footer-subtitle">A Physics-Based Spatiotemporal Simulation Framework for Wind-Blown Sand Encroachment, Windbreak Optimization, and Desert Resource Economics</div>
    <div class="footer-dev">Developed by Mohamed Ghanem</div>
    <div class="footer-links-row">
        <a href="https://github.com/mooghanem" target="_blank" class="footer-btn">GitHub Profile</a>
        <a href="https://www.linkedin.com/in/mohamedkhaledghanem" target="_blank" class="footer-btn">LinkedIn Profile</a>
    </div>
</div>
""", unsafe_allow_html=True)
