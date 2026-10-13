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
# PAGE CONFIGURATION & ORGANIC DESIGN SYSTEM CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AeroDune | Physics-Based Sand Simulation & Economic Framework",
    page_icon="🏜️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Injection: Organic / Natural Design Tokens
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
        --shadow-moss: 0 8px 24px -4px rgba(93, 112, 82, 0.18);
        --shadow-clay: 0 10px 30px -6px rgba(193, 140, 93, 0.22);
    }

    /* Global Typography & Background */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg-paper) !important;
        color: var(--fg-loam) !important;
        font-family: 'Nunito', sans-serif !important;
    }

    /* Texture Overlay Simulation */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image: radial-gradient(#2C2C24 0.5px, transparent 0.5px);
        background-size: 24px 24px;
        opacity: 0.03;
        pointer-events: none;
        z-index: 1;
    }

    /* Headings (Fraunces Soft Serif) */
    h1, h2, h3, h4, .main-title {
        font-family: 'Fraunces', serif !important;
        color: var(--fg-loam) !important;
        font-weight: 700 !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: var(--muted-stone) !important;
        border-right: 1.5px solid var(--border-timber) !important;
        padding-top: 2rem !important;
    }

    /* Tab Navigation Pills */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.75rem !important;
        background-color: var(--accent-sand) !important;
        padding: 0.5rem !important;
        border-radius: 9999px !important;
        border: 1px solid var(--border-timber) !important;
    }

    .stTabs [data-baseweb="tab"] {
        height: 2.75rem !important;
        border-radius: 9999px !important;
        color: var(--fg-loam) !important;
        font-family: 'Nunito', sans-serif !important;
        font-weight: 700 !important;
        padding: 0 1.5rem !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--primary-moss) !important;
        color: var(--primary-mist) !important;
        box-shadow: var(--shadow-moss) !important;
    }

    /* Custom Organic Pill Buttons */
    .stButton>button {
        background-color: var(--primary-moss) !important;
        color: var(--primary-mist) !important;
        font-family: 'Nunito', sans-serif !important;
        font-weight: 700 !important;
        border-radius: 9999px !important;
        padding: 0.75rem 2.25rem !important;
        border: none !important;
        box-shadow: var(--shadow-moss) !important;
        transition: transform 0.25s ease, box-shadow 0.25s ease !important;
    }

    .stButton>button:hover {
        transform: scale(1.03) translateY(-2px) !important;
        box-shadow: var(--shadow-clay) !important;
        background-color: #4f6045 !important;
    }

    /* Organic Card Styling */
    [data-testid="stMetric"], .organic-card {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-timber) !important;
        border-radius: 1.75rem !important;
        padding:
