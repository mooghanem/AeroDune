# AeroDune 🌍💨

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aerodune.streamlit.app/)

A physics-based spatiotemporal simulation framework for wind-blown sand (aeolian) morphodynamics, infrastructure hazard mitigation (Qantara–El Arish Highway, North Sinai), and national macroeconomic silica resource valuation.

---

## 🚀 Live Demo
Experience the interactive simulation framework live on Streamlit Cloud:
👉 **[Try AeroDune Live](https://aerodune.streamlit.app/)**

---

## 📌 Overview
**AeroDune** is an automated Python-based computational framework designed to transition desert management from reactive hazard containment to proactive spatial optimization. By fusing digital elevation models with real-time infrastructure vector data and fluid-sediment mechanics, the system accurately predicts sand accumulation bottlenecks and evaluates the macroeconomic impact of domestic silica sand industrialization.

---

## 🔑 Key Features
* **Spatiotemporal Aeolian Simulation:** Solves the coupled 2D Bagnold Saltation Law and Exner mass-balance continuity equations over high-resolution elevation grids.
* **Infrastructure Risk Integration:** Automatically rasterizes OpenStreetMap highway networks (Qantara–El Arish corridor) to monitor critical sand encroachment ($\Delta z \ge 0.5\text{ m}$).
* **Windbreak Optimization:** Simulates synthetic barrier porosity ($P = 0.30$) to force upstream spatial flux divergence and protect vulnerable transport routes.
* **Macroeconomic Silica Valuation Engine:** Quantifies the 14.4x value-add multiplier and direct GDP impact ($+\$1.17\text{ Billion USD/year}$) of processing North Sinai high-purity silica sand into solar and float glass.

---

## 🛠️ Technology Stack & Libraries
* **NumPy:** Vectorized finite-difference matrix calculations (`np.gradient`, `np.clip`).
* **Pandas:** Tabular climate and wind forcing data ingestion (NASA POWER API).
* **Rasterio:** Geospatial affine transformations and DEM matrix mosaicing (USGS SRTM).
* **OSMnx:** Real-world highway vector geometry fetching and grid rasterization (OpenStreetMap).
* **Streamlit & Plotly:** Interactive multi-panel scientific dashboard and financial visualization.

---

## ⚙️ Local Installation & Running

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mooghanem/aerodune.git](https://github.com/mooghanem/aerodune.git)
   cd aerodune
