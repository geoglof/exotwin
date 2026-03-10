"""
ExoTwin: Digital Twin Dashboard for Exoplanet Habitability
Interactive Streamlit application for exploring exoplanet habitability predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

st.set_page_config(
    page_title="ExoTwin",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'exoplanets_features.csv')


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def make_gauge(score):
    """Create a habitability score gauge chart."""
    if score >= 0.7:
        color = "#2ecc71"
    elif score >= 0.4:
        color = "#f1c40f"
    else:
        color = "#e74c3c"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': '', 'font': {'size': 48}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 0.3], 'color': '#fadbd8'},
                {'range': [0.3, 0.6], 'color': '#fef9e7'},
                {'range': [0.6, 1.0], 'color': '#d5f5e3'},
            ],
        },
        title={'text': "Habitability Score"},
    ))
    fig.update_layout(height=300, margin=dict(t=60, b=20, l=40, r=40))
    return fig


def make_radar(params, planet_name="Your Planet"):
    """Create radar chart comparing planet to Earth."""
    categories = list(params.keys())
    earth_vals = [1.0] * len(categories)  # normalized to Earth = 1

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(params.values()), theta=categories,
        fill='toself', name=planet_name, opacity=0.6,
        line=dict(color='#3498db'),
    ))
    fig.add_trace(go.Scatterpolar(
        r=earth_vals, theta=categories,
        fill='toself', name='Earth', opacity=0.4,
        line=dict(color='#2ecc71'),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(max(params.values()), 1.5)])),
        showlegend=True, height=400, margin=dict(t=40, b=40),
    )
    return fig


# ─── SIDEBAR ───────────────────────────────────────────────
st.sidebar.title("ExoTwin")
st.sidebar.markdown("### Digital Twin for Exoplanet Habitability")
st.sidebar.markdown("---")

mode = st.sidebar.radio("Mode", ["🔍 Explore Database", "🎛️ Custom Planet", "🔬 What-If Simulation"])

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data:** [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)  \n"
    "**Team:** HACK-4-SAGES 2026"
)


# ─── MAIN ──────────────────────────────────────────────────
df = load_data()

if mode == "🔍 Explore Database":
    st.title("🔍 Explore Exoplanet Database")
    st.markdown(f"**{len(df)}** confirmed exoplanets from NASA Exoplanet Archive")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Planets", f"{len(df):,}")
    col2.metric("In Habitable Zone", f"{int(df['in_hz'].sum()):,}")
    col3.metric("With ESI", f"{df['esi'].notna().sum():,}")
    col4.metric("Score > 0.5", f"{(df['habitability_score'] > 0.5).sum()}")

    st.markdown("### Top Habitable Candidates")
    top = df.nlargest(20, 'habitability_score')[
        ['pl_name', 'habitability_score', 'esi', 'in_hz', 'pl_eqt', 'pl_rade', 'pl_bmasse', 'is_rocky']
    ].reset_index(drop=True)
    top.index += 1
    st.dataframe(top, use_container_width=True)

    st.markdown("### Mass vs Radius (colored by Habitability)")
    scatter_data = df.dropna(subset=['pl_bmasse', 'pl_rade', 'habitability_score'])
    fig = px.scatter(
        scatter_data, x='pl_bmasse', y='pl_rade',
        color='habitability_score', color_continuous_scale='RdYlGn',
        hover_name='pl_name', log_x=True, log_y=True,
        labels={'pl_bmasse': 'Mass (Earth masses)', 'pl_rade': 'Radius (Earth radii)'},
    )
    fig.add_scatter(x=[1], y=[1], mode='markers',
                    marker=dict(size=15, color='blue', symbol='star'), name='Earth')
    st.plotly_chart(fig, use_container_width=True)

elif mode == "🎛️ Custom Planet":
    st.title("🎛️ Custom Planet Twin")
    st.markdown("Define a hypothetical planet and predict its habitability.")
    st.markdown("*Model will be loaded here after training (Day 2).*")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Planetary Parameters")
        mass = st.slider("Mass (Earth masses)", 0.1, 50.0, 1.0, 0.1)
        radius = st.slider("Radius (Earth radii)", 0.3, 15.0, 1.0, 0.1)
        period = st.slider("Orbital Period (days)", 1.0, 1000.0, 365.0, 1.0)
        sma = st.slider("Semi-major Axis (AU)", 0.01, 10.0, 1.0, 0.01)
        ecc = st.slider("Orbital Eccentricity", 0.0, 0.9, 0.02, 0.01)

    with col2:
        st.markdown("#### Stellar Parameters")
        star_temp = st.slider("Star Temperature (K)", 2500, 10000, 5778, 50)
        star_lum = st.slider("Star Luminosity (log Solar)", -3.0, 2.0, 0.0, 0.05)
        star_mass = st.slider("Star Mass (Solar)", 0.1, 5.0, 1.0, 0.05)
        star_rad = st.slider("Star Radius (Solar)", 0.1, 5.0, 1.0, 0.05)

    st.markdown("---")
    st.info("⏳ ML model prediction will appear here after Day 2 training. "
            "For now, showing parameter comparison with Earth.")

    radar_params = {
        "Mass": mass / 1.0,
        "Radius": radius / 1.0,
        "Orbital Period": min(period / 365.25, 3),
        "Distance": min(sma / 1.0, 3),
        "Star Temp": star_temp / 5778,
    }

    col_gauge, col_radar = st.columns(2)
    with col_gauge:
        st.markdown("#### Habitability Score")
        st.markdown("*Placeholder — ML model coming Day 2*")
        st.plotly_chart(make_gauge(0.0), use_container_width=True)
    with col_radar:
        st.markdown("#### Comparison with Earth")
        st.plotly_chart(make_radar(radar_params, "Custom Planet"), use_container_width=True)

elif mode == "🔬 What-If Simulation":
    st.title("🔬 What-If Simulation")
    st.markdown("Select a known exoplanet and explore how changing parameters affects habitability.")
    st.markdown("*Full simulation engine coming Day 3.*")

    candidates = df.nlargest(50, 'habitability_score')['pl_name'].tolist()
    selected = st.selectbox("Select a planet", candidates)

    if selected:
        planet = df[df['pl_name'] == selected].iloc[0]
        st.markdown(f"### {selected}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Habitability", f"{planet['habitability_score']:.3f}")
        col2.metric("ESI", f"{planet['esi']:.3f}" if pd.notna(planet['esi']) else "N/A")
        col3.metric("In HZ", "Yes ✓" if planet['in_hz'] == 1 else "No ✗")
        col4.metric("Rocky", "Yes ✓" if planet['is_rocky'] == 1 else "No ✗")

        st.markdown("#### All Parameters")
        params_display = planet[['pl_bmasse', 'pl_rade', 'pl_orbper', 'pl_orbsmax',
                                  'pl_eqt', 'st_teff', 'st_lum', 'escape_velocity',
                                  'stellar_flux', 'esi', 'in_hz', 'habitability_score']].to_frame('Value')
        st.dataframe(params_display, use_container_width=True)
