"""
ExoTwin — Digital Twin for Exoplanet Habitability
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from features import (compute_density, compute_escape_velocity, compute_stellar_luminosity,
                       compute_stellar_flux, compute_habitable_zone, compute_esi,
                       compute_is_rocky, compute_equilibrium_temp,
                       compute_spectral_suitability, compute_stellar_suitability, EARTH)

st.set_page_config(page_title="ExoTwin", layout="wide", initial_sidebar_state="expanded")

C_BG = "#0d1117"
C_SURFACE = "#161b22"
C_PRIMARY = "#58a6ff"
C_SECONDARY = "#79c0ff"
C_ACCENT = "#f78166"
C_TEXT = "#c9d1d9"
C_MUTED = "#8b949e"
C_BORDER = "#30363d"
C_GREEN = "#3fb950"
C_RED = "#f85149"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=C_TEXT, size=13),
    margin=dict(t=36, b=36, l=48, r=24),
    coloraxis_colorbar=dict(
        tickfont=dict(color=C_MUTED),
        title_font=dict(color=C_MUTED),
    ),
)

FEATURE_COLS = [
    'pl_bmasse', 'pl_rade', 'pl_orbper', 'pl_orbsmax', 'pl_orbeccen',
    'pl_eqt', 'pl_dens', 'st_teff', 'st_lum', 'st_mass', 'st_rad',
    'escape_velocity', 'stellar_flux', 'in_hz', 'is_rocky', 'esi',
    'orbit_stability', 'spectral_suitability', 'stellar_suitability',
]

st.markdown(f"""<style>
    .block-container {{ padding-top: 1.5rem; }}
    section[data-testid="stSidebar"] {{ background-color: {C_SURFACE}; border-right: 1px solid {C_BORDER}; }}
    section[data-testid="stSidebar"] .stRadio label {{ font-size: 14px; }}
    .stMetric {{ background: {C_SURFACE}; border: 1px solid {C_BORDER}; border-radius: 6px; padding: 12px 16px; }}
    div[data-testid="stMetricValue"] {{ font-size: 1.6rem; }}
    div[data-testid="stMetricLabel"] {{ font-size: 0.8rem; color: {C_MUTED}; }}
    .score-block {{ background: {C_SURFACE}; border: 1px solid {C_BORDER}; border-radius: 6px; padding: 20px 24px; }}
    .score-value {{ font-size: 2.4rem; font-weight: 600; font-variant-numeric: tabular-nums; }}
    .score-label {{ font-size: 0.8rem; color: {C_MUTED}; margin-bottom: 4px; }}
</style>""", unsafe_allow_html=True)


BASE = os.path.join(os.path.dirname(__file__), '..')
DATA_PATH = os.path.join(BASE, 'data', 'processed', 'exoplanets_final.csv')
MODEL_PATH = os.path.join(BASE, 'models', 'best_model.pkl')
IMPUTER_PATH = os.path.join(BASE, 'models', 'imputer.pkl')


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    return model, imputer


def predict_habitability(model, imputer, params_dict):
    """Predict habitability from a dict of raw planet/star parameters."""
    row = pd.DataFrame([{
        'pl_bmasse': params_dict['mass'],
        'pl_rade': params_dict['radius'],
        'pl_orbper': params_dict['period'],
        'pl_orbsmax': params_dict['sma'],
        'pl_orbeccen': params_dict['ecc'],
        'pl_eqt': None,
        'pl_dens': None,
        'st_teff': params_dict['star_temp'],
        'st_lum': params_dict['star_lum'],
        'st_mass': params_dict['star_mass'],
        'st_rad': params_dict['star_rad'],
        'pl_name': 'custom',
    }])

    row = compute_density(row)
    row = compute_equilibrium_temp(row)
    row = compute_escape_velocity(row)
    row = compute_stellar_flux(row)
    row = compute_habitable_zone(row)
    row = compute_is_rocky(row)
    row = compute_esi(row)
    row = compute_spectral_suitability(row)
    row = compute_stellar_suitability(row)

    X = row[FEATURE_COLS]
    X_imp = pd.DataFrame(imputer.transform(X), columns=FEATURE_COLS)
    score = float(model.predict(X_imp)[0])
    derived = {
        'density': row['pl_dens'].iloc[0],
        'eq_temp': row['pl_eqt'].iloc[0],
        'escape_vel': row['escape_velocity'].iloc[0],
        'flux': row['stellar_flux'].iloc[0],
        'in_hz': row['in_hz'].iloc[0] if pd.notna(row['in_hz'].iloc[0]) else 0,
        'is_rocky': row['is_rocky'].iloc[0] if pd.notna(row['is_rocky'].iloc[0]) else 0,
        'esi': row['esi'].iloc[0],
        'orbit_stability': row['orbit_stability'].iloc[0] if pd.notna(row.get('orbit_stability', pd.Series([np.nan])).iloc[0]) else 0,
        'spectral': row['spectral_suitability'].iloc[0] if pd.notna(row.get('spectral_suitability', pd.Series([np.nan])).iloc[0]) else 0,
        'stellar': row['stellar_suitability'].iloc[0] if pd.notna(row.get('stellar_suitability', pd.Series([np.nan])).iloc[0]) else 0,
    }
    return max(0.0, min(1.0, score)), derived


def score_color(val):
    if val >= 0.6:
        return C_GREEN
    if val >= 0.3:
        return C_PRIMARY
    return C_MUTED


def make_score_bar(score, label="Habitability"):
    color = score_color(score)
    pct = max(0, min(100, score * 100))
    return f"""
    <div class="score-block">
        <div class="score-label">{label}</div>
        <div class="score-value" style="color:{color}">{score:.3f}</div>
        <div style="background:{C_BORDER};border-radius:3px;height:6px;margin-top:8px;">
            <div style="background:{color};width:{pct}%;height:6px;border-radius:3px;"></div>
        </div>
    </div>
    """


def make_radar(params, planet_name="Planet"):
    categories = list(params.keys())
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(params.values()), theta=categories,
        fill='toself', name=planet_name, opacity=0.5,
        line=dict(color=C_PRIMARY, width=1.5),
        fillcolor="rgba(88,166,255,0.12)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[1.0] * len(categories), theta=categories,
        fill='toself', name='Earth', opacity=0.3,
        line=dict(color=C_GREEN, width=1.5),
        fillcolor="rgba(63,185,80,0.08)",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, max(max(params.values()), 1.5)],
                            gridcolor=C_BORDER, tickfont=dict(color=C_MUTED, size=10)),
            angularaxis=dict(gridcolor=C_BORDER, tickfont=dict(color=C_TEXT, size=11)),
        ),
        showlegend=True, legend=dict(font=dict(size=11)), height=360,
    )
    return fig


# ── Load data and model ──
df = load_data()
model, imputer = load_model()

# ── Sidebar ──
st.sidebar.markdown("**ExoTwin**")
st.sidebar.caption("Digital twin for exoplanet habitability")
mode = st.sidebar.radio("Navigation", ["Database", "Custom Planet", "What-If"])
st.sidebar.markdown(f"<br><span style='color:{C_MUTED};font-size:12px;'>"
                    f"{len(df):,} planets &middot; NASA + PHL data</span>",
                    unsafe_allow_html=True)


# ═══════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════
if mode == "Database":
    st.markdown(f"<h2 style='margin-bottom:4px;'>Exoplanet database</h2>"
                f"<span style='color:{C_MUTED};font-size:14px;'>"
                f"{len(df):,} confirmed planets &middot; "
                f"{int(df['in_hz'].sum())} in habitable zone &middot; "
                f"{(df['habitability_score'] > 0.5).sum()} with score above 0.5</span>",
                unsafe_allow_html=True)

    tab_table, tab_scatter, tab_hz, tab_3d = st.tabs(["Candidates", "Mass / Radius", "Habitable Zone", "3D Explorer"])

    with tab_table:
        n_show = st.slider("Show top N planets", 10, 100, 25, 5)
        top = df.nlargest(n_show, 'habitability_score')[
            ['pl_name', 'habitability_score', 'esi', 'in_hz', 'pl_eqt',
             'pl_rade', 'pl_bmasse', 'is_rocky']
        ].reset_index(drop=True)
        top.index += 1
        top.columns = ['Name', 'Score', 'ESI', 'In HZ', 'Eq. Temp (K)',
                        'Radius (R Earth)', 'Mass (M Earth)', 'Rocky']
        st.dataframe(top, use_container_width=True, height=min(36 * n_show + 38, 600))

    with tab_scatter:
        scatter_df = df.dropna(subset=['pl_bmasse', 'pl_rade', 'habitability_score'])
        fig = px.scatter(
            scatter_df, x='pl_bmasse', y='pl_rade',
            color='habitability_score',
            color_continuous_scale=[[0, C_BORDER], [0.4, C_MUTED], [0.7, C_PRIMARY], [1.0, C_GREEN]],
            hover_name='pl_name', log_x=True, log_y=True,
            labels={'pl_bmasse': 'Mass (Earth masses)', 'pl_rade': 'Radius (Earth radii)',
                    'habitability_score': 'Score'},
        )
        fig.add_scatter(
            x=[1], y=[1], mode='markers+text', text=["Earth"], textposition="top center",
            marker=dict(size=10, color=C_GREEN, symbol='diamond'),
            textfont=dict(color=C_GREEN, size=11), name='Earth', showlegend=False,
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7), selector=dict(mode='markers'))
        fig.update_layout(**PLOTLY_LAYOUT, height=520,
                          xaxis=dict(gridcolor=C_BORDER), yaxis=dict(gridcolor=C_BORDER))
        st.plotly_chart(fig, use_container_width=True)

    with tab_hz:
        hz_df = df.dropna(subset=['pl_orbsmax', 'st_teff', 'hz_inner', 'hz_outer', 'in_hz'])
        fig = go.Figure()
        out = hz_df[hz_df['in_hz'] == 0]
        inside = hz_df[hz_df['in_hz'] == 1]
        fig.add_trace(go.Scatter(
            x=out['st_teff'], y=out['pl_orbsmax'], mode='markers',
            marker=dict(size=3, color=C_MUTED, opacity=0.15),
            name='Outside HZ', text=out['pl_name'], hoverinfo='text+x+y',
        ))
        fig.add_trace(go.Scatter(
            x=inside['st_teff'], y=inside['pl_orbsmax'], mode='markers',
            marker=dict(size=6, color=C_GREEN, opacity=0.8),
            name='Inside HZ', text=inside['pl_name'], hoverinfo='text+x+y',
        ))
        sorted_hz = hz_df.sort_values('st_teff')
        fig.add_trace(go.Scatter(
            x=sorted_hz['st_teff'], y=sorted_hz['hz_inner'], mode='lines',
            line=dict(color=C_ACCENT, width=1, dash='dot'), name='HZ inner', opacity=0.5,
        ))
        fig.add_trace(go.Scatter(
            x=sorted_hz['st_teff'], y=sorted_hz['hz_outer'], mode='lines',
            line=dict(color=C_SECONDARY, width=1, dash='dot'), name='HZ outer', opacity=0.5,
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT, height=520,
            xaxis=dict(title="Stellar temperature (K)", gridcolor=C_BORDER),
            yaxis=dict(title="Semi-major axis (AU)", type='log', gridcolor=C_BORDER),
            legend=dict(font=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_3d:
        scatter3d_df = df.dropna(subset=['pl_bmasse', 'pl_rade', 'pl_eqt', 'habitability_score'])
        fig3d = px.scatter_3d(
            scatter3d_df,
            x='pl_bmasse', y='pl_rade', z='pl_eqt',
            color='habitability_score',
            color_continuous_scale=[[0, C_BORDER], [0.3, C_MUTED], [0.6, C_PRIMARY], [1.0, C_GREEN]],
            hover_name='pl_name',
            hover_data={'pl_bmasse': ':.2f', 'pl_rade': ':.2f', 'pl_eqt': ':.0f',
                        'habitability_score': ':.3f', 'in_hz': True},
            log_x=True, log_y=True,
            labels={'pl_bmasse': 'Mass (M⊕)', 'pl_rade': 'Radius (R⊕)',
                    'pl_eqt': 'Eq. Temperature (K)', 'habitability_score': 'Score'},
            opacity=0.7,
        )
        fig3d.add_scatter3d(
            x=[1], y=[1], z=[255], mode='markers+text', text=["Earth"],
            marker=dict(size=6, color=C_GREEN, symbol='diamond'),
            textfont=dict(color=C_GREEN, size=12), name='Earth', showlegend=False,
        )
        fig3d.update_traces(marker=dict(size=3), selector=dict(type='scatter3d', mode='markers'))
        fig3d.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=C_TEXT, size=12),
            height=620,
            scene=dict(
                xaxis=dict(title="Mass (M⊕, log)", backgroundcolor="rgba(0,0,0,0)",
                           gridcolor=C_BORDER, showbackground=True),
                yaxis=dict(title="Radius (R⊕, log)", backgroundcolor="rgba(0,0,0,0)",
                           gridcolor=C_BORDER, showbackground=True),
                zaxis=dict(title="Eq. Temp (K)", backgroundcolor="rgba(0,0,0,0)",
                           gridcolor=C_BORDER, showbackground=True),
                bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(t=8, b=8, l=8, r=8),
        )
        st.plotly_chart(fig3d, use_container_width=True)
        st.markdown(f"<span style='color:{C_MUTED};font-size:12px;'>"
                    f"{len(scatter3d_df):,} planets with mass, radius, and temperature data. "
                    f"Drag to rotate, scroll to zoom. Earth marked as green diamond.</span>",
                    unsafe_allow_html=True)


# ═══════════════════════════════════════════
# CUSTOM PLANET
# ═══════════════════════════════════════════
elif mode == "Custom Planet":
    st.markdown("<h2 style='margin-bottom:4px;'>Custom planet twin</h2>", unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown(f"<span style='color:{C_MUTED};font-size:13px;'>Planet</span>", unsafe_allow_html=True)
        mass = st.slider("Mass (Earth masses)", 0.1, 50.0, 1.0, 0.1)
        radius = st.slider("Radius (Earth radii)", 0.3, 15.0, 1.0, 0.1)
        period = st.slider("Orbital period (days)", 1.0, 1000.0, 365.0, 1.0)
        sma = st.slider("Semi-major axis (AU)", 0.01, 10.0, 1.0, 0.01)
        ecc = st.slider("Eccentricity", 0.0, 0.9, 0.02, 0.01)

        st.markdown(f"<span style='color:{C_MUTED};font-size:13px;'>Host star</span>", unsafe_allow_html=True)
        star_temp = st.slider("Temperature (K)", 2500, 10000, 5778, 50)
        star_lum = st.slider("Luminosity (log Solar)", -3.0, 2.0, 0.0, 0.05)
        star_mass = st.slider("Mass (Solar)", 0.1, 5.0, 1.0, 0.05)
        star_rad = st.slider("Radius (Solar)", 0.1, 5.0, 1.0, 0.05)

    with col_result:
        params = dict(mass=mass, radius=radius, period=period, sma=sma, ecc=ecc,
                      star_temp=star_temp, star_lum=star_lum, star_mass=star_mass, star_rad=star_rad)
        score, derived = predict_habitability(model, imputer, params)

        st.markdown(make_score_bar(score, "Predicted habitability"), unsafe_allow_html=True)

        st.markdown(f"""<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:16px;">
            <div class="score-block"><div class="score-label">ESI</div>
                <div style="font-size:1.2rem;font-weight:600;">{f"{derived['esi']:.3f}" if pd.notna(derived['esi']) else '—'}</div></div>
            <div class="score-block"><div class="score-label">In HZ</div>
                <div style="font-size:1.2rem;font-weight:600;color:{C_GREEN if derived['in_hz'] else C_MUTED}">{'Yes' if derived['in_hz'] else 'No'}</div></div>
            <div class="score-block"><div class="score-label">Rocky</div>
                <div style="font-size:1.2rem;font-weight:600;color:{C_GREEN if derived['is_rocky'] else C_MUTED}">{'Yes' if derived['is_rocky'] else 'No'}</div></div>
            <div class="score-block"><div class="score-label">Eq. temp</div>
                <div style="font-size:1.2rem;font-weight:600;">{derived['eq_temp']:.0f} K</div></div>
            <div class="score-block"><div class="score-label">Esc. velocity</div>
                <div style="font-size:1.2rem;font-weight:600;">{derived['escape_vel']:.1f} km/s</div></div>
            <div class="score-block"><div class="score-label">Density</div>
                <div style="font-size:1.2rem;font-weight:600;">{derived['density']:.2f} g/cm3</div></div>
            <div class="score-block"><div class="score-label">Orbit stability</div>
                <div style="font-size:1.2rem;font-weight:600;">{derived['orbit_stability']:.2f}</div></div>
            <div class="score-block"><div class="score-label">Spectral suit.</div>
                <div style="font-size:1.2rem;font-weight:600;">{derived['spectral']:.2f}</div></div>
            <div class="score-block"><div class="score-label">Stellar suit.</div>
                <div style="font-size:1.2rem;font-weight:600;">{derived['stellar']:.2f}</div></div>
        </div>""", unsafe_allow_html=True)

        radar_params = {
            "Mass": mass / EARTH['mass'],
            "Radius": radius / EARTH['radius'],
            "Density": (derived['density'] / EARTH['density']) if pd.notna(derived['density']) else 0,
            "Esc. vel": (derived['escape_vel'] / EARTH['escape_vel']) if pd.notna(derived['escape_vel']) else 0,
            "Star temp": star_temp / 5778,
            "Spectral": derived['spectral'],
            "Stellar": derived['stellar'],
        }
        st.plotly_chart(make_radar(radar_params, "Custom"), use_container_width=True)


# ═══════════════════════════════════════════
# WHAT-IF SIMULATION
# ═══════════════════════════════════════════
elif mode == "What-If":
    st.markdown("<h2 style='margin-bottom:4px;'>What-if simulation</h2>", unsafe_allow_html=True)

    candidates = df.nlargest(50, 'habitability_score')['pl_name'].tolist()
    selected = st.selectbox("Planet", candidates, label_visibility="collapsed",
                            help="Top 50 candidates by habitability score")

    if selected:
        planet = df[df['pl_name'] == selected].iloc[0]
        original_score = planet['habitability_score']

        if st.session_state.get('_whatif_planet') != selected:
            st.session_state['_whatif_planet'] = selected
            for k in ['wm', 'wr', 'wp', 'ws', 'we', 'wst', 'wsl', 'wsm', 'wsr']:
                st.session_state.pop(k, None)
            st.rerun()

        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown(f"<span style='color:{C_MUTED};font-size:13px;'>Modify parameters for {selected}</span>",
                        unsafe_allow_html=True)

            def safe_val(col, default, as_float=True):
                v = planet[col]
                return float(v) if pd.notna(v) else default

            mass = st.slider("Mass (Earth masses)", 0.1, 50.0, safe_val('pl_bmasse', 1.0), 0.1, key="wm")
            radius = st.slider("Radius (Earth radii)", 0.3, 15.0, safe_val('pl_rade', 1.0), 0.1, key="wr")
            period = st.slider("Orbital period (days)", 1.0, 1000.0,
                               min(safe_val('pl_orbper', 365.0), 1000.0), 1.0, key="wp")
            sma = st.slider("Semi-major axis (AU)", 0.01, 10.0,
                            min(safe_val('pl_orbsmax', 1.0), 10.0), 0.01, key="ws")
            ecc = st.slider("Eccentricity", 0.0, 0.9, safe_val('pl_orbeccen', 0.02), 0.01, key="we")
            star_temp = st.slider("Star temperature (K)", 2500, 10000,
                                  int(safe_val('st_teff', 5778)), 50, key="wst")
            star_lum = st.slider("Star luminosity (log Solar)", -3.0, 2.0,
                                 safe_val('st_lum', 0.0), 0.05, key="wsl")
            star_mass = st.slider("Star mass (Solar)", 0.1, 5.0,
                                  safe_val('st_mass', 1.0), 0.05, key="wsm")
            star_rad = st.slider("Star radius (Solar)", 0.1, 5.0,
                                 safe_val('st_rad', 1.0), 0.05, key="wsr")

        with col_right:
            params = dict(mass=mass, radius=radius, period=period, sma=sma, ecc=ecc,
                          star_temp=star_temp, star_lum=star_lum, star_mass=star_mass, star_rad=star_rad)
            new_score, derived = predict_habitability(model, imputer, params)
            delta = new_score - original_score

            st.markdown(make_score_bar(original_score, f"Original — {selected}"), unsafe_allow_html=True)

            delta_color = C_GREEN if delta > 0.01 else (C_RED if delta < -0.01 else C_MUTED)
            delta_sign = "+" if delta >= 0 else ""
            st.markdown(f"""<div style="margin-top:10px;display:flex;gap:8px;">
                <div style="flex:1;padding:10px 14px;border:1px solid {C_BORDER};border-radius:6px;">
                    <div style="color:{C_MUTED};font-size:0.75rem;margin-bottom:2px;">Modified score</div>
                    <div style="font-size:1.4rem;font-weight:600;color:{score_color(new_score)};">{new_score:.3f}</div>
                </div>
                <div style="flex:1;padding:10px 14px;border:1px solid {C_BORDER};border-radius:6px;">
                    <div style="color:{C_MUTED};font-size:0.75rem;margin-bottom:2px;">Change</div>
                    <div style="font-size:1.4rem;font-weight:600;color:{delta_color};">{delta_sign}{delta:.4f}</div>
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:16px;">
                <div class="score-block"><div class="score-label">ESI</div>
                    <div style="font-size:1.2rem;font-weight:600;">{f"{derived['esi']:.3f}" if pd.notna(derived['esi']) else '—'}</div></div>
                <div class="score-block"><div class="score-label">In HZ</div>
                    <div style="font-size:1.2rem;font-weight:600;color:{C_GREEN if derived['in_hz'] else C_MUTED}">{'Yes' if derived['in_hz'] else 'No'}</div></div>
                <div class="score-block"><div class="score-label">Rocky</div>
                    <div style="font-size:1.2rem;font-weight:600;color:{C_GREEN if derived['is_rocky'] else C_MUTED}">{'Yes' if derived['is_rocky'] else 'No'}</div></div>
                <div class="score-block"><div class="score-label">Orbit stability</div>
                    <div style="font-size:1.2rem;font-weight:600;">{derived['orbit_stability']:.2f}</div></div>
                <div class="score-block"><div class="score-label">Spectral suit.</div>
                    <div style="font-size:1.2rem;font-weight:600;">{derived['spectral']:.2f}</div></div>
                <div class="score-block"><div class="score-label">Stellar suit.</div>
                    <div style="font-size:1.2rem;font-weight:600;">{derived['stellar']:.2f}</div></div>
            </div>""", unsafe_allow_html=True)

            radar_params = {
                "Mass": mass / EARTH['mass'],
                "Radius": radius / EARTH['radius'],
                "Density": (derived['density'] / EARTH['density']) if pd.notna(derived['density']) else 0,
                "Esc. vel": (derived['escape_vel'] / EARTH['escape_vel']) if pd.notna(derived['escape_vel']) else 0,
                "Star temp": star_temp / 5778,
                "Spectral": derived['spectral'],
                "Stellar": derived['stellar'],
            }
            st.plotly_chart(make_radar(radar_params, selected), use_container_width=True)

            # Sensitivity analysis: vary each parameter ±20% and measure impact
            st.markdown(f"<div style='color:{C_MUTED};font-size:13px;margin-top:8px;'>Sensitivity analysis</div>",
                        unsafe_allow_html=True)

            param_defs = [
                ("Mass",     "mass",      mass),
                ("Radius",   "radius",    radius),
                ("Period",   "period",    period),
                ("Distance", "sma",       sma),
                ("Ecc.",     "ecc",       ecc),
                ("Star T",   "star_temp", star_temp),
                ("Star L",   "star_lum",  star_lum),
                ("Star M",   "star_mass", star_mass),
                ("Star R",   "star_rad",  star_rad),
            ]

            sens_names, sens_vals = [], []
            base = params.copy()
            for label, key, val in param_defs:
                if key == "star_lum":
                    up_val = val + 0.08
                elif val == 0 or (key == "ecc" and val < 0.01):
                    up_val = val + 0.1
                else:
                    up_val = val * 1.20
                tweaked = base.copy()
                tweaked[key] = up_val
                sc_up, _ = predict_habitability(model, imputer, tweaked)
                sens_names.append(label)
                sens_vals.append(sc_up - new_score)

            sorted_idx = sorted(range(len(sens_vals)), key=lambda i: abs(sens_vals[i]), reverse=True)
            s_names = [sens_names[i] for i in sorted_idx]
            s_vals = [sens_vals[i] for i in sorted_idx]
            s_colors = [C_GREEN if v > 0 else C_RED if v < 0 else C_MUTED for v in s_vals]

            fig_sens = go.Figure(go.Bar(
                x=s_vals, y=s_names, orientation='h',
                marker_color=s_colors,
                text=[f"{v:+.4f}" for v in s_vals], textposition='outside',
                textfont=dict(size=11),
            ))
            sens_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k != 'margin'}
            fig_sens.update_layout(
                **sens_layout, height=260,
                xaxis=dict(title="Score change (+20%)", gridcolor=C_BORDER, zeroline=True,
                           zerolinecolor=C_MUTED, zerolinewidth=1),
                yaxis=dict(autorange="reversed"),
                margin=dict(t=8, b=36, l=70, r=50),
            )
            st.plotly_chart(fig_sens, use_container_width=True)
