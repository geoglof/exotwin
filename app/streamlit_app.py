"""
ExoTwin — Digital Twin for Exoplanet Habitability
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

st.set_page_config(page_title="ExoTwin", layout="wide", initial_sidebar_state="expanded")

# Void Space palette
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


DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'exoplanets_features.csv')


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def score_color(val):
    if val >= 0.6:
        return C_GREEN
    if val >= 0.3:
        return C_PRIMARY
    return C_MUTED


def make_score_bar(score, label="Habitability"):
    """Simple horizontal bar instead of a gauge."""
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
        fillcolor=f"rgba(88,166,255,0.12)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[1.0] * len(categories), theta=categories,
        fill='toself', name='Earth', opacity=0.3,
        line=dict(color=C_GREEN, width=1.5),
        fillcolor=f"rgba(63,185,80,0.08)",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, max(max(params.values()), 1.5)],
                            gridcolor=C_BORDER, tickfont=dict(color=C_MUTED, size=10)),
            angularaxis=dict(gridcolor=C_BORDER, tickfont=dict(color=C_TEXT, size=11)),
        ),
        showlegend=True,
        legend=dict(font=dict(size=11)),
        height=360,
    )
    return fig


df = load_data()

# Sidebar
st.sidebar.markdown("**ExoTwin**")
st.sidebar.caption("Digital twin for exoplanet habitability")

mode = st.sidebar.radio("Navigation", ["Database", "Custom Planet", "What-If"])

st.sidebar.markdown(f"<br><span style='color:{C_MUTED};font-size:12px;'>"
                    f"{len(df):,} planets from NASA Exoplanet Archive</span>",
                    unsafe_allow_html=True)


if mode == "Database":
    st.markdown(f"<h2 style='margin-bottom:4px;'>Exoplanet database</h2>"
                f"<span style='color:{C_MUTED};font-size:14px;'>"
                f"{len(df):,} confirmed planets &middot; "
                f"{int(df['in_hz'].sum())} in habitable zone &middot; "
                f"{(df['habitability_score'] > 0.5).sum()} with score above 0.5</span>",
                unsafe_allow_html=True)

    tab_table, tab_scatter, tab_hz = st.tabs(["Candidates", "Mass / Radius", "Habitable Zone"])

    with tab_table:
        n_show = st.slider("Show top N planets", 10, 100, 25, 5)
        top = df.nlargest(n_show, 'habitability_score')[
            ['pl_name', 'habitability_score', 'esi', 'in_hz', 'pl_eqt',
             'pl_rade', 'pl_bmasse', 'is_rocky']
        ].reset_index(drop=True)
        top.index += 1
        top.columns = ['Name', 'Score', 'ESI', 'In HZ', 'Eq. Temp (K)',
                        'Radius (R⊕)', 'Mass (M⊕)', 'Rocky']
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
        st.markdown(make_score_bar(0.0, "Habitability — awaiting model"), unsafe_allow_html=True)

        radar_params = {
            "Mass": mass,
            "Radius": radius,
            "Period": min(period / 365.25, 3),
            "Distance": min(sma, 3),
            "Star temp": star_temp / 5778,
        }
        st.plotly_chart(make_radar(radar_params, "Custom"), use_container_width=True)


elif mode == "What-If":
    st.markdown("<h2 style='margin-bottom:4px;'>What-if simulation</h2>", unsafe_allow_html=True)

    candidates = df.nlargest(50, 'habitability_score')['pl_name'].tolist()
    selected = st.selectbox("Planet", candidates, label_visibility="collapsed",
                            help="Top 50 candidates by habitability score")

    if selected:
        planet = df[df['pl_name'] == selected].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score", f"{planet['habitability_score']:.3f}")
        c2.metric("ESI", f"{planet['esi']:.3f}" if pd.notna(planet['esi']) else "—")
        c3.metric("In HZ", "Yes" if planet['in_hz'] == 1 else "No")
        c4.metric("Rocky", "Yes" if planet['is_rocky'] == 1 else "No")

        param_cols = ['pl_bmasse', 'pl_rade', 'pl_orbper', 'pl_orbsmax', 'pl_eqt',
                      'st_teff', 'st_lum', 'escape_velocity', 'stellar_flux',
                      'esi', 'in_hz', 'habitability_score']
        labels = ['Mass (M⊕)', 'Radius (R⊕)', 'Period (d)', 'Dist. (AU)', 'Eq. temp (K)',
                  'Star temp (K)', 'Star lum (log)', 'Esc. vel (km/s)', 'Flux (S⊕)',
                  'ESI', 'In HZ', 'Score']

        vals = planet[param_cols].values
        display_df = pd.DataFrame({'Parameter': labels, 'Value': vals})
        display_df['Value'] = display_df['Value'].apply(
            lambda v: f"{v:.4f}" if pd.notna(v) else "—"
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=460)
