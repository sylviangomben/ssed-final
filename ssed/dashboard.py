"""
SSED Fusion Dashboard

The class deliverable: a Streamlit app that demonstrates
Sample Space Expansion Detection across all three layers.

Run: streamlit run ssed/dashboard.py
"""

import io
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssed.quant_signals import (
    run_quant_signals,
    fetch_prices,
    compute_hmm_signals,
    compute_entropy_signals,
    compute_divergence_signals,
    shannon_entropy,
    REGIME_LABELS,
)
from ssed.narrative_signals import (
    compute_news_signals,
)
from ssed.backtest import run_backtest, run_forward_looking_backtest
from ssed.sector_scanner import scan_sectors, scan_market_movers, SECTOR_ETFS

load_dotenv()

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SSED — Sample Space Expansion Detector",
    page_icon="🔬",
    layout="wide",
)

# ============================================================
# DESIGN SYSTEM — Colors, Fonts, Tokens
# ============================================================

C = {
    "bg":           "#0a0e1a",
    "bg2":          "#0f1526",
    "card":         "#111827",
    "card_border":  "#1e293b",
    "accent":       "#00d4ff",
    "accent2":      "#6366f1",
    "green":        "#00e396",
    "red":          "#ff4560",
    "yellow":       "#feb019",
    "text":         "#c8d6e5",
    "text_dim":     "#64748b",
    "text_bright":  "#f1f5f9",
    "white":        "#ffffff",
    "glow_accent":  "rgba(0,212,255,0.25)",
    "glow_green":   "rgba(0,227,150,0.25)",
    "glow_red":     "rgba(255,69,96,0.2)",
}

# ============================================================
# MASTER CSS
# ============================================================

st.markdown(f"""
<style>
    /* ── Import Inter font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    html {{ scroll-behavior: smooth; }}

    .stApp {{
        background: {C['bg']};
    }}

    /* ── Fade-in animation ── */
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to   {{ opacity: 1; }}
    }}
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50%      {{ opacity: 0.5; }}
    }}
    @keyframes glowPulse {{
        0%, 100% {{ box-shadow: 0 0 5px {C['glow_accent']}, 0 0 15px rgba(0,212,255,0.08); }}
        50%      {{ box-shadow: 0 0 12px {C['glow_accent']}, 0 0 30px rgba(0,212,255,0.15); }}
    }}
    @keyframes shimmer {{
        0%   {{ background-position: -200% 0; }}
        100% {{ background-position: 200% 0; }}
    }}

    /* ── Hide Streamlit boilerplate ── */
    footer {{visibility: hidden;}}
    #MainMenu {{visibility: hidden;}}
    header[data-testid="stHeader"] {{ background: transparent; }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #080c16 0%, {C['bg2']} 100%);
        border-right: 1px solid {C['card_border']};
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdown"] p {{
        font-size: 0.85rem;
    }}
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stDateInput label {{
        color: {C['text_dim']};
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    /* ── Sidebar nav section icons ── */
    .sidebar-nav {{
        padding: 8px 0;
    }}
    .sidebar-nav-item {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 14px;
        margin: 2px 0;
        border-radius: 8px;
        color: {C['text_dim']};
        font-size: 0.82rem;
        font-weight: 500;
        transition: all 0.2s ease;
        cursor: default;
    }}
    .sidebar-nav-item:hover {{
        background: rgba(0,212,255,0.06);
        color: {C['text']};
    }}
    .sidebar-nav-item.active {{
        background: rgba(0,212,255,0.1);
        color: {C['accent']};
        border-left: 3px solid {C['accent']};
    }}
    .sidebar-nav-item .nav-icon {{
        font-size: 1rem;
        width: 22px;
        text-align: center;
    }}

    /* ── Hero Section ── */
    .hero {{
        background: linear-gradient(135deg, rgba(0,212,255,0.04) 0%, rgba(99,102,241,0.04) 50%, rgba(0,227,150,0.02) 100%);
        border: 1px solid {C['card_border']};
        border-radius: 16px;
        padding: 40px 48px;
        margin-bottom: 24px;
        animation: fadeInUp 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }}
    .hero::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, {C['accent']}, {C['accent2']}, {C['green']});
    }}
    .hero-logo {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: -1px;
        background: linear-gradient(135deg, {C['accent']} 0%, {C['accent2']} 50%, {C['green']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.1;
    }}
    .hero-tagline {{
        color: {C['text']};
        font-size: 1rem;
        font-weight: 400;
        margin: 8px 0 16px 0;
        max-width: 600px;
    }}
    .hero-badges {{
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 12px;
    }}
    .hero-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        border: 1px solid;
    }}
    .badge-on {{
        color: {C['green']};
        border-color: rgba(0,227,150,0.3);
        background: rgba(0,227,150,0.06);
    }}
    .badge-off {{
        color: {C['text_dim']};
        border-color: {C['card_border']};
        background: rgba(100,116,139,0.06);
    }}
    .live-dot {{
        width: 6px; height: 6px;
        border-radius: 50%;
        background: {C['green']};
        animation: pulse 2s infinite;
    }}
    .offline-dot {{
        width: 6px; height: 6px;
        border-radius: 50%;
        background: {C['text_dim']};
    }}

    /* ── Section Headers ── */
    .section-hdr {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 14px 0 6px 0;
        margin: 32px 0 16px 0;
        border-bottom: 1px solid {C['card_border']};
        animation: fadeIn 0.5s ease-out;
    }}
    .section-hdr .icon {{
        font-size: 1.3rem;
    }}
    .section-hdr h2 {{
        margin: 0;
        font-size: 1.15rem;
        font-weight: 700;
        color: {C['text_bright']};
        letter-spacing: -0.3px;
    }}
    .section-hdr .label {{
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        padding: 3px 10px;
        border-radius: 4px;
        margin-left: auto;
    }}
    .label-quant {{
        color: {C['accent']};
        background: rgba(0,212,255,0.08);
        border: 1px solid rgba(0,212,255,0.2);
    }}
    .label-narrative {{
        color: {C['accent2']};
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.2);
    }}
    .label-ai {{
        color: {C['green']};
        background: rgba(0,227,150,0.08);
        border: 1px solid rgba(0,227,150,0.2);
    }}
    .label-live {{
        color: {C['yellow']};
        background: rgba(254,176,25,0.08);
        border: 1px solid rgba(254,176,25,0.2);
    }}

    /* ── Metric Cards ── */
    .metric-card {{
        background: {C['card']};
        border: 1px solid {C['card_border']};
        border-radius: 12px;
        padding: 20px 22px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        animation: fadeInUp 0.5s ease-out both;
    }}
    .metric-card:hover {{
        border-color: rgba(0,212,255,0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }}
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        border-radius: 12px 12px 0 0;
    }}
    .metric-card.accent::before  {{ background: {C['accent']}; }}
    .metric-card.green::before   {{ background: {C['green']}; }}
    .metric-card.red::before     {{ background: {C['red']}; }}
    .metric-card.yellow::before  {{ background: {C['yellow']}; }}
    .metric-card.purple::before  {{ background: {C['accent2']}; }}

    .metric-label {{
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: {C['text_dim']};
        margin-bottom: 8px;
    }}
    .metric-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.7rem;
        font-weight: 700;
        color: {C['text_bright']};
        line-height: 1.2;
    }}
    .metric-value.glow-accent {{ color: {C['accent']}; text-shadow: 0 0 20px {C['glow_accent']}; }}
    .metric-value.glow-green  {{ color: {C['green']};  text-shadow: 0 0 20px {C['glow_green']}; }}
    .metric-value.glow-red    {{ color: {C['red']};    text-shadow: 0 0 20px {C['glow_red']}; }}

    .metric-delta {{
        font-size: 0.75rem;
        font-weight: 500;
        margin-top: 4px;
        color: {C['text_dim']};
    }}
    .metric-delta.positive {{ color: {C['green']}; }}
    .metric-delta.negative {{ color: {C['red']}; }}

    /* ── Bloomberg Terminal Card ── */
    .bloomberg-card {{
        background: #0c0c0c;
        border: 1px solid #2a2a2a;
        border-radius: 4px;
        padding: 0;
        font-family: 'JetBrains Mono', monospace;
        overflow: hidden;
        animation: fadeInUp 0.5s ease-out both;
    }}
    .bloomberg-header {{
        background: linear-gradient(90deg, #1a1a2e, #16213e);
        padding: 10px 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #2a2a2a;
    }}
    .bloomberg-title {{
        color: {C['accent']};
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }}
    .bloomberg-timestamp {{
        color: {C['text_dim']};
        font-size: 0.65rem;
    }}
    .bloomberg-body {{
        padding: 16px;
    }}
    .bloomberg-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid rgba(255,255,255,0.03);
    }}
    .bloomberg-row:last-child {{ border-bottom: none; }}
    .bloomberg-label {{
        color: {C['text_dim']};
        font-size: 0.78rem;
        font-weight: 500;
    }}
    .bloomberg-val {{
        font-size: 0.85rem;
        font-weight: 700;
    }}
    .bloomberg-val.pos {{ color: {C['green']}; }}
    .bloomberg-val.neg {{ color: {C['red']}; }}
    .bloomberg-val.neutral {{ color: {C['text_bright']}; }}
    .bloomberg-val.highlight {{ color: {C['accent']}; text-shadow: 0 0 8px {C['glow_accent']}; }}

    /* ── Signal Strength Bar ── */
    .signal-meter {{
        background: {C['card']};
        border: 1px solid {C['card_border']};
        border-radius: 12px;
        padding: 20px 24px;
        animation: fadeInUp 0.5s ease-out both;
    }}
    .signal-meter-title {{
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: {C['text_dim']};
        margin-bottom: 14px;
    }}
    .signal-bar-track {{
        width: 100%;
        height: 8px;
        background: rgba(255,255,255,0.04);
        border-radius: 4px;
        overflow: hidden;
        margin-bottom: 10px;
    }}
    .signal-bar-fill {{
        height: 100%;
        border-radius: 4px;
        transition: width 1s ease-out;
    }}
    .signal-bar-fill.low  {{ background: linear-gradient(90deg, {C['red']}, {C['yellow']}); width: 25%; }}
    .signal-bar-fill.med  {{ background: linear-gradient(90deg, {C['yellow']}, {C['accent']}); width: 50%; }}
    .signal-bar-fill.high {{ background: linear-gradient(90deg, {C['accent']}, {C['green']}); width: 75%; }}
    .signal-bar-fill.max  {{ background: linear-gradient(90deg, {C['green']}, {C['accent']}); width: 100%; }}

    .signal-segments {{
        display: flex;
        gap: 4px;
        margin-bottom: 12px;
    }}
    .signal-seg {{
        flex: 1;
        height: 32px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    .signal-seg.on {{
        border: 1px solid rgba(0,227,150,0.4);
        background: rgba(0,227,150,0.1);
        color: {C['green']};
    }}
    .signal-seg.off {{
        border: 1px solid {C['card_border']};
        background: rgba(255,255,255,0.02);
        color: {C['text_dim']};
    }}
    .signal-score {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 800;
        text-align: center;
        margin-top: 8px;
    }}

    /* ── Classification Banner ── */
    .class-banner {{
        border-radius: 12px;
        padding: 24px 28px;
        margin: 16px 0;
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.6s ease-out;
    }}
    .class-banner::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
    }}
    .class-banner.expansion {{
        background: rgba(0,227,150,0.06);
        border: 1px solid rgba(0,227,150,0.2);
    }}
    .class-banner.expansion::before {{ background: {C['green']}; }}
    .class-banner.regime {{
        background: rgba(254,176,25,0.06);
        border: 1px solid rgba(254,176,25,0.2);
    }}
    .class-banner.regime::before {{ background: {C['yellow']}; }}
    .class-banner.other {{
        background: rgba(0,212,255,0.06);
        border: 1px solid rgba(0,212,255,0.2);
    }}
    .class-banner.other::before {{ background: {C['accent']}; }}
    .class-type {{
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 6px;
    }}
    .class-title {{
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 4px;
    }}
    .class-confidence {{
        font-size: 0.85rem;
        font-weight: 600;
    }}

    /* ── Layer cards (landing) ── */
    .layer-card {{
        background: {C['card']};
        border: 1px solid {C['card_border']};
        border-radius: 12px;
        padding: 28px 24px;
        height: 240px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out both;
    }}
    .layer-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
    }}
    .layer-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
    }}
    .layer-card.l1::before {{ background: {C['accent']}; }}
    .layer-card.l2::before {{ background: {C['accent2']}; }}
    .layer-card.l3::before {{ background: {C['green']}; }}
    .layer-card h4 {{
        font-size: 0.95rem;
        font-weight: 700;
        margin: 0 0 12px 0;
    }}
    .layer-card ul {{
        list-style: none;
        padding: 0;
        margin: 0;
    }}
    .layer-card li {{
        color: {C['text_dim']};
        font-size: 0.82rem;
        padding: 3px 0;
        padding-left: 16px;
        position: relative;
    }}
    .layer-card li::before {{
        content: '';
        position: absolute;
        left: 0;
        top: 50%;
        width: 4px;
        height: 4px;
        border-radius: 50%;
    }}
    .layer-card.l1 h4 {{ color: {C['accent']}; }}
    .layer-card.l1 li::before {{ background: {C['accent']}; }}
    .layer-card.l2 h4 {{ color: {C['accent2']}; }}
    .layer-card.l2 li::before {{ background: {C['accent2']}; }}
    .layer-card.l3 h4 {{ color: {C['green']}; }}
    .layer-card.l3 li::before {{ background: {C['green']}; }}

    /* ── Dividers ── */
    .divider {{
        height: 1px;
        background: linear-gradient(90deg, transparent, {C['card_border']}, transparent);
        margin: 32px 0;
    }}

    /* ── Buttons ── */
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {C['accent']} 0%, {C['accent2']} 100%);
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        color: {C['bg']};
    }}
    .stButton > button[kind="primary"]:hover {{
        box-shadow: 0 4px 20px {C['glow_accent']};
        transform: translateY(-1px);
    }}
    .stButton > button {{
        font-family: 'Inter', sans-serif;
        border-radius: 8px;
        transition: all 0.2s ease;
    }}

    /* ── Expanders ── */
    .streamlit-expanderHeader {{
        background: rgba(0,212,255,0.04);
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
    }}

    /* ── Dataframes ── */
    .stDataFrame {{
        border-radius: 8px;
        overflow: hidden;
    }}

    /* ── Chat ── */
    [data-testid="stChatMessage"] {{
        border-radius: 12px;
        border: 1px solid {C['card_border']};
        background: {C['card']};
    }}

    /* ── Default metric override ── */
    [data-testid="stMetric"] {{
        background: {C['card']};
        border: 1px solid {C['card_border']};
        border-radius: 12px;
        padding: 16px 20px;
    }}
    [data-testid="stMetricLabel"] {{
        color: {C['text_dim']} !important;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-family: 'Inter', sans-serif;
    }}
    [data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.5rem !important;
        font-weight: 700;
        color: {C['text_bright']};
    }}

    /* ── Footer ── */
    .app-footer {{
        text-align: center;
        padding: 24px 0;
        color: {C['text_dim']};
        font-size: 0.75rem;
        border-top: 1px solid {C['card_border']};
        margin-top: 40px;
    }}
    .app-footer a {{
        color: {C['accent']};
        text-decoration: none;
    }}

    /* ── Animation delays for staggered cards ── */
    .delay-1 {{ animation-delay: 0.1s; }}
    .delay-2 {{ animation-delay: 0.2s; }}
    .delay-3 {{ animation-delay: 0.3s; }}
    .delay-4 {{ animation-delay: 0.4s; }}
    .delay-5 {{ animation-delay: 0.5s; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PLOTLY THEME
# ============================================================

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=C["text"]),
    xaxis=dict(
        gridcolor="rgba(0,212,255,0.06)",
        zerolinecolor="rgba(0,212,255,0.1)",
        linecolor=C["card_border"],
    ),
    yaxis=dict(
        gridcolor="rgba(0,212,255,0.06)",
        zerolinecolor="rgba(0,212,255,0.1)",
        linecolor=C["card_border"],
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=C["card_border"],
        borderwidth=1,
        font=dict(size=11, color=C["text"]),
        yanchor="top", y=0.99, xanchor="left", x=0.01,
    ),
    margin=dict(l=0, r=0, t=30, b=0),
    hoverlabel=dict(
        bgcolor=C["card"],
        bordercolor=C["accent"],
        font=dict(color=C["text_bright"], family="Inter, sans-serif"),
    ),
)

# ============================================================
# HELPER: Custom Metric Card HTML
# ============================================================

def metric_card(label, value, delta=None, color="accent", glow=False, delay=0):
    """Render a custom metric card with gradient top border and optional glow."""
    glow_class = f"glow-{color}" if glow else ""
    delta_html = ""
    if delta:
        d_class = "positive" if "+" in str(delta) else "negative" if "-" in str(delta) else ""
        delta_html = f'<div class="metric-delta {d_class}">{delta}</div>'
    delay_class = f"delay-{delay}" if delay else ""
    return f"""
    <div class="metric-card {color} {delay_class}">
        <div class="metric-label">{label}</div>
        <div class="metric-value {glow_class}">{value}</div>
        {delta_html}
    </div>
    """

def section_header(icon, title, label_text="", label_class=""):
    """Render a styled section header."""
    label_html = f'<span class="label {label_class}">{label_text}</span>' if label_text else ""
    return f"""
    <div class="section-hdr">
        <span class="icon">{icon}</span>
        <h2>{title}</h2>
        {label_html}
    </div>
    """

def bloomberg_row(label, value, val_class="neutral"):
    """Single row in Bloomberg terminal card."""
    return f"""
    <div class="bloomberg-row">
        <span class="bloomberg-label">{label}</span>
        <span class="bloomberg-val {val_class}">{value}</span>
    </div>
    """

def signal_convergence_bar(triggered, total, signals_info):
    """Render a visual signal strength meter."""
    if triggered >= total:
        fill_class = "max"
    elif triggered >= total * 0.75:
        fill_class = "high"
    elif triggered >= total * 0.5:
        fill_class = "med"
    else:
        fill_class = "low"

    pct = (triggered / total) * 100

    segs_html = ""
    for sig_name, is_on in signals_info:
        cls = "on" if is_on else "off"
        check = "&#10003;" if is_on else "&#10007;"
        segs_html += f'<div class="signal-seg {cls}">{check}</div>'

    if triggered >= total * 0.75:
        score_color = C["green"]
    elif triggered >= total * 0.5:
        score_color = C["yellow"]
    else:
        score_color = C["red"]

    return f"""
    <div class="signal-meter">
        <div class="signal-meter-title">Signal Convergence</div>
        <div class="signal-bar-track">
            <div class="signal-bar-fill {fill_class}" style="width: {pct}%;"></div>
        </div>
        <div class="signal-segments">{segs_html}</div>
        <div class="signal-score" style="color: {score_color}; text-shadow: 0 0 20px {score_color}40;">
            {triggered}/{total}
        </div>
    </div>
    """


# ============================================================
# SIDEBAR — Navigation + Config
# ============================================================

st.sidebar.markdown(f"""
<div style="padding: 8px 0 16px 0;">
    <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 900;
        background: linear-gradient(135deg, {C['accent']}, {C['accent2']});
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        SSED
    </span>
    <div style="font-size: 0.7rem; color: {C['text_dim']}; margin-top: 2px;">
        Sample Space Expansion Detector
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown(f'<div style="font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; color:{C["text_dim"]}; margin-bottom:8px;">Event Parameters</div>', unsafe_allow_html=True)

event_name = st.sidebar.text_input("Event Name", value="ChatGPT Launch")
event_date = st.sidebar.date_input(
    "Event Date",
    value=datetime(2022, 11, 30),
)
analysis_end = st.sidebar.date_input(
    "Analysis End",
    value=datetime(2024, 12, 1),
)

st.sidebar.markdown("---")
st.sidebar.markdown(f'<div style="font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; color:{C["text_dim"]}; margin-bottom:8px;">Tickers</div>', unsafe_allow_html=True)

# Preset scenarios
preset = st.sidebar.selectbox(
    "Preset Scenario",
    [
        "Custom",
        "ChatGPT Launch (NVDA vs CHGG)",
        "iPhone Revolution (AAPL vs NOK)",
        "Streaming Wars (NFLX vs DIS)",
        "EV Disruption (TSLA vs F)",
        "Cloud Computing (AMZN vs IBM)",
        "Social Media Shift (META vs SNAP)",
    ],
)

preset_configs = {
    "ChatGPT Launch (NVDA vs CHGG)": ("NVDA", "CHGG", "ChatGPT Launch", "2022-11-30"),
    "iPhone Revolution (AAPL vs NOK)": ("AAPL", "NOK", "iPhone Launch", "2007-06-29"),
    "Streaming Wars (NFLX vs DIS)": ("NFLX", "DIS", "Netflix Streaming Pivot", "2013-02-01"),
    "EV Disruption (TSLA vs F)": ("TSLA", "F", "Tesla Mass Market Push", "2017-07-01"),
    "Cloud Computing (AMZN vs IBM)": ("AMZN", "IBM", "AWS Dominance", "2015-01-01"),
    "Social Media Shift (META vs SNAP)": ("META", "SNAP", "Meta AI Pivot", "2023-02-01"),
}

if preset != "Custom" and preset in preset_configs:
    p_winner, p_loser, p_event, p_date = preset_configs[preset]
    winner_ticker = st.sidebar.text_input("Suspected Winner", value=p_winner)
    loser_ticker = st.sidebar.text_input("Suspected Loser", value=p_loser)
    if preset != "Custom":
        event_name = st.sidebar.text_input("Event Name (override)", value=p_event, key="event_override")
else:
    winner_ticker = st.sidebar.text_input("Suspected Winner", value="NVDA")
    loser_ticker = st.sidebar.text_input("Suspected Loser", value="CHGG")

# Additional long tickers for backtest
additional_long = st.sidebar.text_input(
    "Additional Long Tickers (comma-separated)",
    value="MSFT",
    help="Extra tickers to include in the long leg of the backtest",
)
additional_long_list = [t.strip().upper() for t in additional_long.split(",") if t.strip()]

benchmark_ticker = st.sidebar.text_input("Benchmark", value="SPY")

st.sidebar.markdown("---")

has_openai = bool(os.environ.get("OPENAI_API_KEY"))
has_newsapi = bool(os.environ.get("NEWSAPI_KEY"))

st.sidebar.markdown(f"""
<div style="font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; color:{C['text_dim']}; margin-bottom:10px;">System Status</div>
<div style="display:flex; flex-direction:column; gap:6px;">
    <div class="hero-badge {'badge-on' if has_openai else 'badge-off'}">
        <div class="{'live-dot' if has_openai else 'offline-dot'}"></div>
        OpenAI API
    </div>
    <div class="hero-badge {'badge-on' if has_newsapi else 'badge-off'}">
        <div class="{'live-dot' if has_newsapi else 'offline-dot'}"></div>
        NewsAPI
    </div>
    <div class="hero-badge badge-on">
        <div class="live-dot"></div>
        yfinance
    </div>
</div>
""", unsafe_allow_html=True)

if not has_openai:
    st.sidebar.caption(
        "Set OPENAI_API_KEY in .env to enable AI classification (Layer 3)"
    )

# ============================================================
# HERO SECTION
# ============================================================

st.markdown(f"""
<div class="hero">
    <p class="hero-logo">SSED</p>
    <p class="hero-tagline">
        Fusing quantitative regime detection with LLM-powered narrative analysis
        to distinguish regime shifts from sample space expansion.
    </p>
    <div class="hero-badges">
        <div class="hero-badge {'badge-on' if has_openai else 'badge-off'}">
            <div class="{'live-dot' if has_openai else 'offline-dot'}"></div>
            OpenAI
        </div>
        <div class="hero-badge {'badge-on' if has_newsapi else 'badge-off'}">
            <div class="{'live-dot' if has_newsapi else 'offline-dot'}"></div>
            NewsAPI
        </div>
        <div class="hero-badge badge-on">
            <div class="live-dot"></div>
            yfinance
        </div>
        <div class="hero-badge badge-on" style="color:{C['text_dim']}; border-color:{C['card_border']};">
            MGMT 69000 &middot; Purdue University
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Conceptual framework
with st.expander("What is Sample Space Expansion?", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **Regime Shift (P changed)**
        - Probabilities change within a known universe
        - Same assets, same categories, different dynamics
        - Example: Tariff shock changes sector rotation
        - HMM detects this well
        """)
    with col2:
        st.markdown(f"""
        **Sample Space Expansion (X changed)**
        - The investment universe ITSELF changes
        - New asset class emerges, old ones may die
        - Example: ChatGPT creates "AI infrastructure"
        - Requires quant + narrative fusion to detect
        """)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================
# RUN ANALYSIS
# ============================================================

event_date_str = event_date.strftime("%Y-%m-%d")
analysis_end_str = analysis_end.strftime("%Y-%m-%d")

if st.sidebar.button("Run Analysis", type="primary", use_container_width=True):
    st.session_state["running"] = True
    # Force recompute on new analysis
    st.session_state.pop("quant", None)
    st.session_state.pop("news", None)
    st.session_state.pop("bt", None)
    st.session_state.pop("prices_cache", None)
    st.session_state.pop("classification", None)
    st.session_state.pop("ai_narrative", None)
    st.session_state.pop("qa_messages", None)

if st.session_state.get("running"):

    # --------------------------------------------------------
    # AI MARKET NARRATIVE placeholder (rendered after data loads)
    # --------------------------------------------------------
    narrative_placeholder = st.empty()

    # --------------------------------------------------------
    # LAYER 1: Quantitative Signals
    # --------------------------------------------------------
    st.markdown(section_header("📈", "Layer 1 — Quantitative Signals", "DETERMINISTIC", "label-quant"), unsafe_allow_html=True)
    st.caption("HMM, entropy, divergence & concentration — no LLM involved")

    if "quant" not in st.session_state:
        with st.spinner("Fetching market data and computing signals..."):
            quant = run_quant_signals(
                event_date=event_date_str,
                analysis_end=analysis_end_str,
                winner=winner_ticker.upper(),
                loser=loser_ticker.upper(),
                benchmark=benchmark_ticker.upper(),
            )
            st.session_state["quant"] = quant

    quant = st.session_state["quant"]

    # Custom metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card(
            "HMM Regime",
            quant.hmm.regime_label.replace("_", " ").title(),
            delta=f"p = {quant.hmm.regime_probability:.2f}",
            color="accent",
            glow=True,
            delay=1,
        ), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card(
            "Entropy Z-Score",
            f"{quant.entropy.entropy_zscore:.2f}\u03c3",
            delta=f"{quant.entropy.entropy_change:+.3f} bits",
            color="purple" if quant.entropy.entropy_zscore < -2 else "accent",
            glow=abs(quant.entropy.entropy_zscore) > 2,
            delay=2,
        ), unsafe_allow_html=True)
    with col3:
        div_color = "green" if quant.divergence.total_divergence_pct > 0 else "red"
        st.markdown(metric_card(
            "Divergence",
            f"{quant.divergence.total_divergence_pct:+.0f}%",
            delta=f"{winner_ticker} vs {loser_ticker}",
            color=div_color,
            glow=abs(quant.divergence.total_divergence_pct) > 500,
            delay=3,
        ), unsafe_allow_html=True)
    with col4:
        hhi_color = "red" if quant.concentration.hhi_change > 0.02 else "green"
        st.markdown(metric_card(
            "HHI Change",
            f"{quant.concentration.hhi_change:+.4f}",
            delta="More concentrated" if quant.concentration.hhi_change > 0 else "More diversified",
            color=hhi_color,
            delay=4,
        ), unsafe_allow_html=True)

    st.markdown("")

    # Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown(f'<p style="font-size:0.85rem; font-weight:600; color:{C["text_bright"]}; margin-bottom:4px;">Winner vs Loser Divergence</p>', unsafe_allow_html=True)

        tickers = [winner_ticker.upper(), loser_ticker.upper(), benchmark_ticker.upper()]
        if "prices_cache" not in st.session_state:
            st.session_state["prices_cache"] = fetch_prices(tickers, event_date_str, analysis_end_str)
        prices = st.session_state["prices_cache"]

        fig = go.Figure()
        chart_colors = {
            winner_ticker.upper(): C["green"],
            loser_ticker.upper(): C["red"],
            benchmark_ticker.upper(): C["text_dim"],
        }
        for ticker in tickers:
            if ticker in prices.columns:
                normalized = (prices[ticker] / prices[ticker].iloc[0]) * 100
                fig.add_trace(go.Scatter(
                    x=normalized.index,
                    y=normalized.values,
                    name=ticker,
                    line=dict(
                        color=chart_colors.get(ticker, C["accent"]),
                        width=2.5 if ticker != benchmark_ticker.upper() else 1.5,
                        dash="dash" if ticker == benchmark_ticker.upper() else "solid",
                    ),
                ))

        fig.add_hline(y=100, line_dash="dot", line_color="rgba(0,212,255,0.15)", opacity=0.5)
        fig.update_layout(
            **CHART_LAYOUT,
            yaxis_title="Normalized Price (100 = Event Date)",
            yaxis_type="log",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        st.markdown(f'<p style="font-size:0.85rem; font-weight:600; color:{C["text_bright"]}; margin-bottom:4px;">Rolling Entropy (Concentration)</p>', unsafe_allow_html=True)

        ent = quant.entropy
        if ent.rolling_dates and ent.rolling_entropy:
            dates = pd.to_datetime(ent.rolling_dates)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=dates,
                y=ent.rolling_entropy,
                name="Rolling Entropy",
                line=dict(color=C["accent"], width=2),
                fill="tozeroy",
                fillcolor="rgba(0,212,255,0.06)",
            ))
            fig2.add_hline(
                y=ent.baseline_entropy,
                line_dash="dash",
                line_color=C["yellow"],
                annotation_text="Pre-event baseline",
                annotation_font=dict(color=C["yellow"], size=10),
            )

            event_dt = pd.to_datetime(event_date_str)
            if dates.min() <= event_dt <= dates.max():
                fig2.add_shape(
                    type="line",
                    x0=event_dt.isoformat(), x1=event_dt.isoformat(),
                    y0=0, y1=1, yref="paper",
                    line=dict(dash="dot", color=C["red"]),
                )
                fig2.add_annotation(
                    x=event_dt.isoformat(), y=1, yref="paper",
                    text=event_name, showarrow=False,
                    font=dict(color=C["red"], size=10),
                )

            fig2.update_layout(
                **CHART_LAYOUT,
                yaxis_title="Shannon Entropy (bits)",
                height=400,
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Insufficient data for rolling entropy chart")

    # HMM Regime Timeline
    st.markdown(f'<p style="font-size:0.85rem; font-weight:600; color:{C["text_bright"]}; margin-bottom:4px;">HMM Regime States Over Time</p>', unsafe_allow_html=True)

    if quant.hmm.regime_history:
        regime_map = {"low_volatility": 0, "medium_volatility": 1, "high_volatility": 2}
        regime_colors_map = {0: C["green"], 1: C["yellow"], 2: C["red"]}

        regime_values = [regime_map.get(r, 1) for r in quant.hmm.regime_history]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            y=regime_values,
            mode="lines",
            line=dict(color=C["accent"], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.06)",
            name="Regime State",
        ))
        fig3.update_layout(**CHART_LAYOUT, height=200)
        fig3.update_yaxes(
            tickvals=[0, 1, 2],
            ticktext=["Low Vol", "Medium Vol", "High Vol"],
            gridcolor="rgba(0,212,255,0.06)",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with st.expander("HMM Transition Matrix"):
        transmat = quant.hmm.transition_matrix
        labels = ["Low Vol", "Med Vol", "High Vol"]
        df_trans = pd.DataFrame(transmat, index=labels, columns=labels)
        st.dataframe(df_trans.style.format("{:.3f}").background_gradient(cmap="YlOrRd"), use_container_width=True)

    # ── Expansion Signal Evolution (time-series) ──────────────
    st.markdown(
        f'<p style="font-size:0.85rem; font-weight:600; color:{C["text_bright"]}; margin:16px 0 4px;">Expansion Signal Evolution</p>',
        unsafe_allow_html=True,
    )
    if ent.rolling_dates and ent.rolling_entropy and quant.hmm.regime_history:
        import numpy as _np
        exp_dates = pd.to_datetime(ent.rolling_dates)
        ent_arr = _np.array(ent.rolling_entropy)

        # Normalise entropy to 0-1 (inverted: lower entropy = higher concentration signal)
        ent_min, ent_max = ent_arr.min(), ent_arr.max()
        inv_ent = 1.0 - (ent_arr - ent_min) / (ent_max - ent_min + 1e-9)

        # Align HMM regime history to the entropy rolling window
        regime_numeric = _np.array([
            {"low_volatility": 0.0, "medium_volatility": 0.5, "high_volatility": 1.0}.get(r, 0.5)
            for r in quant.hmm.regime_history
        ])
        n = len(inv_ent)
        regime_aligned = regime_numeric[-n:] if len(regime_numeric) >= n else _np.pad(
            regime_numeric, (n - len(regime_numeric), 0), constant_values=0.5
        )

        # Composite: 60% entropy concentration + 40% HMM volatility regime
        expansion_ts = 0.6 * inv_ent + 0.4 * regime_aligned

        fig_exp = go.Figure()
        fig_exp.add_trace(go.Scatter(
            x=exp_dates,
            y=expansion_ts,
            name="Expansion Signal",
            line=dict(color=C["red"], width=2),
            fill="tozeroy",
            fillcolor="rgba(255,82,82,0.07)",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Score: %{y:.3f}<extra></extra>",
        ))
        # Mark the event date
        event_dt = pd.to_datetime(event_date_str)
        if exp_dates.min() <= event_dt <= exp_dates.max():
            fig_exp.add_shape(
                type="line",
                x0=event_dt.isoformat(), x1=event_dt.isoformat(),
                y0=0, y1=1, yref="paper",
                line=dict(dash="dot", color=C["yellow"], width=1.5),
            )
            fig_exp.add_annotation(
                x=event_dt.isoformat(), y=0.97, yref="paper",
                text=event_name, showarrow=False,
                font=dict(color=C["yellow"], size=10),
                xanchor="left", xshift=6,
            )
        fig_exp.update_layout(
            **CHART_LAYOUT,
            yaxis_title="Composite Expansion Score (0–1)",
            height=280,
        )
        fig_exp.update_yaxes(range=[0, 1.1])
        st.plotly_chart(fig_exp, use_container_width=True)
        st.caption(
            "Composite of entropy concentration (60 %) + HMM volatility regime (40 %). "
            "Rising score = signals converging toward sample space expansion."
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # --------------------------------------------------------
    # LAYER 2: Narrative Signals
    # --------------------------------------------------------
    st.markdown(section_header("📰", "Layer 2 — Narrative Signals", "NLP", "label-narrative"), unsafe_allow_html=True)
    st.caption("News sentiment analysis powered by NewsAPI + OpenAI")

    if "news" not in st.session_state:
        with st.spinner("Analyzing narrative signals..."):
            news_signals = compute_news_signals(
                query=f"{event_name} AI market impact",
                from_date=event_date_str,
                to_date=analysis_end_str,
                event_context=f"{event_name} on {event_date_str}",
            )
            st.session_state["news"] = news_signals

    news_signals = st.session_state["news"]

    # Metrics row
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(metric_card(
            "Articles Analyzed",
            str(news_signals.article_count),
            color="purple",
            delay=1,
        ), unsafe_allow_html=True)
    with m2:
        sent_color = "green" if news_signals.avg_sentiment > 0 else "red"
        st.markdown(metric_card(
            "Avg Sentiment",
            f"{news_signals.avg_sentiment:+.3f}",
            color=sent_color,
            glow=True,
            delay=2,
        ), unsafe_allow_html=True)
    with m3:
        st.markdown(metric_card(
            "Trend",
            news_signals.sentiment_trend.title(),
            color="accent",
            delay=3,
        ), unsafe_allow_html=True)

    news_detail_col1, news_detail_col2 = st.columns(2)

    with news_detail_col1:
        if news_signals.novel_theme_counts:
            st.markdown(f'<p style="font-size:0.85rem; font-weight:600; color:{C["text_bright"]};">Novel Themes Detected</p>', unsafe_allow_html=True)
            for theme, count in sorted(
                news_signals.novel_theme_counts.items(), key=lambda x: -x[1]
            ):
                st.markdown(f"- **{theme}**: {count} mention{'s' if count > 1 else ''}")
        else:
            st.info("No novel themes detected")

    with news_detail_col2:
        st.markdown(f'<p style="font-size:0.85rem; font-weight:600; color:{C["text_bright"]};">Top Articles</p>', unsafe_allow_html=True)
        for a in news_signals.top_articles[:4]:
            color = C["green"] if a.sentiment_score > 0.3 else C["red"] if a.sentiment_score < -0.3 else C["yellow"]
            st.markdown(f'<span style="color:{color}; font-family: JetBrains Mono, monospace; font-size:0.8rem;">[{a.sentiment_score:+.2f}]</span> {a.title}', unsafe_allow_html=True)
            if a.novel_themes:
                st.caption(f"Themes: {', '.join(a.novel_themes)}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # --------------------------------------------------------
    # LAYER 3: Fusion & Classification (runs automatically)
    # --------------------------------------------------------
    st.markdown(section_header("🧠", "Layer 3 — AI Fusion & Classification", "NON-DETERMINISTIC · o4-MINI", "label-ai"), unsafe_allow_html=True)

    if has_openai:
        st.caption("OpenAI o4-mini reasoning over all signals — non-deterministic (different each run)")

        if "classification" not in st.session_state:
            with st.spinner("o4-mini is analyzing all signals (non-deterministic LLM reasoning)..."):
                try:
                    from ssed.openai_core import classify_event

                    result = classify_event(
                        event_description=(
                            f"{event_name} — analyzing market impact on "
                            f"{winner_ticker} (winner) vs {loser_ticker} (loser)"
                        ),
                        event_date=event_date_str,
                        analysis_end=analysis_end_str,
                        winner=winner_ticker.upper(),
                        loser=loser_ticker.upper(),
                    )
                    st.session_state["classification"] = result
                except Exception as e:
                    st.error(f"OpenAI API error: {e}\n\nAdd credits at https://platform.openai.com/settings/organization/billing/overview")

        if "classification" in st.session_state:
            result = st.session_state["classification"]

            if result.classification.value == "sample_space_expansion":
                st.markdown(f"""
                <div class="class-banner expansion">
                    <div class="class-type" style="color:{C['green']};">Classification Result</div>
                    <div class="class-title" style="color:{C['green']};">SAMPLE SPACE EXPANSION</div>
                    <div class="class-confidence" style="color:{C['text']};">
                        The universe changed (X expanded) &middot; Confidence: {result.confidence.value.upper()}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif result.classification.value == "regime_shift":
                st.markdown(f"""
                <div class="class-banner regime">
                    <div class="class-type" style="color:{C['yellow']};">Classification Result</div>
                    <div class="class-title" style="color:{C['yellow']};">REGIME SHIFT</div>
                    <div class="class-confidence" style="color:{C['text']};">
                        Probabilities changed (P shifted) &middot; Confidence: {result.confidence.value.upper()}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="class-banner other">
                    <div class="class-type" style="color:{C['accent']};">Classification Result</div>
                    <div class="class-title" style="color:{C['accent']};">{result.classification.value.replace('_', ' ').upper()}</div>
                    <div class="class-confidence" style="color:{C['text']};">
                        Confidence: {result.confidence.value.upper()}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"**What Changed:** {result.what_changed}")
            st.markdown(f"**Reasoning:** {result.reasoning}")

            ev_col1, ev_col2, ev_col3 = st.columns(3)
            with ev_col1:
                st.markdown("**Entropy Signal**")
                st.markdown(result.entropy_interpretation)
            with ev_col2:
                st.markdown("**Divergence Signal**")
                st.markdown(result.divergence_interpretation)
            with ev_col3:
                st.markdown("**HMM Signal**")
                st.markdown(result.hmm_interpretation)

            st.markdown("**Key Evidence:**")
            for e in result.key_evidence:
                st.markdown(f"- {e}")

            with st.expander("Raw Classification JSON"):
                st.json(result.model_dump())

    else:
        st.caption("Heuristic classification (set OPENAI_API_KEY for AI-powered analysis)")

        signals_detected = 0
        evidence = []

        if quant.divergence.total_divergence_pct > 500:
            signals_detected += 1
            evidence.append(
                f"Divergence of {quant.divergence.total_divergence_pct:.0f}% "
                f"({winner_ticker} vs {loser_ticker}) — exceeds 500% threshold"
            )

        if quant.entropy.entropy_zscore < -2:
            signals_detected += 1
            evidence.append(
                f"Entropy z-score of {quant.entropy.entropy_zscore:.2f} — "
                f"unusual concentration (>2\u03c3 below mean)"
            )

        if quant.concentration.hhi_change > 0.02:
            signals_detected += 1
            evidence.append(
                f"HHI increased by {quant.concentration.hhi_change:.4f} — "
                f"market more concentrated"
            )

        if news_signals.novel_theme_counts:
            signals_detected += 1
            themes = ", ".join(news_signals.novel_theme_counts.keys())
            evidence.append(f"Novel narrative themes detected: {themes}")

        if signals_detected >= 3:
            classification = "SAMPLE SPACE EXPANSION"
            what_changed = (
                "The investment universe itself expanded. A new asset class "
                "(AI infrastructure) emerged, while existing categories "
                "(education, knowledge work) faced existential disruption."
            )
            st.markdown(f"""
            <div class="class-banner expansion">
                <div class="class-type" style="color:{C['green']};">Heuristic Classification</div>
                <div class="class-title" style="color:{C['green']};">{classification}</div>
                <div class="class-confidence" style="color:{C['text']};">
                    The universe changed (X expanded) &middot; {signals_detected}/4 signals converge
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif signals_detected >= 2:
            classification = "LIKELY SAMPLE SPACE EXPANSION"
            what_changed = (
                "Multiple signals suggest the investment universe changed, "
                "but not all indicators converge."
            )
            st.markdown(f"""
            <div class="class-banner regime">
                <div class="class-type" style="color:{C['yellow']};">Heuristic Classification</div>
                <div class="class-title" style="color:{C['yellow']};">{classification}</div>
                <div class="class-confidence" style="color:{C['text']};">
                    Likely expansion &middot; {signals_detected}/4 signals converge
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            classification = "REGIME SHIFT"
            what_changed = "Probabilities shifted within the existing universe."
            st.markdown(f"""
            <div class="class-banner other">
                <div class="class-type" style="color:{C['accent']};">Heuristic Classification</div>
                <div class="class-title" style="color:{C['accent']};">{classification}</div>
                <div class="class-confidence" style="color:{C['text']};">
                    P changed within known universe &middot; {signals_detected}/4 signals converge
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"**What Changed:** {what_changed}")

        st.markdown("**Evidence:**")
        for e in evidence:
            st.markdown(f"- {e}")

        missing = 4 - signals_detected
        if missing > 0:
            st.caption(f"{missing} signal(s) did not reach threshold")

    # --------------------------------------------------------
    # AI MARKET NARRATIVE — Non-deterministic LLM summary
    # --------------------------------------------------------
    if has_openai and "ai_narrative" not in st.session_state:
        with st.spinner("Generating AI market narrative (non-deterministic)..."):
            try:
                from openai import OpenAI
                client = OpenAI()

                narrative_context = (
                    f"Event: {event_name} on {event_date_str}\n"
                    f"Winner: {winner_ticker} ({quant.divergence.winner_return_pct:+.0f}%)\n"
                    f"Loser: {loser_ticker} ({quant.divergence.loser_return_pct:.0f}%)\n"
                    f"HMM Regime: {quant.hmm.regime_label} (p={quant.hmm.regime_probability:.2f})\n"
                    f"Entropy Z-Score: {quant.entropy.entropy_zscore:.2f}\n"
                    f"Divergence: {quant.divergence.total_divergence_pct:+.0f}%\n"
                    f"HHI Change: {quant.concentration.hhi_change:+.4f}\n"
                    f"News Sentiment: {news_signals.avg_sentiment:+.3f} ({news_signals.sentiment_trend})\n"
                    f"Novel Themes: {', '.join(news_signals.novel_theme_counts.keys()) if news_signals.novel_theme_counts else 'None'}\n"
                )
                if "classification" in st.session_state:
                    r = st.session_state["classification"]
                    narrative_context += (
                        f"\nAI Classification: {r.classification.value}\n"
                        f"Confidence: {r.confidence.value}\n"
                        f"What Changed: {r.what_changed}\n"
                    )

                narrative_response = client.chat.completions.create(
                    model="gpt-4.1-nano",
                    messages=[
                        {"role": "system", "content": (
                            "You are a senior financial analyst writing a brief market intelligence summary. "
                            "Write a concise 3-4 sentence narrative paragraph interpreting ALL the signals together. "
                            "Focus on what the data means for investors. Be specific with numbers. "
                            "Mention whether this appears to be a regime shift or sample space expansion."
                        )},
                        {"role": "user", "content": f"Write a market narrative summary for this analysis:\n\n{narrative_context}"},
                    ],
                    max_tokens=300,
                )
                st.session_state["ai_narrative"] = narrative_response.choices[0].message.content
            except Exception as e:
                st.session_state["ai_narrative"] = None

    # Render the narrative at the top via placeholder
    with narrative_placeholder.container():
        if has_openai and st.session_state.get("ai_narrative"):
            st.markdown(section_header("🤖", "AI Market Narrative", "NON-DETERMINISTIC · GPT-4.1-NANO", "label-ai"), unsafe_allow_html=True)
            st.markdown(f"""
            <div class="class-banner other" style="animation: fadeInUp 0.6s ease-out;">
                <div class="class-type" style="color:{C['accent']};">AI-Generated Summary (different each run)</div>
                <div style="color:{C['text']}; font-size:0.95rem; line-height:1.7; margin-top:8px;">
                    {st.session_state['ai_narrative']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # --------------------------------------------------------
    # SIGNAL CONVERGENCE — Visual Meter
    # --------------------------------------------------------
    st.markdown(section_header("🎯", "Signal Convergence", "", ""), unsafe_allow_html=True)

    sig_info = [
        ("DIV >500%", quant.divergence.total_divergence_pct > 500),
        ("ENT z<-2", quant.entropy.entropy_zscore < -2),
        ("HHI +", quant.concentration.hhi_change > 0.02),
        ("THEMES", bool(news_signals.novel_theme_counts)),
    ]

    triggered_count = sum(1 for _, on in sig_info if on)

    conv_col1, conv_col2 = st.columns([1, 2])
    with conv_col1:
        st.markdown(signal_convergence_bar(triggered_count, 4, sig_info), unsafe_allow_html=True)

    with conv_col2:
        summary_data = {
            "Signal": [
                "Divergence (>500%)",
                "Entropy (z < -2)",
                "HHI (increasing)",
                "Novel Narratives",
            ],
            "Layer": ["Quant", "Quant", "Quant", "Narrative"],
            "Value": [
                f"{quant.divergence.total_divergence_pct:+.0f}%",
                f"{quant.entropy.entropy_zscore:.2f}\u03c3",
                f"{quant.concentration.hhi_change:+.4f}",
                str(len(news_signals.novel_theme_counts)) + " themes",
            ],
            "Triggered": [
                "\u2705" if quant.divergence.total_divergence_pct > 500 else "\u274c",
                "\u2705" if quant.entropy.entropy_zscore < -2 else "\u274c",
                "\u2705" if quant.concentration.hhi_change > 0.02 else "\u274c",
                "\u2705" if news_signals.novel_theme_counts else "\u274c",
            ],
        }
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(
            df_summary.style.apply(
                lambda row: [
                    "background-color: rgba(0,227,150,0.08)" if row["Triggered"] == "\u2705" else ""
                    for _ in row
                ],
                axis=1,
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # --------------------------------------------------------
    # LONG-SHORT PORTFOLIO BACKTEST — Bloomberg Style
    # --------------------------------------------------------
    st.markdown(section_header("💰", "Long-Short Portfolio Backtest", "STRATEGY", "label-quant"), unsafe_allow_html=True)
    st.caption(
        f"Long AI winners / Short disrupted — equal weight, dollar-neutral, buy-and-hold from event date"
    )

    if "bt" not in st.session_state:
        with st.spinner("Running backtest..."):
            long_list = [winner_ticker.upper()] + [t for t in additional_long_list if t != winner_ticker.upper()]
            bt = run_backtest(
                long_tickers=long_list,
                short_tickers=[loser_ticker.upper()],
                start_date=event_date_str,
                end_date=analysis_end_str,
            )
            st.session_state["bt"] = bt

    bt = st.session_state["bt"]

    # Bloomberg terminal card for backtest metrics
    bt_col1, bt_col2 = st.columns([2, 1])

    with bt_col1:
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=bt.portfolio_values.index, y=bt.portfolio_values.values,
            name="Long-Short Portfolio",
            line=dict(color=C["green"], width=3),
            fill="tozeroy",
            fillcolor="rgba(0,227,150,0.04)",
        ))
        fig_bt.add_trace(go.Scatter(
            x=bt.long_values.index, y=bt.long_values.values,
            name=f"Long ({', '.join(bt.long_tickers)})",
            line=dict(color=C["accent"], width=1.5, dash="dot"),
        ))
        fig_bt.add_trace(go.Scatter(
            x=bt.short_values.index, y=bt.short_values.values,
            name=f"Short ({', '.join(bt.short_tickers)})",
            line=dict(color=C["red"], width=1.5, dash="dot"),
        ))
        if bt.benchmark_values is not None:
            fig_bt.add_trace(go.Scatter(
                x=bt.benchmark_values.index, y=bt.benchmark_values.values,
                name="SPY (Benchmark)",
                line=dict(color=C["text_dim"], width=1.5, dash="dash"),
            ))

        fig_bt.add_hline(y=100, line_dash="dot", line_color="rgba(0,212,255,0.1)", opacity=0.3)
        fig_bt.update_layout(
            **CHART_LAYOUT,
            yaxis_title="Portfolio Value ($100 initial)",
            height=420,
        )
        st.plotly_chart(fig_bt, use_container_width=True)

    with bt_col2:
        ret_class = "pos" if bt.total_return_pct > 0 else "neg"
        alpha_class = "pos" if bt.alpha_pct > 0 else "neg"
        long_class = "pos" if bt.long_return_pct > 0 else "neg"
        short_class = "pos" if bt.short_return_pct > 0 else "neg"
        sharpe_class = "highlight" if bt.sharpe_ratio > 1.5 else "neutral"

        st.markdown(f"""
        <div class="bloomberg-card">
            <div class="bloomberg-header">
                <span class="bloomberg-title">Backtest Results</span>
                <span class="bloomberg-timestamp">{event_date_str} \u2192 {analysis_end_str}</span>
            </div>
            <div class="bloomberg-body">
                {bloomberg_row("Total Return", f"{bt.total_return_pct:+.1f}%", ret_class)}
                {bloomberg_row("Alpha vs SPY", f"{bt.alpha_pct:+.1f}%", alpha_class)}
                {bloomberg_row("Sharpe Ratio", f"{bt.sharpe_ratio:.2f}", sharpe_class)}
                {bloomberg_row("Max Drawdown", f"{bt.max_drawdown_pct:.1f}%", "neg")}
                {bloomberg_row("Volatility", f"{bt.volatility_pct:.1f}%", "neutral")}
                {bloomberg_row("Long Leg", f"{bt.long_return_pct:+.1f}%", long_class)}
                {bloomberg_row("Short Leg", f"{bt.short_return_pct:+.1f}%", short_class)}
                {bloomberg_row("Benchmark", f"{bt.benchmark_return_pct:+.1f}%", "neutral")}
                {bloomberg_row("Annualized", f"{bt.annualized_return_pct:+.1f}%", ret_class)}
                {bloomberg_row("Trading Days", str(bt.trading_days), "neutral")}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # WALK-FORWARD BACKTEST (Bias-Free)
    # --------------------------------------------------------
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        section_header("🔬", "Walk-Forward Backtest (Bias-Free)", "STRATEGY", "label-quant"),
        unsafe_allow_html=True,
    )
    st.caption(
        "Ticker selection made by the system at each rebalance using only data available at T — no hindsight."
    )
    st.info(
        "**Methodology:** At each ~monthly rebalance, stocks in the universe are ranked by "
        "trailing 60-day return using only data up to that date. The system goes long the "
        "top 2 and short the bottom 1 — selected algorithmically, not by the analyst knowing "
        "the outcome. This tests whether trailing divergence signals predict forward returns.",
        icon="ℹ️",
    )

    if "bt_fwd" not in st.session_state:
        with st.spinner("Running walk-forward backtest..."):
            try:
                bt_fwd = run_forward_looking_backtest(
                    start_date=event_date_str,
                    end_date=analysis_end_str,
                )
                st.session_state["bt_fwd"] = bt_fwd
            except Exception as e:
                st.session_state["bt_fwd"] = None
                st.error(f"Walk-forward backtest error: {e}")

    bt_fwd = st.session_state.get("bt_fwd")
    if bt_fwd is not None:
        fwd_col1, fwd_col2 = st.columns([2, 1])
        with fwd_col1:
            fig_fwd = go.Figure()
            fig_fwd.add_trace(go.Scatter(
                x=bt_fwd.portfolio_values.index, y=bt_fwd.portfolio_values.values,
                name="Walk-Forward Portfolio",
                line=dict(color=C["accent2"], width=3),
                fill="tozeroy",
                fillcolor="rgba(99,102,241,0.06)",
            ))
            fig_fwd.add_trace(go.Scatter(
                x=bt_fwd.long_values.index, y=bt_fwd.long_values.values,
                name="Long Leg",
                line=dict(color=C["green"], width=1.5, dash="dot"),
            ))
            fig_fwd.add_trace(go.Scatter(
                x=bt_fwd.short_values.index, y=bt_fwd.short_values.values,
                name="Short Leg",
                line=dict(color=C["red"], width=1.5, dash="dot"),
            ))
            if bt_fwd.benchmark_values is not None:
                fig_fwd.add_trace(go.Scatter(
                    x=bt_fwd.benchmark_values.index, y=bt_fwd.benchmark_values.values,
                    name="SPY Benchmark",
                    line=dict(color=C["text_dim"], width=1, dash="dash"),
                ))
            fig_fwd.update_layout(
                paper_bgcolor=C["card"], plot_bgcolor=C["card"],
                font=dict(color=C["text"], size=11),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(gridcolor=C["card_border"]),
                yaxis=dict(gridcolor=C["card_border"], title="Portfolio Value ($)"),
                height=300,
            )
            st.plotly_chart(fig_fwd, use_container_width=True)

        with fwd_col2:
            fwd_ret_class = "pos" if bt_fwd.total_return_pct > 0 else "neg"
            fwd_alpha_class = "pos" if bt_fwd.alpha_pct > 0 else "neg"
            fwd_sharpe_class = "pos" if bt_fwd.sharpe_ratio > 1 else "neutral"
            st.markdown(f"""
            <div class="bloomberg-card">
                <div class="bloomberg-header">WALK-FORWARD PERFORMANCE</div>
                <div class="bloomberg-body">
                    {bloomberg_row("Total Return", f"{bt_fwd.total_return_pct:+.1f}%", fwd_ret_class)}
                    {bloomberg_row("Alpha vs SPY", f"{bt_fwd.alpha_pct:+.1f}%", fwd_alpha_class)}
                    {bloomberg_row("Sharpe Ratio", f"{bt_fwd.sharpe_ratio:.2f}", fwd_sharpe_class)}
                    {bloomberg_row("Max Drawdown", f"{bt_fwd.max_drawdown_pct:.1f}%", "neg")}
                    {bloomberg_row("Volatility", f"{bt_fwd.volatility_pct:.1f}%", "neutral")}
                    {bloomberg_row("Rebalances", str(bt_fwd.rebalance_count), "neutral")}
                    {bloomberg_row("Universe", str(len(bt_fwd.universe)) + " stocks", "neutral")}
                    {bloomberg_row("Benchmark", f"{bt_fwd.benchmark_return_pct:+.1f}%", "neutral")}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.caption(f"_{bt_fwd.note}_")

    # --------------------------------------------------------
    # THE INSIGHT
    # --------------------------------------------------------
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(section_header("💡", "The Key Insight", "", ""), unsafe_allow_html=True)

    st.markdown(f"""
    **{event_name} was {'a sample space expansion event' if triggered_count >= 3 else 'a potential regime change'}.**

    | Concept | Before | After |
    |---------|--------|-------|
    | Investment universe | Tech = FAANG | Tech = FAANG + "AI Infrastructure" |
    | Risk categories | Standard sector risks | + AI disruption, GPU dependency, LLM competition |
    | Portfolio question | "What's your tech allocation?" | "What's your AI exposure?" |

    **This wasn't just a regime shift (P changing) — the game itself changed (X expanded).**

    - **{winner_ticker}** (+{quant.divergence.winner_return_pct:.0f}%): Owned the new infrastructure
    - **{loser_ticker}** ({quant.divergence.loser_return_pct:.0f}%): Business model disrupted
    - **The paradox**: Sample space expanded BUT entropy decreased — new category, dominated by few players
    """)

    # --------------------------------------------------------
    # MULTI-EVENT COMPARISON
    # --------------------------------------------------------
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(section_header("📊", "Multi-Event Comparison", "", ""), unsafe_allow_html=True)
    st.caption("Compare expansion signals across major market disruptions side by side")

    COMPARISON_EVENTS = {
        "ChatGPT Launch": {"winner": "NVDA", "loser": "CHGG", "start": "2022-11-30", "end": "2024-12-01"},
        "COVID Crash": {"winner": "ZM", "loser": "AAL", "start": "2020-01-01", "end": "2021-06-01"},
        "iPhone Launch": {"winner": "AAPL", "loser": "NOK", "start": "2007-06-29", "end": "2009-06-29"},
        "Bitcoin ETF Approval": {"winner": "COIN", "loser": "PYPL", "start": "2024-01-10", "end": "2025-01-10"},
        "Tesla Mass Market": {"winner": "TSLA", "loser": "F", "start": "2017-07-01", "end": "2019-07-01"},
        "Streaming Wars": {"winner": "NFLX", "loser": "DIS", "start": "2013-02-01", "end": "2015-02-01"},
    }

    selected_events = st.multiselect(
        "Select events to compare",
        list(COMPARISON_EVENTS.keys()),
        default=["ChatGPT Launch", "COVID Crash", "iPhone Launch"],
        key="compare_events",
    )

    if selected_events and st.button("Compare Events", type="primary", key="compare_btn"):
        comparison_results = []

        with st.spinner("Fetching data for all events..."):
            for event_key in selected_events:
                cfg = COMPARISON_EVENTS[event_key]
                try:
                    tickers = [cfg["winner"], cfg["loser"]]
                    prices = fetch_prices(tickers, cfg["start"], cfg["end"])

                    winner_p = prices[cfg["winner"]] if cfg["winner"] in prices.columns else None
                    loser_p = prices[cfg["loser"]] if cfg["loser"] in prices.columns else None

                    winner_ret = ((winner_p.iloc[-1] / winner_p.iloc[0]) - 1) * 100 if winner_p is not None and len(winner_p) > 1 else 0
                    loser_ret = ((loser_p.iloc[-1] / loser_p.iloc[0]) - 1) * 100 if loser_p is not None and len(loser_p) > 1 else 0
                    divergence = float(winner_ret - loser_ret)

                    if winner_p is not None and len(winner_p) > 30:
                        from ssed.quant_signals import shannon_entropy as se
                        rets = winner_p.pct_change().dropna()
                        ent = se(rets.values)
                    else:
                        ent = 0

                    comparison_results.append({
                        "Event": event_key,
                        "Winner": cfg["winner"],
                        "Loser": cfg["loser"],
                        "Winner Return": f"{float(winner_ret):+.1f}%",
                        "Loser Return": f"{float(loser_ret):+.1f}%",
                        "Divergence": f"{divergence:+.0f}%",
                        "Entropy": f"{ent:.3f}",
                        "Period": f"{cfg['start']} to {cfg['end']}",
                        "_divergence": divergence,
                        "_winner_ret": float(winner_ret),
                        "_loser_ret": float(loser_ret),
                    })
                except Exception as e:
                    comparison_results.append({
                        "Event": event_key,
                        "Winner": cfg["winner"],
                        "Loser": cfg["loser"],
                        "Winner Return": "N/A",
                        "Loser Return": "N/A",
                        "Divergence": "N/A",
                        "Entropy": "N/A",
                        "Period": f"{cfg['start']} to {cfg['end']}",
                        "_divergence": 0,
                        "_winner_ret": 0,
                        "_loser_ret": 0,
                    })

        if comparison_results:
            st.session_state["comparison_results"] = comparison_results

    if "comparison_results" in st.session_state:
        comparison_results = st.session_state["comparison_results"]

        display_df = pd.DataFrame(comparison_results).drop(columns=["_divergence", "_winner_ret", "_loser_ret"])
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        events = [r["Event"] for r in comparison_results]
        divs = [r["_divergence"] for r in comparison_results]
        winner_rets = [r["_winner_ret"] for r in comparison_results]
        loser_rets = [r["_loser_ret"] for r in comparison_results]

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=events, y=winner_rets,
            name="Winner Return",
            marker_color=C["green"],
            marker=dict(line=dict(width=0), cornerradius=4),
        ))
        fig_comp.add_trace(go.Bar(
            x=events, y=loser_rets,
            name="Loser Return",
            marker_color=C["red"],
            marker=dict(line=dict(width=0), cornerradius=4),
        ))
        fig_comp.update_layout(
            **CHART_LAYOUT,
            barmode="group",
            yaxis_title="Return (%)",
            height=400,
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        div_colors = [C["red"] if d > 200 else C["yellow"] if d > 100 else C["green"] for d in divs]
        fig_div = go.Figure()
        fig_div.add_trace(go.Bar(
            x=events, y=divs,
            marker_color=div_colors,
            text=[f"{d:+.0f}%" for d in divs],
            textposition="outside",
            textfont=dict(color=C["text"], family="JetBrains Mono, monospace", size=11),
        ))
        fig_div.update_layout(
            **CHART_LAYOUT,
            yaxis_title="Winner-Loser Divergence (%)",
            height=350,
        )
        st.plotly_chart(fig_div, use_container_width=True)

        st.caption(
            "Higher divergence suggests stronger creative destruction — "
            "a key indicator of sample space expansion vs normal regime shift"
        )

    # --------------------------------------------------------
    # EXPORT REPORT
    # --------------------------------------------------------
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(section_header("📄", "Export Report", "", ""), unsafe_allow_html=True)

    if st.button("Generate Report", type="secondary"):
        with st.spinner("Generating report..."):
            report_lines = [
                "SSED — SAMPLE SPACE EXPANSION DETECTOR",
                "=" * 50,
                f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "",
                "EVENT ANALYSIS",
                "-" * 50,
                f"Event: {event_name}",
                f"Date: {event_date_str}",
                f"Analysis Period: {event_date_str} to {analysis_end_str}",
                f"Winner: {winner_ticker} ({quant.divergence.winner_return_pct:+.0f}%)",
                f"Loser: {loser_ticker} ({quant.divergence.loser_return_pct:.0f}%)",
                "",
                "LAYER 1: QUANTITATIVE SIGNALS",
                "-" * 50,
                f"HMM Regime: {quant.hmm.regime_label} (p={quant.hmm.regime_probability:.2f})",
                f"Entropy Z-Score: {quant.entropy.entropy_zscore:.2f}",
                f"Divergence: {quant.divergence.total_divergence_pct:+.0f}%",
                f"HHI Change: {quant.concentration.hhi_change:+.4f}",
                "",
                "LAYER 2: NARRATIVE SIGNALS",
                "-" * 50,
                f"News Articles: {news_signals.article_count}",
                f"Avg Sentiment: {news_signals.avg_sentiment:+.3f}",
                f"Sentiment Trend: {news_signals.sentiment_trend}",
                f"Novel Themes: {', '.join(news_signals.novel_theme_counts.keys()) if news_signals.novel_theme_counts else 'None'}",
            ]
            report_lines += [
                "",
                "LONG-SHORT BACKTEST",
                "-" * 50,
                f"Long: {', '.join(bt.long_tickers)}",
                f"Short: {', '.join(bt.short_tickers)}",
                f"Total Return: {bt.total_return_pct:+.1f}%",
                f"Alpha vs SPY: {bt.alpha_pct:+.1f}%",
                f"Sharpe Ratio: {bt.sharpe_ratio:.2f}",
                f"Max Drawdown: {bt.max_drawdown_pct:.1f}%",
                f"Volatility: {bt.volatility_pct:.1f}%",
                f"Trading Days: {bt.trading_days}",
                "",
                "SIGNAL CONVERGENCE",
                "-" * 50,
                f"Signals Triggered: {triggered_count}/4",
            ]
            for i, sig in enumerate(summary_data["Signal"]):
                report_lines.append(
                    f"  {summary_data['Triggered'][i]} {sig}: {summary_data['Value'][i]}"
                )

            if "classification" in st.session_state:
                r = st.session_state["classification"]
                report_lines += [
                    "",
                    "AI CLASSIFICATION",
                    "-" * 50,
                    f"Classification: {r.classification.value}",
                    f"Confidence: {r.confidence.value}",
                    f"Reasoning: {r.reasoning}",
                ]

            report_lines += [
                "",
                "=" * 50,
                "SSED | MGMT 69000: Mastering AI for Finance",
                "Purdue University",
            ]

            report_text = "\n".join(report_lines)

            st.download_button(
                label="Download Report (.txt)",
                data=report_text,
                file_name=f"ssed_report_{event_date_str}.txt",
                mime="text/plain",
            )
            st.text(report_text)


# ============================================================
# SECTOR HEATMAP & LIVE SCANNER (always available)
# ============================================================

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(section_header("🗺️", "Sector Heatmap — Live Market Scanner", "LIVE", "label-live"), unsafe_allow_html=True)
st.caption(
    "Real-time scan across all S&P 500 sectors for expansion/contraction signals"
)

scan_col1, scan_col2 = st.columns([1, 3])
with scan_col1:
    lookback = st.selectbox("Lookback Period", [90, 180, 365], index=1)

if st.button("Scan Markets", type="primary", key="scan_btn"):
    with st.spinner("Scanning all sectors... (fetching live data)"):
        sector_signals = scan_sectors(lookback_days=lookback)
        st.session_state["sector_signals"] = sector_signals

        movers = scan_market_movers(lookback_days=lookback)
        st.session_state["market_movers"] = movers

if "sector_signals" in st.session_state:
    sector_signals = st.session_state["sector_signals"]

    if sector_signals:
        sectors = [s.sector for s in sector_signals]
        expansion_scores = [s.expansion_score for s in sector_signals]
        returns = [s.return_pct for s in sector_signals]
        divergences = [s.divergence_score for s in sector_signals]
        entropies = [s.entropy_score for s in sector_signals]

        fig_hm = go.Figure()

        hm_colors = []
        for score in expansion_scores:
            if score > 0.6:
                hm_colors.append(C["red"])
            elif score > 0.4:
                hm_colors.append(C["yellow"])
            else:
                hm_colors.append(C["green"])

        fig_hm.add_trace(go.Bar(
            x=sectors,
            y=expansion_scores,
            marker_color=hm_colors,
            text=[f"{s:.3f}" for s in expansion_scores],
            textposition="outside",
            textfont=dict(color=C["text"], family="JetBrains Mono, monospace", size=11),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Expansion Score: %{y:.3f}<br>"
                "<extra></extra>"
            ),
        ))

        fig_hm.update_layout(
            **CHART_LAYOUT,
            yaxis_title="Expansion Score (0-1)",
            height=400,
        )
        fig_hm.update_yaxes(range=[0, max(expansion_scores) * 1.3])
        st.plotly_chart(fig_hm, use_container_width=True)

        st.markdown(f"""
        <div style="display:flex; gap:16px; justify-content:center; padding:8px 0; font-size:0.75rem;">
            <span style="color:{C['red']};">&#9679; High expansion (&gt;0.6)</span>
            <span style="color:{C['yellow']};">&#9679; Moderate (0.4-0.6)</span>
            <span style="color:{C['green']};">&#9679; Low/normal (&lt;0.4)</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<p style="font-size:0.85rem; font-weight:600; color:{C["text_bright"]}; margin-top:16px;">Sector Detail</p>', unsafe_allow_html=True)
        df_sectors = pd.DataFrame([
            {
                "Sector": s.sector,
                "ETF": s.etf,
                "Return": f"{s.return_pct:+.1f}%",
                "Volatility": f"{s.volatility:.1f}%",
                "Entropy": f"{s.entropy_score:.3f}",
                "Divergence": f"{s.divergence_score:.1f}%",
                "Expansion Score": f"{s.expansion_score:.3f}",
                "Signal": "\U0001f534" if s.expansion_score > 0.6 else "\U0001f7e1" if s.expansion_score > 0.4 else "\U0001f7e2",
            }
            for s in sector_signals
        ])
        st.dataframe(df_sectors, use_container_width=True, hide_index=True)

        high_signal = [s for s in sector_signals if s.expansion_score > 0.5]
        if high_signal:
            st.warning(
                f"**Sectors showing elevated expansion signals:** "
                f"{', '.join(s.sector for s in high_signal)}"
            )

    if "market_movers" in st.session_state and not st.session_state["market_movers"].empty:
        st.markdown(f'<p style="font-size:0.85rem; font-weight:600; color:{C["text_bright"]}; margin-top:16px;">Top Movers — Winners & Losers</p>', unsafe_allow_html=True)
        st.caption("Biggest divergences across all tracked stocks")

        movers_df = st.session_state["market_movers"]

        mov_col1, mov_col2 = st.columns(2)
        with mov_col1:
            st.markdown(f'<p style="font-size:0.8rem; font-weight:600; color:{C["green"]};">Top Winners</p>', unsafe_allow_html=True)
            winners = movers_df[movers_df["Return"] > 0].head(5)
            if not winners.empty:
                st.dataframe(winners, use_container_width=True, hide_index=True)
        with mov_col2:
            st.markdown(f'<p style="font-size:0.8rem; font-weight:600; color:{C["red"]};">Top Losers</p>', unsafe_allow_html=True)
            losers = movers_df[movers_df["Return"] < 0].tail(5).sort_values("Return")
            if not losers.empty:
                st.dataframe(losers, use_container_width=True, hide_index=True)

        if len(movers_df) >= 2:
            top_winner_ret = movers_df["Return"].max()
            top_loser_ret = movers_df["Return"].min()
            spread = top_winner_ret - top_loser_ret
            st.metric(
                "Winner-Loser Spread",
                f"{spread:.1f}%",
                delta="Wide divergence" if spread > 100 else "Normal range",
            )

# ============================================================
# AI-POWERED APP Q&A (always visible)
# ============================================================

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(section_header("❓", "AI-Powered App Q&A", "NON-DETERMINISTIC · GPT-4.1-NANO", "label-ai"), unsafe_allow_html=True)
st.caption(
    "Ask the AI anything — how SSED works, what the signals mean, how to interpret results, "
    "or questions about the current analysis. Powered by GPT-4.1-nano (non-deterministic)."
)

APP_QA_SYSTEM_PROMPT = """You are the AI help assistant for SSED (Sample Space Expansion Detector), a financial analytics tool built for MGMT 69000 at Purdue University.

WHAT SSED DOES:
SSED detects when the investment universe itself changes (sample space expansion) vs when probabilities shift within a known universe (regime shift). It fuses quantitative signals with LLM-powered narrative analysis.

ARCHITECTURE — 3 LAYERS:

Layer 1: Quantitative Signals (deterministic, no AI)
- HMM Regime Detection: Gaussian Hidden Markov Model with 3 volatility states (low/medium/high). Fitted via hmmlearn. Regimes sorted by variance. Deteriorating log-likelihood suggests the model is seeing data it wasn't trained on.
- Shannon Entropy: Rolling entropy of return distributions. Low entropy = concentrated returns (few stocks dominating). Z-score < -2 is unusual.
- Winner/Loser Divergence: Spread between suspected winner and loser. >500% suggests creative destruction, not normal rotation. Velocity measures acceleration.
- HHI Concentration: Herfindahl-Hirschman Index comparing sector weights before/after event. Increasing HHI = market dominated by fewer sectors.

Layer 2: Narrative Signals (AI-powered, non-deterministic)
- News sentiment scored by GPT-4.1-nano (-1 to +1)
- Novel theme detection: flags terms like "AI infrastructure", "generative AI" as new category indicators
- Falls back to keyword heuristics without OpenAI key

Layer 3: AI Fusion & Classification (non-deterministic)
- OpenAI o4-mini reasons over all signals via function calling (6 tools, strict: True)
- Produces typed RegimeClassification via Pydantic: classification, confidence, reasoning, key evidence
- Classifications: sample_space_expansion, regime_shift, mean_reversion, inconclusive
- Runs automatically on every analysis — different reasoning each time

KEY CONCEPTS:
- Regime Shift (P changed): Probabilities change within known universe. Same assets, different dynamics. Example: tariff shock.
- Sample Space Expansion (X changed): Universe itself changes. New asset classes emerge, old ones die. Example: ChatGPT created "AI infrastructure".
- The detection signal: when BOTH statistical measures (entropy, HMM, divergence, HHI) AND narrative measures (novel themes, new terminology) simultaneously indicate the model is breaking down.

FEATURES:
- Long-Short Backtest: Dollar-neutral strategy (50% long winners, 50% short losers). Metrics: total return, alpha, Sharpe, max drawdown.
- Multi-Event Comparison: Compare expansion signals across historical disruptions side by side.
- Sector Heatmap: Live scan of all 11 S&P 500 sectors for expansion/contraction signals.
- AI Market Narrative: GPT-4.1-nano generates a summary paragraph interpreting all signals (non-deterministic).
- AI Chat: Ask questions about current analysis results.
- App Q&A (this feature): Ask about methodology and how the app works.

DATA SOURCES:
- Market prices: yfinance (Yahoo Finance, no API key required)
- News: NewsAPI (free tier)
- SEC filings: EDGAR API
- AI: OpenAI (o4-mini for classification, GPT-4.1-nano for sentiment/narrative/chat)

SIGNAL CONVERGENCE THRESHOLDS:
- Divergence > 500%
- Entropy z-score < -2
- HHI increasing > 0.02
- Novel narrative themes present
When 3+ of 4 signals trigger → sample space expansion. 2 → likely expansion. <2 → regime shift.

Answer questions clearly and concisely. Use specific numbers and examples when relevant. If asked about something outside the app, redirect to app-related topics.
You can answer BOTH questions about how the app works AND questions about the current analysis results (if an analysis has been run)."""

if "qa_messages" not in st.session_state:
    st.session_state["qa_messages"] = []

# Suggested questions
if not st.session_state["qa_messages"]:
    st.markdown(f"""
    <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px;">
        <span style="background:{C['card']}; border:1px solid {C['card_border']}; border-radius:20px;
            padding:6px 14px; font-size:0.78rem; color:{C['text']};">
            What is sample space expansion?
        </span>
        <span style="background:{C['card']}; border:1px solid {C['card_border']}; border-radius:20px;
            padding:6px 14px; font-size:0.78rem; color:{C['text']};">
            How does the HMM regime detection work?
        </span>
        <span style="background:{C['card']}; border:1px solid {C['card_border']}; border-radius:20px;
            padding:6px 14px; font-size:0.78rem; color:{C['text']};">
            What makes the AI non-deterministic?
        </span>
        <span style="background:{C['card']}; border:1px solid {C['card_border']}; border-radius:20px;
            padding:6px 14px; font-size:0.78rem; color:{C['text']};">
            How do I interpret the entropy z-score?
        </span>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state["qa_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if qa_input := st.chat_input("Ask about SSED — methodology, signals, analysis results, or how it works..."):
    st.session_state["qa_messages"].append({"role": "user", "content": qa_input})
    with st.chat_message("user"):
        st.markdown(qa_input)

    with st.chat_message("assistant"):
        if has_openai:
            try:
                from openai import OpenAI
                client = OpenAI()
                # Build dynamic system prompt with analysis data if available
                qa_prompt = APP_QA_SYSTEM_PROMPT
                if st.session_state.get("running") and "quant" in st.session_state:
                    q = st.session_state["quant"]
                    qa_prompt += f"\n\nCURRENT ANALYSIS DATA:\n"
                    qa_prompt += f"Event: {event_name} on {event_date_str}\n"
                    qa_prompt += f"Winner: {winner_ticker} ({q.divergence.winner_return_pct:+.0f}%)\n"
                    qa_prompt += f"Loser: {loser_ticker} ({q.divergence.loser_return_pct:.0f}%)\n"
                    qa_prompt += f"HMM Regime: {q.hmm.regime_label} (p={q.hmm.regime_probability:.2f})\n"
                    qa_prompt += f"Entropy Z-Score: {q.entropy.entropy_zscore:.2f}\n"
                    qa_prompt += f"Divergence: {q.divergence.total_divergence_pct:+.0f}%\n"
                    qa_prompt += f"HHI Change: {q.concentration.hhi_change:+.4f}\n"
                    if "bt" in st.session_state:
                        b = st.session_state["bt"]
                        qa_prompt += f"Backtest: Long {', '.join(b.long_tickers)} / Short {', '.join(b.short_tickers)}\n"
                        qa_prompt += f"Total Return: {b.total_return_pct:+.1f}%, Alpha: {b.alpha_pct:+.1f}%, Sharpe: {b.sharpe_ratio:.2f}\n"
                    if "classification" in st.session_state:
                        r = st.session_state["classification"]
                        qa_prompt += f"AI Classification: {r.classification.value} (confidence: {r.confidence.value})\n"
                        qa_prompt += f"Reasoning: {r.reasoning}\n"
                messages = [{"role": "system", "content": qa_prompt}]
                messages += [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state["qa_messages"]
                ]
                with st.spinner("Thinking..."):
                    response = client.chat.completions.create(
                        model="gpt-4.1-nano",
                        messages=messages,
                        max_tokens=600,
                    )
                    reply = response.choices[0].message.content
            except Exception as e:
                reply = f"OpenAI API error: {e}"
        else:
            reply = (
                "App Q&A requires an OpenAI API key. Set `OPENAI_API_KEY` in your `.env` file.\n\n"
                "**In the meantime, here's a quick overview:**\n\n"
                "SSED uses 3 layers to detect sample space expansion:\n"
                "- **Layer 1** (Quant): HMM regimes, Shannon entropy, divergence, HHI\n"
                "- **Layer 2** (Narrative): News sentiment + novel theme detection via GPT-4.1-nano\n"
                "- **Layer 3** (Fusion): o4-mini reasons over all signals via function calling\n\n"
                "When 3+ of 4 convergence signals trigger, the event is classified as sample space expansion."
            )

        st.markdown(reply)
        st.session_state["qa_messages"].append({"role": "assistant", "content": reply})

# Footer
st.markdown(f"""
<div class="app-footer">
    <strong>SSED</strong> &mdash; Sample Space Expansion Detector &nbsp;&middot;&nbsp;
    MGMT 69000: Mastering AI for Finance &nbsp;&middot;&nbsp;
    Purdue University
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------------
# DEFAULT VIEW (before analysis runs)
# --------------------------------------------------------
if not st.session_state.get("running"):
    st.markdown(f"""
    <div style="text-align: center; padding: 48px 20px; animation: fadeInUp 0.6s ease-out;">
        <p style="font-size: 1.1rem; color: {C['text_dim']}; margin-bottom: 24px;">
            Configure parameters in the sidebar and click <strong style="color:{C['accent']};">Run Analysis</strong> to begin
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="layer-card l1 delay-1">
            <h4>📈 Layer 1: Quantitative</h4>
            <ul>
                <li>HMM regime detection (3 states)</li>
                <li>Rolling Shannon entropy</li>
                <li>Winner/loser divergence</li>
                <li>HHI concentration index</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="layer-card l2 delay-2">
            <h4>📰 Layer 2: Narrative</h4>
            <ul>
                <li>News sentiment (GPT-4.1-nano)</li>
                <li>Novel theme detection</li>
                <li>Narrative shift analysis</li>
                <li>Category emergence signals</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="layer-card l3 delay-3">
            <h4>🧠 Layer 3: AI Fusion</h4>
            <ul>
                <li>OpenAI o4-mini reasoning</li>
                <li>6 function-calling tools</li>
                <li>Typed classification output</li>
                <li>Evidence chain generation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
