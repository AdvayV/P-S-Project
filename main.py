import streamlit as st
import pandas as pd
import numpy as np
import heapq
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from scipy import stats

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

st.set_page_config(
    page_title="Stock Analysis Dashboard",
    layout="wide"
)

st.markdown("""
<style>
    /* ── Global Overrides ─────────────────────────────────────────── */
    [data-testid="stAppViewContainer"] { background: #f7f9fc; }
    [data-testid="stHeader"]          { background: transparent; }

    /* ── Section Headers ──────────────────────────────────────────── */
    .section-header {
        font-size: 1.4rem; font-weight: 800; letter-spacing: -0.3px;
        background: linear-gradient(135deg, #1a73e8 0%, #6c5ce7 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-top: 1.2rem; margin-bottom: 0.6rem;
    }

    /* ── Info Boxes (glassmorphism) ────────────────────────────────── */
    .info-box {
        background: rgba(240, 244, 255, 0.85);
        backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
        border-radius: 12px; padding: 14px 18px;
        border: 1px solid rgba(26, 115, 232, 0.18);
        margin-bottom: 14px; font-size: 0.88rem;
        color: #333; line-height: 1.55;
        box-shadow: 0 2px 12px rgba(26,115,232,0.06);
    }

    /* ── Explanation Boxes (beginner tips) ─────────────────────────── */
    .explain-box {
        background: linear-gradient(135deg, #fffbe6 0%, #fff9e0 100%);
        border-left: 4px solid #f9a825; border-radius: 10px;
        padding: 14px 18px; margin: 10px 0 16px 0;
        font-size: 0.87rem; color: #5a4e00; line-height: 1.55;
    }
    .explain-box b { color: #e65100; }

    /* ── Metric Cards ─────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: #ffffff; border-radius: 12px;
        padding: 16px 14px 12px; text-align: center;
        border: 1px solid #e8ecf1;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(26,115,232,0.10);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important; font-weight: 600;
        color: #666 !important; text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.35rem !important; font-weight: 800;
        color: #1a1a2e !important;
    }

    /* ── Tabs ─────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: #ffffff;
        border-radius: 12px; padding: 4px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; padding: 10px 20px;
        font-weight: 600; font-size: 0.88rem;
        transition: background 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1a73e8, #6c5ce7) !important;
        color: #fff !important; border-radius: 8px;
    }

    /* ── Buttons ───────────────────────────────────────────────────── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1a73e8, #6c5ce7);
        border: none; border-radius: 8px; font-weight: 700;
        letter-spacing: 0.3px; transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(26,115,232,0.3);
    }

    /* ── Data Frames ──────────────────────────────────────────────── */
    [data-testid="stDataFrame"]  { border-radius: 10px; overflow: hidden; }

    /* ── Sidebar ──────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f4ff 0%, #e8ecf8 100%);
    }

    /* ── Dividers ─────────────────────────────────────────────────── */
    hr { border: none; height: 1px; background: linear-gradient(90deg, transparent, #c8d0e8, transparent); margin: 1.5rem 0; }

    /* ── Summary Tab — Hero Banner ────────────────────────────────── */
    .summary-hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px; padding: 28px 32px; margin-bottom: 20px;
        color: #ffffff; position: relative; overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.18);
    }
    .summary-hero::before {
        content: ''; position: absolute; top: -50%; right: -20%;
        width: 400px; height: 400px; border-radius: 50%;
        background: radial-gradient(circle, rgba(108,92,231,0.15) 0%, transparent 70%);
    }
    .summary-hero .hero-title {
        font-size: 1.6rem; font-weight: 900; letter-spacing: -0.5px;
        margin-bottom: 4px;
    }
    .summary-hero .hero-sub {
        font-size: 0.9rem; color: rgba(255,255,255,0.65); margin-bottom: 16px;
    }
    .summary-hero .hero-return {
        font-size: 2.2rem; font-weight: 900; letter-spacing: -1px;
    }
    .summary-hero .hero-return-label {
        font-size: 0.82rem; color: rgba(255,255,255,0.55);
        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px;
    }

    /* ── Summary Tab — Score Card ─────────────────────────────────── */
    .score-card {
        background: #ffffff; border-radius: 14px; padding: 18px 20px;
        border: 1px solid #e8ecf1;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; text-align: center; min-height: 120px;
    }
    .score-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(26,115,232,0.12);
    }
    .score-card .sc-label {
        font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.6px; color: #888; margin-bottom: 6px;
        width: 100%; text-align: center;
    }
    .score-card .sc-value {
        font-size: 1.6rem; font-weight: 900; color: #1a1a2e;
        width: 100%; text-align: center;
    }
    .score-card .sc-delta {
        font-size: 0.82rem; font-weight: 600; margin-top: 2px;
        width: 100%; text-align: center;
    }

    /* ── Summary Tab — Analysis Card ─────────────────────────────── */
    .analysis-card {
        background: #ffffff; border-radius: 14px; padding: 20px 22px;
        border: 1px solid #e8ecf1; margin-bottom: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    }
    .analysis-card .ac-header {
        font-size: 1.05rem; font-weight: 800; color: #1a1a2e;
        margin-bottom: 10px; display: flex; align-items: center; gap: 8px;
    }
    .analysis-card .ac-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 7px 0; border-bottom: 1px solid #f0f2f5;
        font-size: 0.88rem; color: #444;
    }
    .analysis-card .ac-row:last-child { border-bottom: none; }
    .analysis-card .ac-val {
        font-weight: 700; color: #1a1a2e;
    }

    /* ── Summary Tab — Model Comparison Card ─────────────────────── */
    .model-compare-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
        border-radius: 14px; padding: 22px 24px;
        border: 1px solid rgba(26,115,232,0.15);
        margin-bottom: 16px;
        box-shadow: 0 3px 16px rgba(26,115,232,0.06);
    }
    .model-compare-card .mc-title {
        font-size: 1.1rem; font-weight: 800; color: #1a1a2e;
        margin-bottom: 6px;
    }
    .model-compare-card .mc-subtitle {
        font-size: 0.82rem; color: #777; margin-bottom: 14px;
    }
    .model-compare-card .mc-row {
        display: flex; align-items: center; gap: 12px;
        padding: 10px 14px; border-radius: 10px; margin-bottom: 8px;
        background: rgba(255,255,255,0.7);
        border: 1px solid #e8ecf1;
    }
    .model-compare-card .mc-icon {
        font-size: 1.5rem; flex-shrink: 0;
    }
    .model-compare-card .mc-label {
        font-size: 0.85rem; font-weight: 700; color: #1a1a2e;
    }
    .model-compare-card .mc-desc {
        font-size: 0.8rem; color: #666; margin-top: 2px;
    }
    .model-compare-card .mc-value {
        margin-left: auto; font-size: 1.15rem; font-weight: 800;
        color: #1a73e8; flex-shrink: 0;
    }

    /* ── Summary Tab — Verdict Card ──────────────────────────────── */
    .verdict-card {
        border-radius: 14px; padding: 22px 26px;
        margin: 16px 0; position: relative; overflow: hidden;
    }
    .verdict-card .vc-title {
        font-size: 1.2rem; font-weight: 800; margin-bottom: 10px;
    }
    .verdict-card .vc-body {
        font-size: 0.92rem; line-height: 1.7; color: #333;
    }
    .verdict-card .vc-body b { color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div style="text-align:center; padding: 10px 0 0;">'
    '<span style="font-size:2.2rem; font-weight:900; '
    'background: linear-gradient(135deg, #1a73e8, #6c5ce7); '
    '-webkit-background-clip: text; -webkit-text-fill-color: transparent;">'
    '📊 Stock Analysis &amp; ML Dashboard</span><br>'
    '<span style="color:#666; font-size:0.95rem;">'
    'Understand stocks through data — no finance degree needed</span></div>',
    unsafe_allow_html=True,
)
st.markdown("")

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#fafbfe",
    font=dict(color="#333333"),
    xaxis=dict(showgrid=True, gridcolor="#e8ecf1", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#e8ecf1", zeroline=False),
    legend=dict(bgcolor="#ffffff", bordercolor="#ddd", borderwidth=1),
    margin=dict(l=50, r=30, t=50, b=50),
    hovermode="x unified",
)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window=period).mean()
    loss  = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def detect_currency_symbol(identifier):
    val = (identifier or "").upper().strip()
    if ".NS" in val or ".BO" in val or "^NSEI" in val or "^BSESN" in val:
        return "₹"
    return "$"

def detect_currency_symbol_from_file(uploaded_file):
    if uploaded_file is None:
        return "$"
    filename = getattr(uploaded_file, "name", "")
    return detect_currency_symbol(filename)

def fmt_price(value, symbol):
    return f"{symbol}{value:,.2f}"

def load_and_clean_csv(uploaded_file):
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)
    if "Date" not in df.columns and df.iloc[0].iloc[0] == "Date":
        df.columns = df.iloc[0]
        df = df.drop(index=0).reset_index(drop=True)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    present = [c for c in numeric_cols if c in df.columns]
    df = df.dropna(subset=present).reset_index(drop=True)
    return df, numeric_cols

def add_features(df):
    df = df.copy()
    df["Rolling_Mean_20"] = df["Close"].rolling(window=20).mean()
    df["Rolling_Std_20"] = df["Close"].rolling(window=20).std()
    df["Expected_High_68"] = df["Rolling_Mean_20"] + df["Rolling_Std_20"]
    df["Expected_Low_68"] = df["Rolling_Mean_20"] - df["Rolling_Std_20"]
    df["Daily_Return"]  = df["Close"].pct_change()
    df["Volatility_10"] = df["Daily_Return"].rolling(window=10).std()
    df["RSI_14"]        = compute_rsi(df["Close"], 14)
    df["Close_next"]    = df["Close"].shift(-1)
    df["Price_Change"]  = df["Close_next"] - df["Close"]
    return df

def compute_health_snapshot(df):
    returns = df["Daily_Return"].dropna()
    if returns.empty or len(df) < 2:
        return {
            "health_score": 0.0,
            "health_word": "Weak",
            "health_color": "#e63946",
            "total_return": 0.0,
            "win_rate": 0.0,
        }

    start_price = df["Close"].iloc[0]
    end_price = df["Close"].iloc[-1]
    total_return = ((end_price - start_price) / start_price) * 100
    win_rate = (returns > 0).mean() * 100

    df_t = df[["Close"]].dropna().copy()
    df_t["i"] = np.arange(len(df_t))
    lr = LinearRegression().fit(df_t[["i"]], df_t["Close"])
    r2 = lr.score(df_t[["i"]], df_t["Close"])

    latest_rsi = df["RSI_14"].dropna().iloc[-1] if not df["RSI_14"].dropna().empty else 50
    avg_vol = df["Volatility_10"].mean()
    rec_vol = df["Volatility_10"].iloc[-10:].mean() if len(df) >= 10 else avg_vol

    trend_score = min(max((total_return + 30) / 60 * 100, 0), 100)
    rsi_score = max(100 - abs(latest_rsi - 50) * 2, 0)
    wr_score = win_rate
    vol_ratio = rec_vol / avg_vol if avg_vol > 0 else 1
    vol_score = max(0, 100 - abs(vol_ratio - 1) * 200)
    r2_sc = r2 * 100
    health_score = min(max((trend_score + rsi_score + wr_score + vol_score + r2_sc) / 5, 0), 100)

    if health_score >= 65:
        health_word = "Healthy"
        health_color = "#00a86b"
    elif health_score >= 40:
        health_word = "Mixed"
        health_color = "#ff9800"
    else:
        health_word = "Weak"
        health_color = "#e63946"

    return {
        "health_score": health_score,
        "health_word": health_word,
        "health_color": health_color,
        "total_return": total_return,
        "win_rate": win_rate,
    }

def build_live_chart(hist, date_col, chart_style, ticker):
    UP   = "#00a86b"
    DOWN = "#e63946"

    hist["RSI"] = compute_rsi(hist["Close"], 14)
    hist["Rolling_Mean_20"] = hist["Close"].rolling(window=20).mean()
    hist["Rolling_Std_20"] = hist["Close"].rolling(window=20).std()
    hist["Expected_High_68"] = hist["Rolling_Mean_20"] + hist["Rolling_Std_20"]
    hist["Expected_Low_68"] = hist["Rolling_Mean_20"] - hist["Rolling_Std_20"]
    
    pct_chg = (hist["Close"].iloc[-1] - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2] * 100 if len(hist) > 1 else 0

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.22, 0.23],
        vertical_spacing=0.03,
        subplot_titles=[
            f"{ticker.upper()} Price & 68% Probable Range",
            "Volume",
            "RSI (14)"
        ],
    )

    if chart_style == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=hist[date_col],
            open=hist["Open"], high=hist["High"],
            low=hist["Low"],   close=hist["Close"],
            name="OHLC",
            increasing_line_color=UP,  decreasing_line_color=DOWN,
            increasing_fillcolor=UP,   decreasing_fillcolor=DOWN,
        ), row=1, col=1)
    else:
        line_color = UP if pct_chg >= 0 else DOWN
        fig.add_trace(go.Scatter(
            x=hist[date_col], y=hist["Close"],
            name="Close", mode="lines",
            line=dict(color=line_color, width=1.8),
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=hist[date_col].tolist() + hist[date_col].tolist()[::-1],
        y=hist["Expected_High_68"].tolist() + hist["Expected_Low_68"].tolist()[::-1],
        fill="toself", fillcolor="rgba(26,115,232,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="68% Probable Range",
        hoverinfo="skip"
    ), row=1, col=1)

    bar_colors = [UP if c >= o else DOWN for c, o in zip(hist["Close"], hist["Open"])]
    fig.add_trace(go.Bar(
        x=hist[date_col], y=hist["Volume"],
        name="Volume", marker_color=bar_colors,
        marker_opacity=0.75,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=hist[date_col], y=hist["RSI"],
        name="RSI(14)", mode="lines",
        line=dict(color="#8e24aa", width=1.5),
    ), row=3, col=1)

    fig.add_hline(y=70, line_color=DOWN, line_dash="dash", line_width=0.8, row=3, col=1)
    fig.add_hline(y=30, line_color=UP,   line_dash="dash", line_width=0.8, row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor=DOWN, opacity=0.08, row=3, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor=UP,   opacity=0.08, row=3, col=1)

    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=7,  label="1W", step="day",   stepmode="backward"),
                dict(count=1,  label="1M", step="month", stepmode="backward"),
                dict(count=3,  label="3M", step="month", stepmode="backward"),
                dict(count=6,  label="6M", step="month", stepmode="backward"),
                dict(count=1,  label="1Y", step="year",  stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="#f0f4ff", activecolor="#1a73e8",
            font=dict(color="#333"),
        ),
        rangeslider=dict(visible=False),
        row=1, col=1,
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=680,
        title=dict(text=f"<b>{ticker.upper()}</b> Interactive Chart",
                   font=dict(color="#ffffff", size=14)),
        showlegend=True,
        xaxis_rangeslider_visible=False,
    )
    for ann in fig.layout.annotations:
        ann.font.color = "#333"

    fig.update_yaxes(gridcolor="#e8ecf1", showgrid=True)
    return fig

if "active_stock" not in st.session_state:
    st.session_state["active_stock"] = "Stock 1"

def render_stock_selector(tab_key):
    st.markdown("**Stock Selector**")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Stock 1", key=f"sel_stock1_{tab_key}", use_container_width=True):
            st.session_state["active_stock"] = "Stock 1"
    with c2:
        if st.button("Stock 2", key=f"sel_stock2_{tab_key}", use_container_width=True):
            st.session_state["active_stock"] = "Stock 2"
    st.caption(f"Currently viewing: **{st.session_state['active_stock']}**")
    return st.session_state["active_stock"]


# Modular tab integration
tab_live, tab_csv, tab_ml, tab_stats, tab_mc, tab_anomaly, tab_summary = st.tabs([
    "📡 Live Chart", "📂 CSV Analysis", "🤖 ML Prediction", "📊 Statistics",
    "🎲 Monte Carlo", "🔍 Anomaly Detection", "📋 Summary"
])

from pathlib import Path as _Path
_TABS_DIR = _Path(__file__).resolve().parent / "tabs"
def _run_tab(filename):
    code = (_TABS_DIR / filename).read_text(encoding="utf-8")
    exec(compile(code, str(_TABS_DIR / filename), "exec"), globals())

with tab_live:
    _run_tab("live.py")
with tab_csv:
    _run_tab("csv_analysis.py")
with tab_ml:
    _run_tab("ml_prediction.py")
with tab_stats:
    _run_tab("statistics.py")
with tab_mc:
    _run_tab("monte_carlo.py")
with tab_anomaly:
    _run_tab("anomaly_detection.py")
with tab_summary:
    _run_tab("summary.py")
