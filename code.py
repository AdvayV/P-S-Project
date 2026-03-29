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

tab_live, tab_csv, tab_ml, tab_stats, tab_mc, tab_anomaly, tab_summary = st.tabs([
    "📡 Live Chart", "📂 CSV Analysis", "🤖 ML Prediction", "📊 Statistics",
    "🎲 Monte Carlo", "🔍 Anomaly Detection", "📋 Summary"
])

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

with tab_live:
    st.markdown('<div class="section-header">Real-Time Interactive Price Chart</div>', unsafe_allow_html=True)
    active_stock = render_stock_selector("live")

    if not YFINANCE_AVAILABLE:
        st.error("yfinance not installed. Run: pip install yfinance then restart the app.")
    else:
        st.markdown(
            '<div class="info-box">'
            '<b>💡 What is this?</b> Type a stock ticker (like AAPL for Apple or RELIANCE.NS for Reliance) '
            'and see its price chart in real time. The <b>shaded band</b> shows where the price '
            'is <i>most likely</i> to stay (68% of the time) based on recent behavior.</div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="explain-box">'
            '<b>🔰 New to stocks?</b> A <b>ticker</b> is a short code for a company on the stock market '
            '(e.g. GOOGL = Google). '
            '<b>Candlestick charts</b> show 4 prices per day: Open, High, Low, Close — green means '
            'the price went up, red means it went down. '
            '<b>RSI</b> (Relative Strength Index) measures momentum: above 70 = possibly overpriced, '
            'below 30 = possibly underpriced. '
            '<b>Volume</b> is how many shares were traded — high volume = lots of interest.</div>',
            unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            ticker_input = st.text_input("Stock Ticker", value="AAPL", placeholder="e.g. AAPL, RELIANCE.NS")
        with c2:
            live_period   = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"], index=2)
        with c3:
            live_interval = st.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "1d"], index=5)
        with c4:
            chart_type   = st.radio("Style", ["Line", "Candlestick"])

        col_btn, col_auto = st.columns([1, 3])
        with col_btn:
            fetch_btn = st.button("Fetch / Refresh", type="primary")
        with col_auto:
            auto_refresh = st.toggle("Auto-Refresh")
            refresh_secs = st.slider("Interval (sec)", 10, 120, 30, disabled=not auto_refresh)

        live_status  = st.empty()
        live_metrics = st.empty()
        live_chart   = st.empty()

        def fetch_and_draw(ticker, period, interval, style):
            live_status.info(f"Fetching {ticker.upper()} ...")
            try:
                hist = yf.Ticker(ticker.strip()).history(period=period, interval=interval)
                if hist.empty:
                    live_status.error("No data. Check ticker or try a different interval.")
                    return
                hist.index = pd.to_datetime(hist.index)
                hist = hist.reset_index()
                date_col = hist.columns[0]

                latest  = hist["Close"].iloc[-1]
                prev    = hist["Close"].iloc[-2] if len(hist) > 1 else latest
                change  = latest - prev
                pct_chg = (change / prev) * 100 if prev != 0 else 0

                with live_metrics.container():
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Last Price",   f"${latest:.2f}", f"{change:+.2f}")
                    m2.metric("Change %",     f"{pct_chg:+.2f}%")
                    m3.metric("Period High",  f"${hist['High'].max():.2f}")
                    m4.metric("Period Low",   f"${hist['Low'].min():.2f}")
                    m5.metric("Total Volume", f"{int(hist['Volume'].sum()):,}")

                fig = build_live_chart(hist, date_col, style, ticker)
                live_chart.plotly_chart(fig, use_container_width=True)
                live_status.success(
                    f"Last updated: {pd.Timestamp.now().strftime('%H:%M:%S')} | "
                    f"Drag to zoom, Double-click to reset, Hover for details"
                )
            except Exception as ex:
                live_status.error(f"Error: {ex}")

        if fetch_btn:
            fetch_and_draw(ticker_input, live_period, live_interval, chart_type)

        if auto_refresh:
            fetch_and_draw(ticker_input, live_period, live_interval, chart_type)
            time.sleep(refresh_secs)
            st.rerun()

with tab_csv:
    st.markdown('<div class="section-header">CSV Upload & Technical Analysis</div>', unsafe_allow_html=True)
    active_stock = render_stock_selector("csv")
    st.markdown(
        '<div class="info-box">'
        '<b>📂 Upload your stock CSV</b> with columns: Date, Open, High, Low, Close, Volume. '
        'You can generate one from the <b>Stock CSV Generator</b> tool (generate_csv.py).</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="explain-box">'
        '<b>🔰 What happens here?</b> Once you upload a file, the dashboard automatically calculates: '
        '<b>Win Rate</b> (% of days the stock went up), '
        '<b>95% Max Drop</b> (the worst daily loss you\'d see 95% of the time), '
        '<b>RSI</b> (momentum indicator), and a <b>68% Probable Range</b> band that shows '
        'where the price is statistically most likely to be. '
        'Think of it as a weather forecast — but for stock prices.</div>',
        unsafe_allow_html=True)

    uploaded_file_1 = st.file_uploader("Upload Stock 1 CSV", type="csv", key="csv_stock_1")
    uploaded_file_2 = st.file_uploader("Upload Stock 2 CSV (optional, for comparison)", type="csv", key="csv_stock_2")

    active_uploaded_file = uploaded_file_1 if active_stock == "Stock 1" else uploaded_file_2
    uploaded_file = active_uploaded_file if active_uploaded_file else uploaded_file_1

    if uploaded_file:
        df, numeric_cols = load_and_clean_csv(uploaded_file)
        df = add_features(df)
        df_ml = df.dropna().reset_index(drop=True)

        with st.sidebar:
            st.header("CSV Controls")
            show_raw      = st.checkbox("Show Raw Data", value=False)
            chart_style_c = st.radio("Chart Style", ["Line", "Candlestick"], key="csv_chart")
            st.subheader("Top Price Changes")
            N         = st.slider("Top N days", 1, 30, 5)
            direction = st.selectbox("Direction", ("Max Gains", "Max Losses", "Both"))
            st.subheader("Exit Simulator")
            min_d = df["Date"].min().to_pydatetime()
            max_d = df["Date"].max().to_pydatetime()
            inv_date    = st.date_input("Investment Date", value=min_d, min_value=min_d, max_value=max_d)
            inv_capital = st.number_input("Capital", min_value=0.0, value=10000.0, step=100.0)

        if show_raw:
            st.subheader("Raw Data")
            st.dataframe(df.head(100), use_container_width=True, height=250)

        q1, q2, q3, q4, q5 = st.columns(5)
        
        hist_prob_gain = (df['Daily_Return'] > 0).mean() * 100
        var_95 = df['Daily_Return'].quantile(0.05) * 100
        
        q1.metric("Total Days Analyzed", len(df))
        q2.metric("Historical Win Rate", f"{hist_prob_gain:.1f}%")
        q3.metric("95% Prob. Max Drop", f"{var_95:.2f}%")
        q4.metric("Avg Close", f"{df['Close'].mean():.2f}")
        q5.metric("Avg Daily Return", f"{df['Daily_Return'].mean()*100:.3f}%")

        st.subheader("Top Price Change Days")
        col_g, col_l = st.columns(2)
        if direction in ("Max Gains", "Both"):
            gains = heapq.nlargest(N, enumerate(df_ml["Price_Change"]), key=lambda x: x[1])
            gain_data = [[df_ml.iloc[i]["Date"].strftime("%Y-%m-%d"),
                          f"{df_ml.iloc[i]['Close']:.2f}",
                          f"{df_ml.iloc[i]['Close_next']:.2f}",
                          f"+{c:.2f}"] for i, c in gains]
            with col_g:
                st.markdown("**Biggest Gains**")
                st.dataframe(pd.DataFrame(gain_data, columns=["Date","Close","Next Close","Change"]),
                             use_container_width=True, hide_index=True)

        if direction in ("Max Losses", "Both"):
            losses = heapq.nsmallest(N, enumerate(df_ml["Price_Change"]), key=lambda x: x[1])
            loss_data = [[df_ml.iloc[i]["Date"].strftime("%Y-%m-%d"),
                          f"{df_ml.iloc[i]['Close']:.2f}",
                          f"{df_ml.iloc[i]['Close_next']:.2f}",
                          f"{c:.2f}"] for i, c in losses]
            with col_l:
                st.markdown("**Biggest Losses**")
                st.dataframe(pd.DataFrame(loss_data, columns=["Date","Close","Next Close","Change"]),
                             use_container_width=True, hide_index=True)

        st.subheader("Optimal Past Exit Simulation")
        try:
            target = pd.to_datetime(inv_date)
            valid  = df[df["Date"] >= target]
            if not valid.empty:
                buy_p = valid.iloc[0]["Close"]; buy_d = valid.iloc[0]["Date"]
                fut   = valid.iloc[1:]
                if not fut.empty:
                    best = fut.loc[fut["Close"].idxmax()]
                    profit = (best["Close"] - buy_p) * (inv_capital / buy_p)
                    pct_p  = ((best["Close"] - buy_p) / buy_p) * 100
                    e1, e2, e3 = st.columns(3)
                    e1.metric("Purchase Price",       f"{buy_p:.2f}",       f"on {buy_d.date()}")
                    e2.metric("Best Exit Price",      f"{best['Close']:.2f}",f"on {best['Date'].date()}")
                    e3.metric("Max Potential Profit", f"{profit:,.2f}",      f"{pct_p:.2f}%")
                else:
                    st.info("No future data after selected date.")
            else:
                st.warning("No data for selected date or later.")
        except Exception as ex:
            st.error(f"Simulation error: {ex}")

        st.subheader("Interactive Price & Probability Chart")

        UP, DOWN = "#00a86b", "#e63946"
        fig_csv = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.55, 0.22, 0.23],
            vertical_spacing=0.03,
            subplot_titles=["Price & 68% Probable Range", "Volume", "RSI (14)"],
        )

        if chart_style_c == "Candlestick":
            fig_csv.add_trace(go.Candlestick(
                x=df["Date"], open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name="OHLC",
                increasing_line_color=UP, decreasing_line_color=DOWN,
                increasing_fillcolor=UP, decreasing_fillcolor=DOWN,
            ), row=1, col=1)
        else:
            fig_csv.add_trace(go.Scatter(
                x=df["Date"], y=df["Close"], name="Close",
                mode="lines", line=dict(color="#1a73e8", width=2),
            ), row=1, col=1)

        fig_csv.add_trace(go.Scatter(
            x=df["Date"].tolist() + df["Date"].tolist()[::-1],
            y=df["Expected_High_68"].tolist() + df["Expected_Low_68"].tolist()[::-1],
            fill="toself", fillcolor="rgba(26,115,232,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="68% Probable Range"
        ), row=1, col=1)

        bar_colors = [UP if r >= 0 else DOWN for r in df["Daily_Return"].fillna(0)]
        fig_csv.add_trace(go.Bar(
            x=df["Date"], y=df["Volume"], name="Volume",
            marker_color=bar_colors, marker_opacity=0.7,
        ), row=2, col=1)

        fig_csv.add_trace(go.Scatter(
            x=df["Date"], y=df["RSI_14"], name="RSI(14)",
            mode="lines", line=dict(color="#8e24aa", width=1.5),
        ), row=3, col=1)
        fig_csv.add_hline(y=70, line_color=DOWN, line_dash="dash", row=3, col=1)
        fig_csv.add_hline(y=30, line_color=UP,   line_dash="dash", row=3, col=1)
        fig_csv.add_hrect(y0=70, y1=100, fillcolor=DOWN, opacity=0.08, row=3, col=1)
        fig_csv.add_hrect(y0=0,  y1=30,  fillcolor=UP,   opacity=0.08, row=3, col=1)

        fig_csv.update_xaxes(
            rangeselector=dict(
                buttons=[
                    dict(count=1,  label="1M", step="month", stepmode="backward"),
                    dict(count=3,  label="3M", step="month", stepmode="backward"),
                    dict(count=6,  label="6M", step="month", stepmode="backward"),
                    dict(count=1,  label="1Y", step="year",  stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor="#f0f4ff", activecolor="#1a73e8",
                font=dict(color="#333"),
            ),
            row=1, col=1,
        )
        fig_csv.update_layout(**PLOTLY_LAYOUT, height=680)
        for ann in fig_csv.layout.annotations:
            ann.font.color = "#333"
        fig_csv.update_yaxes(gridcolor="#e8ecf1")

        st.plotly_chart(fig_csv, use_container_width=True)

        # ── Auto-Generated Chart Analysis ─────────────────────────────────────
        st.subheader("Chart Analysis (Auto-Generated)")
        st.markdown(
            '<div class="info-box">This analysis is automatically generated from the uploaded data '
            'using basic statistics and linear regression. It describes the trend, volatility, '
            'momentum, and key price levels in simple words.</div>',
            unsafe_allow_html=True)

        analysis_parts = []

        # --- 1. Overall Trend (Linear Regression slope) ---
        from sklearn.linear_model import LinearRegression as LR_Analysis
        df_trend = df[["Close"]].dropna().copy()
        df_trend["day_num"] = np.arange(len(df_trend))
        lr_trend = LR_Analysis()
        lr_trend.fit(df_trend[["day_num"]], df_trend["Close"])
        slope = lr_trend.coef_[0]
        r2_trend = lr_trend.score(df_trend[["day_num"]], df_trend["Close"])

        total_days = len(df)
        start_price = df["Close"].iloc[0]
        end_price = df["Close"].iloc[-1]
        total_change_pct = ((end_price - start_price) / start_price) * 100

        if slope > 0 and total_change_pct > 5:
            trend_word = "**strong uptrend**"
            trend_emoji = "📈"
        elif slope > 0:
            trend_word = "**mild uptrend**"
            trend_emoji = "↗️"
        elif slope < 0 and total_change_pct < -5:
            trend_word = "**strong downtrend**"
            trend_emoji = "📉"
        elif slope < 0:
            trend_word = "**mild downtrend**"
            trend_emoji = "↘️"
        else:
            trend_word = "**sideways movement**"
            trend_emoji = "➡️"

        analysis_parts.append(
            f"{trend_emoji} **Trend:** Over the {total_days} trading days analyzed, "
            f"the stock moved from **{start_price:.2f}** to **{end_price:.2f}** "
            f"({total_change_pct:+.2f}%), showing a {trend_word}. "
            f"A linear regression line fitted to the closing prices has a daily slope of "
            f"**{slope:.4f}** (R² = {r2_trend:.3f}), meaning the trend explains "
            f"**{r2_trend*100:.1f}%** of the price movement."
        )

        # --- 2. Volatility Assessment ---
        avg_vol = df["Volatility_10"].mean()
        recent_vol = df["Volatility_10"].iloc[-10:].mean() if len(df) >= 10 else avg_vol
        max_daily = df["Daily_Return"].max() * 100
        min_daily = df["Daily_Return"].min() * 100

        if recent_vol > avg_vol * 1.3:
            vol_status = "**higher than average**"
            vol_note = "The stock has been swinging more aggressively in recent days, indicating increased uncertainty."
        elif recent_vol < avg_vol * 0.7:
            vol_status = "**lower than average**"
            vol_note = "The stock has been relatively calm recently, with smaller day-to-day moves."
        else:
            vol_status = "**near average levels**"
            vol_note = "The stock's recent price swings are in line with its historical behavior."

        analysis_parts.append(
            f"📊 **Volatility:** The average 10-day volatility is **{avg_vol:.5f}**, and "
            f"recent volatility is {vol_status} at **{recent_vol:.5f}**. {vol_note} "
            f"The biggest single-day gain was **{max_daily:+.2f}%** and the biggest drop "
            f"was **{min_daily:+.2f}%**."
        )

        # --- 3. RSI / Momentum ---
        latest_rsi = df["RSI_14"].dropna().iloc[-1] if not df["RSI_14"].dropna().empty else 50

        if latest_rsi > 70:
            rsi_status = "**overbought** (above 70)"
            rsi_note = "This suggests the stock may have risen too fast and could be due for a pullback."
        elif latest_rsi < 30:
            rsi_status = "**oversold** (below 30)"
            rsi_note = "This suggests the stock may have fallen too much and could be due for a bounce."
        elif latest_rsi > 55:
            rsi_status = "**bullish territory** (above 55)"
            rsi_note = "Momentum is leaning positive, indicating buyers are in control."
        elif latest_rsi < 45:
            rsi_status = "**bearish territory** (below 45)"
            rsi_note = "Momentum is leaning negative, indicating sellers have more influence."
        else:
            rsi_status = "**neutral** (45-55 range)"
            rsi_note = "There is no strong momentum in either direction right now."

        analysis_parts.append(
            f"⚡ **Momentum (RSI):** The current RSI(14) value is **{latest_rsi:.1f}**, "
            f"which is in {rsi_status}. {rsi_note}"
        )

        # --- 4. Support & Resistance ---
        recent = df.tail(60) if len(df) >= 60 else df
        support = recent["Low"].min()
        resistance = recent["High"].max()
        current = df["Close"].iloc[-1]
        dist_to_support = ((current - support) / current) * 100
        dist_to_resist = ((resistance - current) / current) * 100

        analysis_parts.append(
            f"🔒 **Key Levels:** Based on the last {len(recent)} trading days, "
            f"the nearest **support** (lowest low) is at **{support:.2f}** "
            f"({dist_to_support:.1f}% below current price) and the nearest **resistance** "
            f"(highest high) is at **{resistance:.2f}** ({dist_to_resist:.1f}% above). "
            f"The current price of **{current:.2f}** sits "
            f"{'closer to resistance, suggesting limited upside in the short term.' if dist_to_resist < dist_to_support else 'closer to support, suggesting there is room to move up.'}"
        )

        # --- 5. Win Rate & Risk ---
        win_rate = (df["Daily_Return"] > 0).mean() * 100
        var_95 = df["Daily_Return"].quantile(0.05) * 100
        avg_ret = df["Daily_Return"].mean() * 100

        analysis_parts.append(
            f"🎯 **Historical Probability:** Out of all trading days, "
            f"**{win_rate:.1f}%** ended with a positive return (win rate). "
            f"The average daily return is **{avg_ret:.4f}%**. "
            f"On the worst 5% of days, the stock dropped by at least **{abs(var_95):.2f}%** "
            f"(this is the 95% Value-at-Risk)."
        )

        # --- 6. Summary Verdict ---
        signals = []
        if total_change_pct > 5:
            signals.append("uptrend")
        elif total_change_pct < -5:
            signals.append("downtrend")
        if latest_rsi > 70:
            signals.append("overbought")
        elif latest_rsi < 30:
            signals.append("oversold")
        if recent_vol > avg_vol * 1.3:
            signals.append("high volatility")
        if win_rate > 55:
            signals.append("positive historical bias")

        summary_line = ", ".join(signals) if signals else "no strong signals"
        analysis_parts.append(
            f"📝 **Summary:** Key signals detected: **{summary_line}**. "
            f"This analysis is based purely on historical data and basic statistics — "
            f"past performance does not guarantee future results."
        )

        for part in analysis_parts:
            st.markdown(part)
            st.markdown("")

        st.markdown("---")

        with st.expander("Correlation Matrix"):
            corr = df[[c for c in numeric_cols if c in df.columns] +
                       ["Daily_Return", "Volatility_10", "RSI_14"]].corr()
            st.dataframe(corr.style.background_gradient(cmap="RdYlGn").format(precision=3),
                         use_container_width=True)
    else:
        st.info("Upload a stock CSV to get started.")

with tab_ml:
    st.markdown('<div class="section-header">Machine Learning: Next-Day Price Prediction</div>', unsafe_allow_html=True)
    active_stock = render_stock_selector("ml")
    st.markdown(
        '<div class="info-box">'
        '<b>🤖 How does this work?</b> We train a Linear Regression model on the stock\'s past prices, '
        'volume, momentum, and probability bands to predict what tomorrow\'s closing price might be.</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="explain-box">'
        '<b>🔰 Plain English:</b> Imagine you have years of exam scores and study hours. '
        'Linear Regression draws the best-fit line through that data to predict your next score. '
        'Here, instead of exams, we\'re using stock data.<br><br>'
        '<b>RMSE</b> = average error in dollars (lower is better).<br>'
        '<b>R² Score</b> = how well the model fits (1.0 = perfect, 0 = no fit).<br>'
        '<b>Feature Importance</b> = which factors (price, volume, RSI) matter most for the prediction.</div>',
        unsafe_allow_html=True)

    if uploaded_file:
        features = ["Open", "High", "Low", "Close", "Volume", "Expected_High_68", "Expected_Low_68", "RSI_14"]
        features = [f for f in features if f in df_ml.columns]
        X = df_ml[features]; y = df_ml["Close_next"]

        test_pct = st.slider("Test set size (%)", 10, 40, 20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, shuffle=False, test_size=test_pct/100)
        test_dates = df_ml["Date"].iloc[-len(y_test):]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        pm1, pm2, pm3 = st.columns(3)
        pm1.metric("RMSE (lower = better)",  f"{rmse:.2f}")
        pm2.metric("R2 Score (1 = perfect)", f"{r2:.4f}")
        pm3.metric("Training samples",       len(X_train))

        fig_ml = go.Figure()
        fig_ml.add_trace(go.Scatter(
            x=test_dates, y=y_test.values, name="Actual",
            mode="lines", line=dict(color="#1a73e8", width=2),
            hovertemplate="Date: %{x}<br>Actual: %{y:.2f}<extra></extra>",
        ))
        fig_ml.add_trace(go.Scatter(
            x=test_dates, y=y_pred, name="Predicted",
            mode="lines", line=dict(color="#ff7043", width=2, dash="dash"),
            hovertemplate="Date: %{x}<br>Predicted: %{y:.2f}<extra></extra>",
        ))
        fig_ml.add_trace(go.Scatter(
            x=list(test_dates) + list(test_dates[::-1]),
            y=list(y_test.values) + list(y_pred[::-1]),
            fill="toself", fillcolor="rgba(171,71,188,0.07)",
            line=dict(color="rgba(0,0,0,0)"), name="Error band", showlegend=False,
        ))
        fig_ml.update_layout(
            **PLOTLY_LAYOUT,
            title="<b>Next-Day Close: Actual vs Predicted</b>",
            yaxis_title="Price", height=420,
        )
        st.plotly_chart(fig_ml, use_container_width=True)

        st.subheader("Feature Importance (Coefficients)")
        coef_df = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
        coef_df = coef_df.sort_values("Coefficient", key=abs, ascending=True)
        bar_c   = ["#00a86b" if v >= 0 else "#e63946" for v in coef_df["Coefficient"]]
        fig_coef= go.Figure(go.Bar(
            y=coef_df["Feature"], x=coef_df["Coefficient"],
            orientation="h", marker_color=bar_c,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        fig_coef.update_layout(
            **PLOTLY_LAYOUT,
            title="Which features drive the prediction?",
            xaxis_title="Coefficient", height=320,
        )
        st.plotly_chart(fig_coef, use_container_width=True)

        last_pred  = model.predict(X.iloc[[-1]])[0]
        last_close = df_ml["Close"].iloc[-1]
        diff       = last_pred - last_close

        # Store ML results for Summary tab
        st.session_state[f"ml_results_{active_stock}"] = {
            "last_pred": last_pred,
            "last_close": last_close,
            "rmse": rmse,
            "r2": r2,
            "diff": diff,
        }

        st.subheader("Tomorrow's Predicted Close")
        n1, n2, n3 = st.columns(3)
        n1.metric("Today's Close",          f"{last_close:.2f}")
        n2.metric("Predicted Next Close",   f"{last_pred:.2f}", f"{diff:+.2f}")
        n3.metric("Signal", "Likely UP" if diff > 0 else "Likely DOWN")
    else:
        st.info("Upload a CSV in the CSV Analysis tab first.")

with tab_stats:
    st.markdown('<div class="section-header">Statistical & Probability Analysis</div>', unsafe_allow_html=True)
    active_stock = render_stock_selector("stats")
    st.markdown(
        '<div class="info-box">'
        '<b>📈 Deep dive into the numbers.</b> This tab shows how the stock\'s daily returns '
        'are distributed, how volatile it has been, and its cumulative performance over time.</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="explain-box">'
        '<b>🔰 What do these charts mean?</b><br>'
        '• <b>Histogram</b> — Groups daily returns into buckets. A tall bar means many days had that return. '
        'A bell shape = normal behavior; wide spread = risky stock.<br>'
        '• <b>Box Plot</b> — Shows the middle 50% of returns (the box), the median (line), and outliers (dots). '
        'Big outliers = surprise crashes or rallies.<br>'
        '• <b>Cumulative Return</b> — If you invested $100 on day one, this line shows your profit/loss over time.<br>'
        '• <b>Volatility</b> — How wildly the price swings day-to-day. Higher = riskier.<br>'
        '• <b>Skewness</b> — Negative = more crash risk; Positive = more upside surprises.<br>'
        '• <b>Kurtosis</b> — Higher value = more extreme moves than a normal distribution would predict.</div>',
        unsafe_allow_html=True)

    if uploaded_file:
        returns = df["Daily_Return"].dropna()

        st.subheader("Descriptive Statistics")
        desc = returns.describe().to_frame("Value")
        desc.loc["skewness"] = returns.skew()
        desc.loc["kurtosis"] = returns.kurtosis()
        st.dataframe(desc.style.format("{:.6f}"), use_container_width=False)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=returns, nbinsx=60, name="Returns",
                marker_color="#1a73e8", marker_opacity=0.75,
                hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
            ))
            fig_hist.add_vline(x=returns.mean(),   line_color="#e65100",
                               annotation_text="Mean", annotation_position="top right")
            fig_hist.add_vline(x=returns.median(), line_color="#8e24aa", line_dash="dash",
                               annotation_text="Median", annotation_position="top left")
            fig_hist.update_layout(
                **PLOTLY_LAYOUT, title="Histogram of Daily Returns",
                xaxis_title="Daily Return", yaxis_title="Frequency", height=350,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                x=returns, name="Daily Return",
                marker_color="#1a73e8", boxmean="sd",
                hovertemplate="Value: %{x:.5f}<extra></extra>",
            ))
            fig_box.update_layout(
                **PLOTLY_LAYOUT, title="Box Plot of Daily Returns",
                xaxis_title="Daily Return", height=350,
            )
            st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("Cumulative Return Over Time")
        cum = (1 + returns).cumprod() - 1
        color_cum = "#00a86b" if cum.iloc[-1] >= 0 else "#e63946"
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=df["Date"].iloc[1:], y=cum.values * 100,
            name="Cumulative Return", mode="lines",
            fill="tozeroy",
            line=dict(color=color_cum, width=2),
            fillcolor=f"rgba({'0,168,107' if cum.iloc[-1]>=0 else '230,57,70'},0.12)",
            hovertemplate="Date: %{x}<br>Return: %{y:.2f}%<extra></extra>",
        ))
        fig_cum.add_hline(y=0, line_color="#888", line_dash="dash", line_width=0.7)
        fig_cum.update_layout(
            **PLOTLY_LAYOUT,
            title=f"<b>Total Cumulative Return: {cum.iloc[-1]*100:.2f}%</b>",
            yaxis_title="Cumulative Return (%)",
            height=380,
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        st.subheader("Rolling Volatility (10-Day)")
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=df["Date"], y=df["Volatility_10"],
            name="10-Day Volatility", mode="lines",
            fill="tozeroy",
            line=dict(color="#e65100", width=1.8),
            fillcolor="rgba(230,81,0,0.1)",
            hovertemplate="Date: %{x}<br>Volatility: %{y:.5f}<extra></extra>",
        ))
        fig_vol.update_layout(
            **PLOTLY_LAYOUT,
            title="Rolling 10-Day Volatility of Returns",
            yaxis_title="Volatility", height=320,
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    else:
        st.info("Upload a CSV in the CSV Analysis tab first.")

with tab_mc:
    st.markdown('<div class="section-header">Monte Carlo Simulation</div>', unsafe_allow_html=True)
    active_stock = render_stock_selector("mc")

    st.markdown(
        '<div class="info-box">'
        '<b>What is Monte Carlo Simulation?</b><br><br>'
        'Imagine you want to know where a stock price might be in the future. '
        'Nobody can predict it exactly, but we <i>can</i> look at how the stock has behaved in the past '
        '(its average daily change and how wildly it swings) and then roll the dice thousands of times '
        'to create thousands of possible future price paths.<br><br>'
        'Each path is one "what-if" scenario. When we overlay all of them, we get a <b>probability fan</b> '
        'that shows the most likely range the price could land in. '
        'The darker the shaded area, the more likely the price is to be there.'
        '</div>',
        unsafe_allow_html=True)

    if uploaded_file:
        returns = df["Daily_Return"].dropna()
        last_close = df["Close"].iloc[-1]

        st.subheader("Simulation Settings")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            n_simulations = st.slider("Number of simulations", 100, 10000, 1000, step=100,
                                      help="More simulations = smoother probability fan, but slower.")
        with mc2:
            n_days = st.slider("Days to forecast", 5, 252, 30,
                               help="Trading days into the future (252 ~ 1 year).")
        with mc3:
            confidence = st.selectbox("Confidence band", ["68%", "90%", "95%"], index=1,
                                       help="Width of the shaded probability band.")

        conf_map = {"68%": (16, 84), "90%": (5, 95), "95%": (2.5, 97.5)}
        lo_pct, hi_pct = conf_map[confidence]

        run_mc = st.button("Run Monte Carlo Simulation", type="primary")

        if run_mc:
            mu    = returns.mean()
            sigma = returns.std()

            st.markdown(
                f'<div class="info-box">'
                f'Using historical data: <b>mean daily return = {mu*100:.4f}%</b>, '
                f'<b>daily volatility (std dev) = {sigma*100:.4f}%</b>. '
                f'Starting price: <b>{last_close:.2f}</b>.'
                f'</div>', unsafe_allow_html=True)

            np.random.seed(42)
            daily_returns = np.random.normal(mu, sigma, size=(n_days, n_simulations))
            price_paths   = np.zeros((n_days + 1, n_simulations))
            price_paths[0] = last_close
            for t in range(1, n_days + 1):
                price_paths[t] = price_paths[t - 1] * (1 + daily_returns[t - 1])

            days_axis = np.arange(n_days + 1)

            median_path = np.median(price_paths, axis=1)
            lo_band     = np.percentile(price_paths, lo_pct, axis=1)
            hi_band     = np.percentile(price_paths, hi_pct, axis=1)
            p5          = np.percentile(price_paths, 5,  axis=1)
            p95         = np.percentile(price_paths, 95, axis=1)

            # --- Fan chart ---
            fig_mc = go.Figure()

            sample_count = min(100, n_simulations)
            for i in range(sample_count):
                fig_mc.add_trace(go.Scatter(
                    x=days_axis, y=price_paths[:, i],
                    mode="lines", line=dict(color="rgba(100,149,237,0.06)", width=0.5),
                    showlegend=False, hoverinfo="skip",
                ))

            fig_mc.add_trace(go.Scatter(
                x=list(days_axis) + list(days_axis[::-1]),
                y=list(hi_band) + list(lo_band[::-1]),
                fill="toself", fillcolor="rgba(0,210,255,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{confidence} Probability Band",
                hoverinfo="skip",
            ))

            fig_mc.add_trace(go.Scatter(
                x=days_axis, y=median_path,
                mode="lines", name="Median Path",
                line=dict(color="#ff9800", width=2.5),
                hovertemplate="Day %{x}: %{y:.2f}<extra>Median</extra>",
            ))
            fig_mc.add_trace(go.Scatter(
                x=days_axis, y=lo_band,
                mode="lines", name=f"{lo_pct}th Percentile",
                line=dict(color="#ef5350", width=1, dash="dash"),
                hovertemplate="Day %{x}: %{y:.2f}<extra>Low bound</extra>",
            ))
            fig_mc.add_trace(go.Scatter(
                x=days_axis, y=hi_band,
                mode="lines", name=f"{hi_pct}th Percentile",
                line=dict(color="#26a69a", width=1, dash="dash"),
                hovertemplate="Day %{x}: %{y:.2f}<extra>High bound</extra>",
            ))

            fig_mc.update_layout(
                **PLOTLY_LAYOUT,
                title=f"<b>Monte Carlo Simulation — {n_simulations:,} paths over {n_days} days</b>",
                xaxis_title="Trading Days from Today",
                yaxis_title="Simulated Price",
                height=520,
            )
            st.plotly_chart(fig_mc, use_container_width=True)

            # --- Final price distribution ---
            final_prices = price_paths[-1]

            st.subheader("Distribution of Final Prices")
            st.markdown(
                '<div class="info-box">'
                'This histogram shows where the price ended up across all simulations on the last day. '
                'Think of it as answering: "If I could replay the next '
                f'{n_days} days thousands of times, what prices would I see most often?"'
                '</div>', unsafe_allow_html=True)

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=final_prices, nbinsx=80, name="Final Price",
                marker_color="#00d2ff", marker_opacity=0.75,
                hovertemplate="Price: %{x:.2f}<br>Count: %{y}<extra></extra>",
            ))
            fig_dist.add_vline(x=last_close, line_color="#ff9800", line_width=2,
                               annotation_text=f"Start: {last_close:.2f}",
                               annotation_position="top right")
            fig_dist.add_vline(x=np.median(final_prices), line_color="#ab47bc",
                               line_dash="dash", line_width=1.5,
                               annotation_text=f"Median: {np.median(final_prices):.2f}",
                               annotation_position="top left")
            fig_dist.update_layout(
                **PLOTLY_LAYOUT,
                title="Histogram of Simulated Final Prices",
                xaxis_title="Price", yaxis_title="Frequency", height=380,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            # --- Summary metrics ---
            st.subheader("Key Takeaways")
            prob_profit = (final_prices > last_close).mean() * 100
            prob_loss   = 100 - prob_profit
            expected    = np.mean(final_prices)
            worst5      = np.percentile(final_prices, 5)
            best5       = np.percentile(final_prices, 95)

            # Store Monte Carlo results for Summary tab
            st.session_state[f"mc_results_{active_stock}"] = {
                "prob_profit": prob_profit,
                "expected": expected,
                "worst5": worst5,
                "best5": best5,
                "last_close": last_close,
                "mu": mu,
                "sigma": sigma,
            }

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Prob. of Profit",  f"{prob_profit:.1f}%",
                      help="Percentage of simulations that ended above today's price.")
            k2.metric("Expected Price",   f"{expected:.2f}",
                      f"{((expected - last_close)/last_close)*100:+.2f}%",
                      help="Average price across all simulations.")
            k3.metric("Best Case (95th)", f"{best5:.2f}",
                      f"{((best5 - last_close)/last_close)*100:+.2f}%",
                      help="Only 5% of simulations ended above this price.")
            k4.metric("Worst Case (5th)", f"{worst5:.2f}",
                      f"{((worst5 - last_close)/last_close)*100:+.2f}%",
                      help="Only 5% of simulations ended below this price.")

            st.markdown("---")
            st.markdown(
                f"**In plain English:** Based on {n_simulations:,} simulated scenarios over {n_days} trading days, "
                f"there is a **{prob_profit:.0f}% chance** the stock goes up and a **{prob_loss:.0f}% chance** it goes down. "
                f"The most likely outcome (median) is a price of **{np.median(final_prices):.2f}**. "
                f"With {confidence} confidence, the price should land between "
                f"**{lo_band[-1]:.2f}** and **{hi_band[-1]:.2f}**.")

    else:
        st.info("Upload a CSV in the CSV Analysis tab first.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_anomaly:
    st.markdown('<div class="section-header">Anomaly Detection using Isolation Forest</div>',
                unsafe_allow_html=True)
    active_stock = render_stock_selector("anomaly")

    st.markdown(
        '<div class="info-box">'
        '<b>What is an Isolation Forest?</b><br><br>'
        'It is a machine learning algorithm specifically designed to find anomalies. '
        'Instead of looking for normal patterns, it tries to "isolate" each day. '
        'Days with highly unusual trading behaviors (bizarre price swings or massive volume) '
        'are isolated much faster than normal days. <br><br>'
        '<b>How do we detect anomalies?</b><br>'
        'The algorithm assigns an <b>Anomaly Score</b> to every day. Days with scores below '
        'the threshold are flagged as outliers.'
        '</div>',
        unsafe_allow_html=True)

    if uploaded_file:
        st.subheader("Settings")
        ac1, ac2 = st.columns(2)
        with ac1:
            contamination = st.slider("Expected Anomaly Rate (%)", 1.0, 10.0, 3.0, 0.5,
                                      help="What percentage of days do you think are true anomalies? "
                                           "Higher % = more days flagged.")
        with ac2:
            n_estimators = st.slider("Number of Trees", 50, 300, 100, 25,
                                     help="More trees = more robust detection but slightly slower.")

        run_anomaly = st.button("Run Anomaly Detection", type="primary")

        if run_anomaly:
            with st.spinner("Finding anomalies..."):
                from sklearn.ensemble import IsolationForest

                # Prepare features
                feature_cols = ["Open", "High", "Low", "Close", "Volume",
                                "Daily_Return", "Volatility_10", "RSI_14"]
                feature_cols = [c for c in feature_cols if c in df.columns]
                df_ae = df[feature_cols].dropna().copy()
                ae_dates = df.loc[df_ae.index, "Date"]
                ae_close = df.loc[df_ae.index, "Close"]

                # Standardize
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_ae)

                # Train Isolation Forest
                iso_forest = IsolationForest(
                    n_estimators=n_estimators,
                    contamination=contamination / 100.0,
                    random_state=42
                )
                
                # Predictions: -1 for anomalies, 1 for normal
                preds = iso_forest.fit_predict(X_scaled)
                is_anomaly = (preds == -1)
                
                # Anomaly scores (lower = more anomalous)
                anomaly_scores = iso_forest.decision_function(X_scaled)

                anomaly_count = is_anomaly.sum()
                anomaly_pct = (anomaly_count / len(is_anomaly)) * 100

            st.success(f"Analysis complete! Detected **{anomaly_count}** anomalous days "
                       f"out of {len(is_anomaly)} ({anomaly_pct:.1f}%)")

            # Metrics
            st.subheader("Detection Results")
            r1, r2, r3 = st.columns(3)
            r1.metric("Total Days Analyzed", len(is_anomaly))
            r2.metric("Anomalies Found", int(anomaly_count))
            r3.metric("Anomaly Rate", f"{anomaly_pct:.1f}%")

            # --- Chart 1: Price with anomaly markers ---
            st.subheader("Price Chart with Anomalies Highlighted")
            fig_anom = go.Figure()
            fig_anom.add_trace(go.Scatter(
                x=ae_dates, y=ae_close,
                mode="lines", name="Close Price",
                line=dict(color="#1a73e8", width=1.8),
            ))
            fig_anom.add_trace(go.Scatter(
                x=ae_dates[is_anomaly], y=ae_close[is_anomaly],
                mode="markers", name="Anomaly",
                marker=dict(color="#e63946", size=10, symbol="x",
                            line=dict(width=2, color="#b71c1c")),
                hovertemplate="Date: %{x}<br>Price: %{y:.2f}<br>ANOMALY<extra></extra>",
            ))
            fig_anom.update_layout(
                **PLOTLY_LAYOUT,
                title="<b>Closing Price with Detected Anomalies</b>",
                yaxis_title="Price", height=420,
            )
            st.plotly_chart(fig_anom, use_container_width=True)

            # --- Chart 2: Anomaly Score ---
            st.subheader("Anomaly Score per Day")
            st.markdown(
                '<div class="info-box">'
                'The anomaly score shows how unusual a day is. <b>Negative scores</b> indicate anomalies. '
                'The dashed line marks the zero boundary.'
                '</div>', unsafe_allow_html=True)

            fig_err = go.Figure()
            fig_err.add_trace(go.Scatter(
                x=ae_dates, y=anomaly_scores,
                mode="lines", name="Anomaly Score",
                line=dict(color="#8e24aa", width=1.2),
                fill="tozeroy", fillcolor="rgba(142,36,170,0.08)",
                hovertemplate="Date: %{x}<br>Score: %{y:.3f}<extra></extra>",
            ))
            fig_err.add_hline(
                y=0, line_color="#e63946",
                line_dash="dash", line_width=1.5,
                annotation_text="Anomaly Boundary",
                annotation_position="top right",
            )
            fig_err.add_trace(go.Scatter(
                x=ae_dates[is_anomaly], y=anomaly_scores[is_anomaly],
                mode="markers", name="Anomaly",
                marker=dict(color="#e63946", size=8, symbol="diamond"),
                hovertemplate="Date: %{x}<br>Score: %{y:.3f}<br>ANOMALY<extra></extra>",
            ))
            fig_err.update_layout(
                **PLOTLY_LAYOUT,
                title="<b>Isolation Forest Anomaly Scores</b>",
                yaxis_title="Score (Negative = Anomaly)", height=350,
            )
            st.plotly_chart(fig_err, use_container_width=True)

            # --- Anomaly table ---
            st.subheader("Anomalous Days Detail")
            if anomaly_count > 0:
                anom_df = pd.DataFrame({
                    "Date": ae_dates[is_anomaly].dt.strftime("%Y-%m-%d").values,
                    "Close": ae_close[is_anomaly].values,
                    "Daily Return": (df.loc[df_ae.index, "Daily_Return"][is_anomaly] * 100).values,
                    "Score": anomaly_scores[is_anomaly],
                }).sort_values("Score", ascending=True)  # Lower score = more anomalous
                anom_df["Close"] = anom_df["Close"].map("{:.2f}".format)
                anom_df["Daily Return"] = anom_df["Daily Return"].map("{:+.2f}%".format)
                anom_df["Score"] = anom_df["Score"].map("{:.3f}".format)
                st.dataframe(anom_df, use_container_width=True, hide_index=True)
            else:
                st.info("No anomalies detected at this setting.")

            # Plain English
            st.markdown("---")
            st.markdown(
                f"**In plain English:** We used an Isolation Forest algorithm looking at {len(feature_cols)} different "
                f"aspects of the trading day (price, volume, returns, momentum, etc.). The algorithm isolated "
                f"**{anomaly_count} days** ({anomaly_pct:.1f}%) that acted completely differently from the rest of "
                f"the dataset. These are the days marked with an X above, and they often correspond to "
                f"major earnings events, market shocks, or highly unusual volume spikes."
            )
    else:
        st.info("Upload a CSV in the CSV Analysis tab first.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB — SUMMARY  (Revamped)
# ══════════════════════════════════════════════════════════════════════════════
with tab_summary:
    st.markdown('<div class="section-header">📋 Executive Summary Report</div>',
                unsafe_allow_html=True)
    active_stock = render_stock_selector("summary")

    stock1_df = None
    stock2_df = None
    if "uploaded_file_1" in locals() and uploaded_file_1:
        stock1_df, _ = load_and_clean_csv(uploaded_file_1)
        stock1_df = add_features(stock1_df)
    if "uploaded_file_2" in locals() and uploaded_file_2:
        stock2_df, _ = load_and_clean_csv(uploaded_file_2)
        stock2_df = add_features(stock2_df)

    if stock1_df is not None and stock2_df is not None:
        s1 = compute_health_snapshot(stock1_df)
        s2 = compute_health_snapshot(stock2_df)
        better = "Stock 1" if s1["health_score"] >= s2["health_score"] else "Stock 2"
        better_color = s1["health_color"] if better == "Stock 1" else s2["health_color"]
        st.markdown(
            f'<div class="analysis-card">'
            f'<div class="ac-header">⚖️ Stock Comparison (Investor Pick)</div>'
            f'<div class="ac-row"><span>Stock 1 Health Score</span><span class="ac-val" style="color:{s1["health_color"]};">{s1["health_score"]:.0f}/100 · {s1["health_word"]}</span></div>'
            f'<div class="ac-row"><span>Stock 2 Health Score</span><span class="ac-val" style="color:{s2["health_color"]};">{s2["health_score"]:.0f}/100 · {s2["health_word"]}</span></div>'
            f'<div class="ac-row"><span>Stock 1 Total Return</span><span class="ac-val">{s1["total_return"]:+.2f}%</span></div>'
            f'<div class="ac-row"><span>Stock 2 Total Return</span><span class="ac-val">{s2["total_return"]:+.2f}%</span></div>'
            f'<div class="ac-row"><span>Stock 1 Win Rate</span><span class="ac-val">{s1["win_rate"]:.1f}%</span></div>'
            f'<div class="ac-row"><span>Stock 2 Win Rate</span><span class="ac-val">{s2["win_rate"]:.1f}%</span></div>'
            f'<div class="ac-row"><span>Better Pick (by Health Score)</span><span class="ac-val" style="color:{better_color};">{better}</span></div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="explain-box">'
            '<b>🔰 How to use this:</b> The better pick is chosen by the higher Health Score '
            '(trend, momentum, win rate, volatility stability, and trend fit). '
            'If scores are close, compare risk (volatility and VaR) before deciding.'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown("---")

    if uploaded_file:
        # ── Compute all summary data up front ─────────────────────────────
        returns_s = df["Daily_Return"].dropna()
        start_price = df["Close"].iloc[0]
        end_price   = df["Close"].iloc[-1]
        total_return = ((end_price - start_price) / start_price) * 100
        date_start  = df["Date"].iloc[0].strftime("%b %d, %Y")
        date_end    = df["Date"].iloc[-1].strftime("%b %d, %Y")
        win_rate_s  = (returns_s > 0).mean() * 100
        var_95_s    = returns_s.quantile(0.05) * 100
        avg_ret_s   = returns_s.mean() * 100
        std_ret_s   = returns_s.std() * 100

        # Trend
        df_t = df[["Close"]].dropna().copy()
        df_t["i"] = np.arange(len(df_t))
        from sklearn.linear_model import LinearRegression as LR_Sum
        lr_sum = LR_Sum().fit(df_t[["i"]], df_t["Close"])
        slope_s = lr_sum.coef_[0]
        r2_s = lr_sum.score(df_t[["i"]], df_t["Close"])

        if slope_s > 0 and total_return > 5:
            trend_label = "Strong Uptrend"; trend_emoji = "📈"
        elif slope_s > 0:
            trend_label = "Mild Uptrend"; trend_emoji = "↗️"
        elif slope_s < 0 and total_return < -5:
            trend_label = "Strong Downtrend"; trend_emoji = "📉"
        elif slope_s < 0:
            trend_label = "Mild Downtrend"; trend_emoji = "↘️"
        else:
            trend_label = "Sideways"; trend_emoji = "➡️"

        # RSI
        latest_rsi_s = df["RSI_14"].dropna().iloc[-1] if not df["RSI_14"].dropna().empty else 50
        if latest_rsi_s > 70:     rsi_label = "Overbought"; rsi_color = "#e63946"
        elif latest_rsi_s < 30:   rsi_label = "Oversold";   rsi_color = "#e63946"
        elif latest_rsi_s > 55:   rsi_label = "Bullish";    rsi_color = "#00a86b"
        elif latest_rsi_s < 45:   rsi_label = "Bearish";    rsi_color = "#e63946"
        else:                     rsi_label = "Neutral";     rsi_color = "#ff9800"

        # Volatility
        avg_vol_s = df["Volatility_10"].mean()
        rec_vol_s = df["Volatility_10"].iloc[-10:].mean() if len(df) >= 10 else avg_vol_s
        if rec_vol_s > avg_vol_s * 1.3:   vol_label = "High"
        elif rec_vol_s < avg_vol_s * 0.7: vol_label = "Low"
        else:                             vol_label = "Normal"

        # Support / Resistance
        recent_df = df.tail(60) if len(df) >= 60 else df
        support_s = recent_df["Low"].min()
        resist_s  = recent_df["High"].max()

        # Health Score (0-100)
        trend_score = min(max((total_return + 30) / 60 * 100, 0), 100)
        rsi_score   = 100 - abs(latest_rsi_s - 50) * 2
        rsi_score   = max(rsi_score, 0)
        wr_score    = win_rate_s
        vol_ratio   = rec_vol_s / avg_vol_s if avg_vol_s > 0 else 1
        vol_score   = max(0, 100 - abs(vol_ratio - 1) * 200)
        r2_sc       = r2_s * 100
        health_score = (trend_score + rsi_score + wr_score + vol_score + r2_sc) / 5
        health_score = min(max(health_score, 0), 100)

        if health_score >= 65:   health_color = "#00a86b"; health_word = "Healthy"
        elif health_score >= 40: health_color = "#ff9800"; health_word = "Mixed"
        else:                    health_color = "#e63946"; health_word = "Weak"

        ret_color = "#00e676" if total_return >= 0 else "#ff5252"

        # ── 1. HERO BANNER ────────────────────────────────────────────────
        st.markdown(
            f'<div class="summary-hero">'
            f'<div class="hero-title">📊 Stock Analysis Report</div>'
            f'<div class="hero-sub">{len(df)} trading days · {date_start} → {date_end}</div>'
            f'<div style="display:flex; align-items:flex-end; gap:40px; flex-wrap:wrap;">'
            f'<div>'
            f'<div class="hero-return-label">Total Return</div>'
            f'<div class="hero-return" style="color:{ret_color};">{total_return:+.2f}%</div>'
            f'</div>'
            f'<div>'
            f'<div class="hero-return-label">Current Price</div>'
            f'<div class="hero-return" style="font-size:1.6rem;">${end_price:.2f}</div>'
            f'</div>'
            f'<div>'
            f'<div class="hero-return-label">Health Score</div>'
            f'<div class="hero-return" style="font-size:1.6rem; color:{health_color};">'
            f'{health_score:.0f}/100 · {health_word}</div>'
            f'</div>'
            f'</div></div>',
            unsafe_allow_html=True)

        # ── 2. HEALTH SCORE GAUGE ─────────────────────────────────────────
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            title={"text": "Stock Health Score", "font": {"size": 16, "color": "#333"}},
            number={"suffix": "/100", "font": {"size": 36, "color": "#1a1a2e"}},
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#999"),
                bar=dict(color=health_color, thickness=0.3),
                bgcolor="#f0f2f5",
                borderwidth=0,
                steps=[
                    dict(range=[0, 30],  color="#ffebee"),
                    dict(range=[30, 60], color="#fff8e1"),
                    dict(range=[60, 100], color="#e8f5e9"),
                ],
                threshold=dict(line=dict(color="#1a1a2e", width=3), thickness=0.8, value=health_score),
            ),
        ))
        fig_gauge.update_layout(
            height=220, margin=dict(l=30, r=30, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#333"),
        )
        gc1, gc2 = st.columns([1, 1])
        with gc1:
            st.plotly_chart(fig_gauge, use_container_width=True)
        with gc2:
            st.markdown(
                '<div class="analysis-card">'
                '<div class="ac-header">🧮 Score Breakdown</div>'
                f'<div class="ac-row"><span>Trend Strength</span><span class="ac-val">{trend_score:.0f}/100</span></div>'
                f'<div class="ac-row"><span>RSI Balance</span><span class="ac-val">{rsi_score:.0f}/100</span></div>'
                f'<div class="ac-row"><span>Win Rate</span><span class="ac-val">{wr_score:.0f}/100</span></div>'
                f'<div class="ac-row"><span>Volatility Stability</span><span class="ac-val">{vol_score:.0f}/100</span></div>'
                f'<div class="ac-row"><span>Trend Fit (R²)</span><span class="ac-val">{r2_sc:.0f}/100</span></div>'
                '</div>',
                unsafe_allow_html=True)
            st.markdown(
                '<div class="explain-box">'
                '<b>🔰 What is this score?</b> We combined 5 factors — trend direction, '
                'momentum balance (RSI), historical win rate, volatility stability, and how '
                'well a trend line fits — into a single 0–100 score. '
                '<b>60+</b> = healthy, <b>30–60</b> = mixed signals, <b>below 30</b> = weak.</div>',
                unsafe_allow_html=True)

        st.markdown("---")

        # ── 3. KEY METRICS ROW ────────────────────────────────────────────
        km1, km2, km3, km4 = st.columns(4)
        km1.metric("Current Price", f"${end_price:.2f}",
                   f"{((end_price - start_price)/start_price*100):+.1f}% overall")
        km2.metric("Win Rate", f"{win_rate_s:.1f}%",
                   "of days were positive")
        km3.metric("95% Value-at-Risk", f"{var_95_s:.2f}%",
                   "worst daily loss at 95% conf.")
        km4.metric("Avg Daily Return", f"{avg_ret_s:.4f}%",
                   f"Std Dev: {std_ret_s:.4f}%")

        st.markdown("---")

        # ── 4. TWO-COLUMN ANALYSIS GRID ───────────────────────────────────
        left_col, right_col = st.columns(2)

        with left_col:
            # Trend & Momentum Card
            st.markdown(
                '<div class="analysis-card">'
                f'<div class="ac-header">{trend_emoji} Trend & Momentum</div>'
                f'<div class="ac-row"><span>Direction</span><span class="ac-val">{trend_label}</span></div>'
                f'<div class="ac-row"><span>Daily Slope</span><span class="ac-val">{slope_s:.4f}</span></div>'
                f'<div class="ac-row"><span>Trend Fit (R²)</span><span class="ac-val">{r2_s:.4f}</span></div>'
                f'<div class="ac-row"><span>RSI(14)</span><span class="ac-val" style="color:{rsi_color};">{latest_rsi_s:.1f} · {rsi_label}</span></div>'
                f'<div class="ac-row"><span>Total Change</span><span class="ac-val" style="color:{ret_color};">{total_return:+.2f}%</span></div>'
                '</div>',
                unsafe_allow_html=True)

            # Support & Resistance Card
            dist_sup = ((end_price - support_s) / end_price) * 100
            dist_res = ((resist_s - end_price) / end_price) * 100
            st.markdown(
                '<div class="analysis-card">'
                '<div class="ac-header">🔒 Support & Resistance (60-day)</div>'
                f'<div class="ac-row"><span>Support (Low)</span><span class="ac-val">${support_s:.2f}</span></div>'
                f'<div class="ac-row"><span>Distance to Support</span><span class="ac-val">{dist_sup:.1f}% below</span></div>'
                f'<div class="ac-row"><span>Resistance (High)</span><span class="ac-val">${resist_s:.2f}</span></div>'
                f'<div class="ac-row"><span>Distance to Resistance</span><span class="ac-val">{dist_res:.1f}% above</span></div>'
                f'<div class="ac-row"><span>Position</span><span class="ac-val">{"Near Resistance ⚠️" if dist_res < dist_sup else "Near Support ✅"}</span></div>'
                '</div>',
                unsafe_allow_html=True)

        with right_col:
            # Volatility Profile Card
            st.markdown(
                '<div class="analysis-card">'
                '<div class="ac-header">📊 Volatility Profile</div>'
                f'<div class="ac-row"><span>Avg 10-Day Volatility</span><span class="ac-val">{avg_vol_s:.5f}</span></div>'
                f'<div class="ac-row"><span>Recent Volatility</span><span class="ac-val">{rec_vol_s:.5f}</span></div>'
                f'<div class="ac-row"><span>Status</span><span class="ac-val">{vol_label}</span></div>'
                f'<div class="ac-row"><span>Biggest Gain</span><span class="ac-val" style="color:#00a86b;">{returns_s.max()*100:+.2f}%</span></div>'
                f'<div class="ac-row"><span>Biggest Drop</span><span class="ac-val" style="color:#e63946;">{returns_s.min()*100:+.2f}%</span></div>'
                '</div>',
                unsafe_allow_html=True)

            # Distribution Shape Card
            skew_val = returns_s.skew()
            kurt_val = returns_s.kurtosis()
            if skew_val < -0.3:   skew_word = "Left-skewed (crash risk ⚠️)"
            elif skew_val > 0.3:  skew_word = "Right-skewed (upside surprises)"
            else:                 skew_word = "Symmetric ✅"

            if kurt_val > 3:      kurt_word = "Heavy tails (extreme moves likely)"
            elif kurt_val > 1:    kurt_word = "Moderate tails"
            else:                 kurt_word = "Light tails (close to normal)"

            st.markdown(
                '<div class="analysis-card">'
                '<div class="ac-header">🔔 Distribution Shape</div>'
                f'<div class="ac-row"><span>Skewness</span><span class="ac-val">{skew_val:.4f}</span></div>'
                f'<div class="ac-row"><span>Interpretation</span><span class="ac-val">{skew_word}</span></div>'
                f'<div class="ac-row"><span>Kurtosis</span><span class="ac-val">{kurt_val:.4f}</span></div>'
                f'<div class="ac-row"><span>Interpretation</span><span class="ac-val">{kurt_word}</span></div>'
                f'<div class="ac-row"><span>68% Range (latest)</span><span class="ac-val">${df["Expected_Low_68"].iloc[-1]:.2f} — ${df["Expected_High_68"].iloc[-1]:.2f}</span></div>'
                '</div>',
                unsafe_allow_html=True)

        st.markdown("---")

        # ── 5. MODEL COMPARISON ───────────────────────────────────────────
        st.markdown('<div class="section-header">🔬 Model Comparison — What Each Method Says</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="explain-box">'
            '<b>🔰 Why multiple models?</b> No single model is perfect. '
            '<b>Linear Regression</b> looks at patterns in past data to guess tomorrow\'s price. '
            '<b>Monte Carlo</b> runs thousands of random "what-if" scenarios to show a range of possibilities. '
            '<b>Statistical Analysis</b> measures the stock\'s personality — how it tends to behave. '
            '<b>Anomaly Detection</b> flags the unusual days that don\'t fit normal patterns. '
            'By comparing all four, you get a fuller picture.</div>',
            unsafe_allow_html=True)

        # ML Prediction results (compute fresh for summary)
        ml_available = len(df_ml) > 20
        if ml_available:
            features_sum = ["Open", "High", "Low", "Close", "Volume",
                            "Expected_High_68", "Expected_Low_68", "RSI_14"]
            features_sum = [f for f in features_sum if f in df_ml.columns]
            X_s = df_ml[features_sum]; y_s = df_ml["Close_next"]
            X_tr, X_te, y_tr, y_te = train_test_split(X_s, y_s, shuffle=False, test_size=0.2)
            m_s = LinearRegression().fit(X_tr, y_tr)
            rmse_s = np.sqrt(mean_squared_error(y_te, m_s.predict(X_te)))
            r2_ml = r2_score(y_te, m_s.predict(X_te))
            next_p = m_s.predict(X_s.iloc[[-1]])[0]
            diff_p = next_p - df_ml["Close"].iloc[-1]
            ml_signal = "📈 Likely UP" if diff_p > 0 else "📉 Likely DOWN"
            ml_confidence = "High" if r2_ml > 0.9 else "Medium" if r2_ml > 0.7 else "Low"
        else:
            next_p = end_price; diff_p = 0; rmse_s = 0; r2_ml = 0
            ml_signal = "⚠️ Not enough data"; ml_confidence = "N/A"

        _mc = st.session_state.get(f"mc_results_{active_stock}")

        comp1, comp2 = st.columns(2)

        with comp1:
            # Linear Regression Card
            st.markdown(
                '<div class="model-compare-card">'
                '<div class="mc-title">🤖 Linear Regression (ML)</div>'
                '<div class="mc-subtitle">Learns patterns from historical features to predict tomorrow</div>'
                f'<div class="mc-row"><span class="mc-icon">🎯</span><div><div class="mc-label">Predicted Next Close</div>'
                f'<div class="mc-desc">Based on 8 features (price, volume, RSI, bands)</div></div>'
                f'<span class="mc-value">${next_p:.2f}</span></div>'
                f'<div class="mc-row"><span class="mc-icon">📊</span><div><div class="mc-label">Accuracy (R²)</div>'
                f'<div class="mc-desc">How well the model fits test data (1.0 = perfect)</div></div>'
                f'<span class="mc-value">{r2_ml:.4f}</span></div>'
                f'<div class="mc-row"><span class="mc-icon">📏</span><div><div class="mc-label">Error Margin (RMSE)</div>'
                f'<div class="mc-desc">Average error in price units</div></div>'
                f'<span class="mc-value">±${rmse_s:.2f}</span></div>'
                f'<div class="mc-row"><span class="mc-icon">🚦</span><div><div class="mc-label">Signal</div>'
                f'<div class="mc-desc">Confidence: {ml_confidence}</div></div>'
                f'<span class="mc-value">{ml_signal}</span></div>'
                '</div>',
                unsafe_allow_html=True)

            # Statistical Analysis Card
            st.markdown(
                '<div class="model-compare-card">'
                '<div class="mc-title">📐 Statistical Analysis</div>'
                '<div class="mc-subtitle">Measures the stock\'s historical personality using math</div>'
                f'<div class="mc-row"><span class="mc-icon">🏆</span><div><div class="mc-label">Win Rate</div>'
                f'<div class="mc-desc">% of days with positive returns</div></div>'
                f'<span class="mc-value">{win_rate_s:.1f}%</span></div>'
                f'<div class="mc-row"><span class="mc-icon">⚡</span><div><div class="mc-label">Avg Daily Return</div>'
                f'<div class="mc-desc">Expected return on any given day</div></div>'
                f'<span class="mc-value">{avg_ret_s:.4f}%</span></div>'
                f'<div class="mc-row"><span class="mc-icon">🛡️</span><div><div class="mc-label">95% Max Loss (VaR)</div>'
                f'<div class="mc-desc">Worst loss 95% of the time</div></div>'
                f'<span class="mc-value">{var_95_s:.2f}%</span></div>'
                f'<div class="mc-row"><span class="mc-icon">📏</span><div><div class="mc-label">Volatility</div>'
                f'<div class="mc-desc">Recent vs Average: {vol_label}</div></div>'
                f'<span class="mc-value">{rec_vol_s:.5f}</span></div>'
                '</div>',
                unsafe_allow_html=True)

        with comp2:
            # Monte Carlo Card
            if _mc:
                mc_signal = "📈 Bullish" if _mc["prob_profit"] > 55 else "📉 Bearish" if _mc["prob_profit"] < 45 else "➡️ Neutral"
                st.markdown(
                    '<div class="model-compare-card">'
                    '<div class="mc-title">🎲 Monte Carlo Simulation</div>'
                    '<div class="mc-subtitle">Runs thousands of random "what-if" futures based on past behavior</div>'
                    f'<div class="mc-row"><span class="mc-icon">💰</span><div><div class="mc-label">Probability of Profit</div>'
                    f'<div class="mc-desc">% of simulations ending above current price</div></div>'
                    f'<span class="mc-value">{_mc["prob_profit"]:.1f}%</span></div>'
                    f'<div class="mc-row"><span class="mc-icon">🎯</span><div><div class="mc-label">Expected Price</div>'
                    f'<div class="mc-desc">Average across all simulated outcomes</div></div>'
                    f'<span class="mc-value">${_mc["expected"]:.2f}</span></div>'
                    f'<div class="mc-row"><span class="mc-icon">🚀</span><div><div class="mc-label">Best Case (95th)</div>'
                    f'<div class="mc-desc">Only 5% of simulations beat this</div></div>'
                    f'<span class="mc-value">${_mc["best5"]:.2f}</span></div>'
                    f'<div class="mc-row"><span class="mc-icon">⚠️</span><div><div class="mc-label">Worst Case (5th)</div>'
                    f'<div class="mc-desc">Only 5% of simulations were worse</div></div>'
                    f'<span class="mc-value">${_mc["worst5"]:.2f}</span></div>'
                    '</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="model-compare-card">'
                    '<div class="mc-title">🎲 Monte Carlo Simulation</div>'
                    '<div class="mc-subtitle">Run the Monte Carlo tab first to see results here</div>'
                    '</div>',
                    unsafe_allow_html=True)

            # Anomaly Detection Card (summary from session or quick stats)
            st.markdown(
                '<div class="model-compare-card">'
                '<div class="mc-title">🔍 Anomaly Detection (Isolation Forest)</div>'
                '<div class="mc-subtitle">Flags days with bizarre trading behavior unlike the rest</div>'
                f'<div class="mc-row"><span class="mc-icon">🌲</span><div><div class="mc-label">Algorithm</div>'
                f'<div class="mc-desc">Ensemble of random isolation trees</div></div>'
                f'<span class="mc-value">Isolation Forest</span></div>'
                f'<div class="mc-row"><span class="mc-icon">📐</span><div><div class="mc-label">Features Used</div>'
                f'<div class="mc-desc">OHLCV, returns, volatility, RSI</div></div>'
                f'<span class="mc-value">8 features</span></div>'
                f'<div class="mc-row"><span class="mc-icon">🔎</span><div><div class="mc-label">Purpose</div>'
                f'<div class="mc-desc">Find market shocks, earnings surprises, unusual volume</div></div>'
                f'<span class="mc-value">Outlier Detection</span></div>'
                f'<div class="mc-row"><span class="mc-icon">💡</span><div><div class="mc-label">Status</div>'
                f'<div class="mc-desc">Run the Anomaly Detection tab to generate results</div></div>'
                f'<span class="mc-value">→ Run Tab</span></div>'
                '</div>',
                unsafe_allow_html=True)

        st.markdown("---")

        # ── 6. COMBINED ML + MC FORECAST (preserved) ─────────────────────
        _ml_r = st.session_state.get(f"ml_results_{active_stock}")

        if _ml_r and _mc:
            st.markdown('<div class="section-header">🔮 Combined Forecast (ML + Monte Carlo)</div>',
                        unsafe_allow_html=True)

            consensus = (_ml_r["last_pred"] + _mc["expected"]) / 2
            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("ML Predicted Next Close",
                       f"{_ml_r['last_pred']:.2f}", f"{_ml_r['diff']:+.2f} from current")
            fc2.metric("MC Expected Price",
                       f"{_mc['expected']:.2f}",
                       f"{((_mc['expected'] - _mc['last_close']) / _mc['last_close'] * 100):+.2f}%")
            fc3.metric("Consensus Forecast",
                       f"{consensus:.2f}",
                       f"{((consensus - _ml_r['last_close']) / _ml_r['last_close'] * 100):+.2f}%",
                       help="Average of ML prediction and MC expected price.")

            st.markdown("")

            # Composite Signal
            predicted_dir = 1 if _ml_r["diff"] > 0 else -1
            composite = (_ml_r["r2"] * predicted_dir) + (_mc["prob_profit"] / 100 - 0.5)

            if composite > 0.15:
                signal_label = "Bullish"; signal_color = "#00a86b"; signal_emoji = "🟢"
            elif composite < -0.15:
                signal_label = "Bearish"; signal_color = "#e63946"; signal_emoji = "🔴"
            else:
                signal_label = "Neutral"; signal_color = "#ff9800"; signal_emoji = "🟡"

            st.markdown(
                f'<div style="background:{signal_color}12; border-left:4px solid {signal_color}; '
                f'padding:16px 20px; border-radius:8px; margin-bottom:16px;">'
                f'<span style="font-size:1.3rem; font-weight:700; color:{signal_color};">{signal_emoji} Composite Signal: {signal_label}</span><br>'
                f'<span style="color:#555; font-size:0.92rem;">'
                f'Score = (R² × direction) + (P(profit) − 0.5) = '
                f'({_ml_r["r2"]:.3f} × {predicted_dir}) + ({_mc["prob_profit"]/100:.3f} − 0.5) = '
                f'<b>{composite:+.3f}</b></span></div>',
                unsafe_allow_html=True)

            # Mini 30-day fan chart
            st.markdown("**30-Day Monte Carlo Fan Chart**")
            np.random.seed(99)
            _n_sim = 1000; _n_days = 30
            _dr = np.random.normal(_mc["mu"], _mc["sigma"], size=(_n_days, _n_sim))
            _pp = np.zeros((_n_days + 1, _n_sim))
            _pp[0] = _mc["last_close"]
            for _t in range(1, _n_days + 1):
                _pp[_t] = _pp[_t - 1] * (1 + _dr[_t - 1])

            _days = np.arange(_n_days + 1)
            _med  = np.median(_pp, axis=1)
            _p5   = np.percentile(_pp, 5,  axis=1)
            _p95  = np.percentile(_pp, 95, axis=1)

            fig_mini = go.Figure()
            for _i in range(min(80, _n_sim)):
                fig_mini.add_trace(go.Scatter(
                    x=_days, y=_pp[:, _i], mode="lines",
                    line=dict(color="rgba(100,149,237,0.06)", width=0.5),
                    showlegend=False, hoverinfo="skip"))
            fig_mini.add_trace(go.Scatter(
                x=list(_days) + list(_days[::-1]),
                y=list(_p95) + list(_p5[::-1]),
                fill="toself", fillcolor="rgba(0,210,255,0.13)",
                line=dict(color="rgba(0,0,0,0)"), name="90% Band", hoverinfo="skip"))
            fig_mini.add_trace(go.Scatter(
                x=_days, y=_med, mode="lines", name="Median",
                line=dict(color="#ff9800", width=2),
                hovertemplate="Day %{x}: %{y:.2f}<extra>Median</extra>"))
            fig_mini.add_trace(go.Scatter(
                x=_days, y=_p5, mode="lines", name="5th Pctl",
                line=dict(color="#ef5350", width=1, dash="dash"),
                hovertemplate="Day %{x}: %{y:.2f}<extra>5th</extra>"))
            fig_mini.add_trace(go.Scatter(
                x=_days, y=_p95, mode="lines", name="95th Pctl",
                line=dict(color="#26a69a", width=1, dash="dash"),
                hovertemplate="Day %{x}: %{y:.2f}<extra>95th</extra>"))
            fig_mini.add_trace(go.Scatter(
                x=[1], y=[_ml_r["last_pred"]],
                mode="markers", name="ML Prediction (Day 1)",
                marker=dict(color="#d500f9", size=12, symbol="diamond",
                            line=dict(width=2, color="#7b1fa2")),
                hovertemplate="ML Predicted: %{y:.2f}<extra></extra>"))
            fig_mini.update_layout(
                **PLOTLY_LAYOUT,
                title="<b>30-Day Monte Carlo Fan Chart</b> — 1,000 simulations",
                xaxis_title="Trading Days", yaxis_title="Simulated Price", height=380)
            st.plotly_chart(fig_mini, use_container_width=True)

            # Risk-Reward row
            st.markdown("**Risk-Reward Summary**")
            rr1, rr2, rr3, rr4 = st.columns(4)
            rr1.metric("Best Case (95th)", f"{_mc['best5']:.2f}",
                       f"{((_mc['best5'] - _mc['last_close']) / _mc['last_close'] * 100):+.2f}%")
            rr2.metric("Expected Case", f"{consensus:.2f}",
                       f"{((consensus - _ml_r['last_close']) / _ml_r['last_close'] * 100):+.2f}%")
            rr3.metric("Worst Case (5th)", f"{_mc['worst5']:.2f}",
                       f"{((_mc['worst5'] - _mc['last_close']) / _mc['last_close'] * 100):+.2f}%")
            rr4.metric("Uncertainty (RMSE)", f"±{_ml_r['rmse']:.2f}",
                       help="ML model typical prediction error in price units.")

            st.markdown("---")

        # ── 7. PLAIN ENGLISH VERDICT ──────────────────────────────────────
        # Build verdict
        verdict_parts = []
        verdict_parts.append(f"Over the last <b>{len(df)} trading days</b>, this stock moved from "
                             f"<b>${start_price:.2f}</b> to <b>${end_price:.2f}</b>, "
                             f"a total return of <b>{total_return:+.2f}%</b>.")

        if total_return > 10:
            verdict_parts.append("That's a strong positive performance — the stock has been going up significantly.")
        elif total_return > 0:
            verdict_parts.append("The stock has shown modest gains over this period.")
        elif total_return > -10:
            verdict_parts.append("The stock has declined slightly over this period.")
        else:
            verdict_parts.append("The stock has dropped significantly — it's been in a downtrend.")

        if win_rate_s > 55:
            verdict_parts.append(f"Historically, <b>{win_rate_s:.0f}% of days ended positive</b>, which suggests a bullish tendency.")
        elif win_rate_s < 45:
            verdict_parts.append(f"Only <b>{win_rate_s:.0f}% of days ended positive</b>, showing more losing days than winning ones.")
        else:
            verdict_parts.append(f"The win rate is <b>{win_rate_s:.0f}%</b> — roughly a coin flip, meaning mixed performance.")

        if vol_label == "High":
            verdict_parts.append("⚠️ Recent volatility is <b>higher than average</b> — the stock is swinging more wildly than usual, indicating higher risk.")
        elif vol_label == "Low":
            verdict_parts.append("Recent volatility is <b>lower than average</b> — the stock has been relatively calm.")

        if ml_available and diff_p > 0:
            verdict_parts.append(f"The ML model predicts the next close at <b>${next_p:.2f}</b> (slightly up), "
                                 f"with R² = {r2_ml:.3f} accuracy.")
        elif ml_available:
            verdict_parts.append(f"The ML model predicts the next close at <b>${next_p:.2f}</b> (slightly down), "
                                 f"with R² = {r2_ml:.3f} accuracy.")

        if _mc:
            verdict_parts.append(f"Monte Carlo simulations show a <b>{_mc['prob_profit']:.0f}% chance of profit</b>, "
                                 f"with the expected price at <b>${_mc['expected']:.2f}</b>.")

        verdict_parts.append("<br><br>⚠️ <i>This analysis is based purely on historical data and statistics — "
                             "past performance does not guarantee future results. This is not financial advice.</i>")

        verd_bg = "#e8f5e9" if health_score >= 60 else "#fff8e1" if health_score >= 40 else "#ffebee"
        verd_border = health_color

        st.markdown(
            f'<div class="verdict-card" style="background:{verd_bg}; border-left:5px solid {verd_border};">'
            f'<div class="vc-title">📝 What Does This All Mean? (Plain English Summary)</div>'
            f'<div class="vc-body">{" ".join(verdict_parts)}</div>'
            f'</div>',
            unsafe_allow_html=True)

        st.markdown("---")

        # ── 8. METHODS TABLE (in expander) ────────────────────────────────
        with st.expander("📚 Methods & P&S Concepts Used", expanded=False):
            st.markdown(
                "| Method | Where Used | P&S Concept |\n"
                "|--------|-----------|-------------|\n"
                "| Rolling Mean & Std Dev | 68% Probability Band | Normal Distribution, Empirical Rule |\n"
                "| Linear Regression | Trend Analysis, ML Prediction | Regression, R², RMSE |\n"
                "| RSI (Relative Strength Index) | Momentum Analysis | Moving Averages |\n"
                "| Monte Carlo Simulation | Future Price Scenarios | Random Sampling, CLT |\n"
                "| Isolation Forest | Anomaly Detection | Ensemble Trees, Outlier Detection |\n"
                "| Value-at-Risk (VaR) | Risk Assessment | Quantiles, Probability |\n"
                "| Heap (nlargest/nsmallest) | Top Gain/Loss Days | Data Structures |\n"
                "| Histogram & Box Plot | Return Distribution | Descriptive Statistics |\n"
                "| Correlation Matrix | Feature Relationships | Correlation Coefficients |\n"
                "| Skewness & Kurtosis | Distribution Shape | Higher-Order Moments |\n"
                "| Composite Signal | ML + MC Forecast | Weighted Scoring, Confidence Intervals |\n"
                "| StandardScaler | Feature Normalization | Z-Score Standardization |\n"
                "| Composite Health Score | Summary Dashboard | Multi-factor Index |"
            )

        # ── 9. APPENDIX: PLAIN-ENGLISH GLOSSARY ────────────────────────────
        with st.expander("📖 Appendix: Plain-English Glossary", expanded=False):
            st.markdown(
                '<div class="explain-box" style="background: #f8f9ff; border-left-color: #1a73e8;">'
                '<b>📖 What do all these terms mean?</b> Here\'s a simple guide to every '
                'concept used in this dashboard — no finance degree required!</div>',
                unsafe_allow_html=True)

            st.markdown("### 📊 Basic Statistics")
            st.markdown("""
**Rolling Mean (Moving Average)**
> Imagine tracking your average test score over your last 5 exams instead of all-time.
> That's a rolling mean! We use a 20-day rolling mean to smooth out daily noise and see the real trend.
> *Example: If prices were $10, $12, $11, $13, $14, the 5-day rolling mean is $12.*

**Standard Deviation (Std Dev)**
> Measures how "spread out" the data is. Low std dev = prices stay close to average (stable).
> High std dev = prices swing wildly (volatile). Think of it as a "surprise meter."

**68% Probability Band**
> Based on the bell curve rule: ~68% of data falls within 1 standard deviation of the mean.
> So if a stock's average is $100 with std dev $5, it will likely stay between $95–$105 most days.

**Percentiles (25th, 50th, 75th)**
> If you scored in the 75th percentile on a test, you beat 75% of students.
> Same idea here: the 25th percentile price means 25% of days were lower than this.
""")

            st.markdown("### 📈 Trend & Momentum")
            st.markdown("""
**Linear Regression**
> Draws the "best-fit line" through price data. If the line slopes up, the stock is trending up.
> It's like drawing a straight ruler through a zigzag — shows the overall direction.

**R² Score (R-Squared)**
> How well does that line fit the data? R² = 1.0 means perfect fit (prices follow the line exactly).
> R² = 0.5 means the line explains only half the movement; the rest is random wobble.

**RSI (Relative Strength Index)**
> A 0–100 score measuring if a stock is "overbought" (>70, maybe due for a dip) or
> "oversold" (<30, maybe due for a bounce). It compares recent gains vs losses.
> *Think of it as: "Has this stock been winning too much lately?"*

**Support & Resistance**
> Support = a price floor where the stock keeps bouncing back up (buyers step in).
> Resistance = a price ceiling it struggles to break above (sellers take profit).
""")

            st.markdown("### 🎲 Monte Carlo Simulation")
            st.markdown("""
**What is Monte Carlo?**
> Instead of predicting ONE future, we simulate THOUSANDS of possible futures using random sampling.
> Named after the Monte Carlo casino — it's like rolling dice thousands of times to see all possible outcomes.

**How it works here:**
> 1. We calculate the stock's average daily return and how much it typically swings (std dev)
> 2. We randomly generate thousands of possible daily returns from a bell curve
> 3. We compound them forward: Tomorrow = Today × (1 + random return)
> 4. After 1000+ simulations, we see the range of where prices could land

**Probability of Profit**
> What % of our simulated futures ended higher than today's price?
> If 65% ended higher, there's roughly a 65% chance (based on history) of profit.

**Confidence Band (68% / 90% / 95%)**
> The shaded area where we expect the price to land. A 95% band means:
> "We're 95% confident the future price will fall somewhere in this range."
""")

            st.markdown("### 🔍 Anomaly Detection (Isolation Forest)")
            st.markdown("""
**What is an Anomaly?**
> A day that looks "weird" compared to normal trading — bizarre price swings, unusual volume,
> or patterns that don't match the rest of the data.

**How Isolation Forest works:**
> Imagine a game where you try to separate one sheep from the flock using fences.
> Normal sheep are surrounded by others — takes many fences to isolate them.
> A sheep standing alone in a corner? One fence does the job.
> Anomalies are "easy to isolate" — they stand out from the crowd.

**Anomaly Score**
> A number showing how unusual each day is. Negative scores = anomalies (easy to isolate).
> Positive scores = normal days (buried in the crowd).
""")

            st.markdown("### 🤖 Machine Learning Terms")
            st.markdown("""
**Train/Test Split**
> We teach the model on 70% of data (training), then test it on the remaining 30% (testing).
> This checks if it actually learned patterns or just memorized the answers.

**RMSE (Root Mean Square Error)**
> Measures prediction errors in the same units as the data (dollars).
> RMSE = $2 means predictions are typically off by about $2.

**Feature Importance**
> Which inputs matter most for the prediction? If "RSI" has high importance,
> momentum heavily influences where the model thinks the price will go.

**StandardScaler (Z-Score Normalization)**
> Converts all features to the same scale (mean=0, std=1) so no single feature
> dominates just because it has bigger numbers.
""")

            st.markdown("### 📊 Risk Metrics")
            st.markdown("""
**Win Rate**
> What % of days had positive returns? Win rate of 52% means the stock went up
> slightly more often than it went down.

**Value-at-Risk (VaR) at 95%**
> The worst daily loss you'd expect 95% of the time.
> VaR = -2.5% means: "On 95% of days, you won't lose more than 2.5%."
> (But 5% of the time, losses could be worse!)

**Skewness**
> Is the data lopsided? Negative skew = more extreme losses than gains (long left tail).
> Positive skew = more extreme gains than losses (long right tail).

**Kurtosis**
> Are there more extreme events than a normal bell curve would predict?
> High kurtosis = "fat tails" = more big surprises (good or bad) than expected.
""")

            st.markdown("### 🏆 Health Score Components")
            st.markdown("""
The **Health Score (0–100)** combines 5 factors into one number:

| Component | What it measures | Ideal value |
|-----------|-----------------|-------------|
| **Trend Strength** | Is the stock going up or down overall? | Higher = uptrend |
| **RSI Balance** | Is momentum neutral (not overbought/oversold)? | Close to 50 |
| **Win Rate** | % of positive days | Higher = more consistent |
| **Volatility Stability** | Are recent swings normal or extreme? | Close to historical avg |
| **Trend Fit (R²)** | How clean/predictable is the trend? | Higher = cleaner trend |

**Interpretation:** 60+ = Healthy, 30–60 = Mixed signals, Below 30 = Weak
""")

        st.markdown("---")
        st.caption(
            "📊 This executive report was auto-generated from the uploaded CSV data. "
            "All analyses use historical data only — past performance does not guarantee future results."
        )
    else:
        st.markdown(
            '<div class="info-box">'
            '<b>📂 No data loaded.</b> Upload a stock CSV in the <b>CSV Analysis</b> tab '
            'to generate your executive summary report. The report will automatically pull '
            'together insights from all tabs — trend, volatility, ML prediction, Monte Carlo, '
            'and anomaly detection — into one easy-to-read page.</div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="explain-box">'
            '<b>🔰 How to read this report:</b> Once generated, you\'ll see:<br>'
            '• A <b>Health Score</b> (0–100) combining 5 analysis factors<br>'
            '• <b>Visual cards</b> for trend, momentum, volatility, and distribution<br>'
            '• A <b>Model Comparison</b> showing what each method (ML, Monte Carlo, Stats, Anomaly) says<br>'
            '• A <b>Plain English Summary</b> explaining everything without jargon</div>',
            unsafe_allow_html=True)
