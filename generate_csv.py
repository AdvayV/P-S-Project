import streamlit as st
import pandas as pd
import io

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

st.set_page_config(page_title="Stock & Crypto CSV Generator", layout="wide")

# ── Custom Styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #f7f9fc; }
    [data-testid="stHeader"]          { background: transparent; }

    .gen-header {
        text-align: center; padding: 12px 0 6px;
    }
    .gen-header .title {
        font-size: 2rem; font-weight: 900;
        background: linear-gradient(135deg, #1a73e8, #6c5ce7);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .gen-header .subtitle {
        color: #666; font-size: 0.92rem;
    }

    .info-box {
        background: rgba(240,244,255,0.85);
        backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
        border-radius: 12px; padding: 14px 18px;
        border: 1px solid rgba(26,115,232,0.18);
        margin-bottom: 14px; font-size: 0.88rem;
        color: #333; line-height: 1.55;
        box-shadow: 0 2px 12px rgba(26,115,232,0.06);
    }

    .explain-box {
        background: linear-gradient(135deg, #fffbe6, #fff9e0);
        border-left: 4px solid #f9a825; border-radius: 10px;
        padding: 14px 18px; margin: 10px 0 16px 0;
        font-size: 0.87rem; color: #5a4e00; line-height: 1.55;
    }
    .explain-box b { color: #e65100; }

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

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1a73e8, #6c5ce7);
        border: none; border-radius: 8px; font-weight: 700;
        letter-spacing: 0.3px; transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(26,115,232,0.3);
    }

    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

    hr { border: none; height: 1px;
         background: linear-gradient(90deg, transparent, #c8d0e8, transparent);
         margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="gen-header">'
    '<div class="title">📥 Stock & Crypto CSV Generator</div>'
    '<div class="subtitle">Fetch historical data for any stock, crypto, or index — then download it for analysis</div>'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown("")

if not YFINANCE_AVAILABLE:
    st.error("yfinance is not installed. Run: `pip install yfinance`")
    st.stop()

st.markdown(
    '<div class="explain-box">'
    '<b>🔰 How to use this tool:</b> '
    '① Pick a stock, crypto, or index from the list (or type a custom ticker). '
    '② Choose a time period (e.g. 1 year). '
    '③ Click <b>Fetch Data</b>. '
    '④ Download the CSV file. '
    '⑤ Upload it in the main dashboard (<code>streamlit run code.py</code>) under the CSV Analysis tab.'
    '</div>',
    unsafe_allow_html=True,
)

# ── Popular tickers list (type to search) ─────────────────────────────────────
POPULAR_TICKERS = {
    # ── Cryptocurrency ────────────────────────────────────────────────
    "BTC-USD - Bitcoin": "BTC-USD",
    "ETH-USD - Ethereum": "ETH-USD",
    "BNB-USD - Binance Coin": "BNB-USD",
    "SOL-USD - Solana": "SOL-USD",
    "XRP-USD - Ripple (XRP)": "XRP-USD",
    "ADA-USD - Cardano": "ADA-USD",
    "DOGE-USD - Dogecoin": "DOGE-USD",
    "DOT-USD - Polkadot": "DOT-USD",
    "AVAX-USD - Avalanche": "AVAX-USD",
    "MATIC-USD - Polygon": "MATIC-USD",
    "LINK-USD - Chainlink": "LINK-USD",
    "UNI-USD - Uniswap": "UNI-USD",
    # ── US Large Cap ──────────────────────────────────────────────────
    "AAPL - Apple": "AAPL",
    "MSFT - Microsoft": "MSFT",
    "GOOGL - Alphabet (Google)": "GOOGL",
    "AMZN - Amazon": "AMZN",
    "NVDA - NVIDIA": "NVDA",
    "META - Meta (Facebook)": "META",
    "TSLA - Tesla": "TSLA",
    "BRK-B - Berkshire Hathaway": "BRK-B",
    "JPM - JPMorgan Chase": "JPM",
    "V - Visa": "V",
    "JNJ - Johnson & Johnson": "JNJ",
    "WMT - Walmart": "WMT",
    "PG - Procter & Gamble": "PG",
    "MA - Mastercard": "MA",
    "UNH - UnitedHealth": "UNH",
    "DIS - Walt Disney": "DIS",
    "HD - Home Depot": "HD",
    "NFLX - Netflix": "NFLX",
    "ADBE - Adobe": "ADBE",
    "CRM - Salesforce": "CRM",
    "PYPL - PayPal": "PYPL",
    "INTC - Intel": "INTC",
    "AMD - AMD": "AMD",
    "CSCO - Cisco": "CSCO",
    "PEP - PepsiCo": "PEP",
    "KO - Coca-Cola": "KO",
    "COST - Costco": "COST",
    "ABBV - AbbVie": "ABBV",
    "MRK - Merck": "MRK",
    "NKE - Nike": "NKE",
    "BA - Boeing": "BA",
    "GS - Goldman Sachs": "GS",
    "UBER - Uber": "UBER",
    "SQ - Block (Square)": "SQ",
    "SNAP - Snap": "SNAP",
    "SPOT - Spotify": "SPOT",
    # ── Indian Stocks (NSE) ───────────────────────────────────────────
    "RELIANCE.NS - Reliance Industries": "RELIANCE.NS",
    "TCS.NS - Tata Consultancy Services": "TCS.NS",
    "INFY.NS - Infosys": "INFY.NS",
    "HDFCBANK.NS - HDFC Bank": "HDFCBANK.NS",
    "ICICIBANK.NS - ICICI Bank": "ICICIBANK.NS",
    "HINDUNILVR.NS - Hindustan Unilever": "HINDUNILVR.NS",
    "SBIN.NS - State Bank of India": "SBIN.NS",
    "BHARTIARTL.NS - Bharti Airtel": "BHARTIARTL.NS",
    "ITC.NS - ITC Limited": "ITC.NS",
    "KOTAKBANK.NS - Kotak Mahindra Bank": "KOTAKBANK.NS",
    "LT.NS - Larsen & Toubro": "LT.NS",
    "AXISBANK.NS - Axis Bank": "AXISBANK.NS",
    "WIPRO.NS - Wipro": "WIPRO.NS",
    "HCLTECH.NS - HCL Technologies": "HCLTECH.NS",
    "TATAMOTORS.NS - Tata Motors": "TATAMOTORS.NS",
    "SUNPHARMA.NS - Sun Pharma": "SUNPHARMA.NS",
    "MARUTI.NS - Maruti Suzuki": "MARUTI.NS",
    "TITAN.NS - Titan Company": "TITAN.NS",
    "BAJFINANCE.NS - Bajaj Finance": "BAJFINANCE.NS",
    "ADANIENT.NS - Adani Enterprises": "ADANIENT.NS",
    "TATASTEEL.NS - Tata Steel": "TATASTEEL.NS",
    "POWERGRID.NS - Power Grid Corp": "POWERGRID.NS",
    "NTPC.NS - NTPC": "NTPC.NS",
    "ONGC.NS - ONGC": "ONGC.NS",
    "COALINDIA.NS - Coal India": "COALINDIA.NS",
    # ── Indices ───────────────────────────────────────────────────────
    "^NSEI - Nifty 50": "^NSEI",
    "^BSESN - Sensex": "^BSESN",
    "^GSPC - S&P 500": "^GSPC",
    "^DJI - Dow Jones": "^DJI",
}

st.subheader("1️⃣ Select a Stock, Crypto, or Index")

mode = st.radio("Pick method", ["Search from list", "Type custom ticker"], horizontal=True)

c1, c2 = st.columns(2)
with c1:
    if mode == "Search from list":
        selected_label = st.selectbox(
            "Start typing to search",
            options=list(POPULAR_TICKERS.keys()),
            index=0,
            help="Type a company name, crypto, or ticker symbol to filter the list."
        )
        ticker = POPULAR_TICKERS[selected_label]
        st.caption(f"Ticker: **{ticker}**")
    else:
        ticker = st.text_input("Stock / Crypto Ticker Symbol", value="BTC-USD",
                               placeholder="e.g. AAPL, BTC-USD, RELIANCE.NS")
with c2:
    period = st.selectbox("Time Period", [
        "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"
    ], index=3)

interval = st.selectbox("Data Interval", ["1d", "1wk", "1mo"], index=0,
                         help="1d = daily prices, 1wk = weekly, 1mo = monthly")

st.markdown("---")

fetch = st.button("🔍 Fetch Data", type="primary")

if fetch:
    with st.spinner(f"Fetching {ticker.upper()} data..."):
        try:
            tkr = yf.Ticker(ticker.strip())
            hist = tkr.history(period=period, interval=interval)

            if hist.empty:
                st.error("No data returned. Check the ticker symbol and try again.")
                st.stop()

            hist = hist.reset_index()

            # Keep only the standard columns the dashboard expects
            date_col = hist.columns[0]
            hist = hist.rename(columns={date_col: "Date"})
            keep_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            keep_cols = [c for c in keep_cols if c in hist.columns]
            hist = hist[keep_cols]

            # Format date cleanly
            hist["Date"] = pd.to_datetime(hist["Date"]).dt.strftime("%Y-%m-%d")

            # Round price columns
            for col in ["Open", "High", "Low", "Close"]:
                if col in hist.columns:
                    hist[col] = hist[col].round(2)

            st.success(f"✅ Fetched **{len(hist)}** rows for **{ticker.upper()}**")

            # Preview
            st.subheader("2️⃣ Preview")
            st.dataframe(hist, use_container_width=True, height=350)

            # Quick stats
            st.subheader("3️⃣ Quick Stats")
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Rows", len(hist))
            s2.metric("Date Range", f"{hist['Date'].iloc[0]} → {hist['Date'].iloc[-1]}")
            if "Close" in hist.columns:
                s3.metric("Start Price", f"${hist['Close'].iloc[0]:,.2f}")
                s4.metric("End Price", f"${hist['Close'].iloc[-1]:,.2f}")
                change = ((float(hist['Close'].iloc[-1]) - float(hist['Close'].iloc[0])) /
                          float(hist['Close'].iloc[0])) * 100
                s5.metric("Total Change", f"{change:+.2f}%")

            # Download button
            st.markdown("---")
            st.subheader("4️⃣ Download")
            csv_buffer = io.StringIO()
            hist.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            filename = f"{ticker.strip().upper()}_{period}_{interval}.csv"

            st.download_button(
                label=f"⬇️ Download {filename}",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                type="primary",
            )

            st.markdown(
                '<div class="info-box">'
                f'<b>Next step:</b> Upload <code>{filename}</code> in the main dashboard '
                '(<code>streamlit run code.py</code>) under the <b>CSV Analysis</b> tab to '
                'see charts, ML predictions, Monte Carlo simulations, and more.'
                '</div>',
                unsafe_allow_html=True,
            )

        except Exception as ex:
            st.error(f"Error fetching data: {ex}")
