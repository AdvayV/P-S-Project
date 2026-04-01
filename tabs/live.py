# Auto-extracted tab module from code.py
# Uses globals provided by main.py

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
            curr_symbol = detect_currency_symbol(ticker)
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
                m1.metric("Last Price",   fmt_price(latest, curr_symbol), f"{change:+.2f}")
                m2.metric("Change %",     f"{pct_chg:+.2f}%")
                m3.metric("Period High",  fmt_price(hist['High'].max(), curr_symbol))
                m4.metric("Period Low",   fmt_price(hist['Low'].min(), curr_symbol))
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

