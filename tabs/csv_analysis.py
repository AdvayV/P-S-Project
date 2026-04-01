# Auto-extracted tab module from code.py
# Uses globals provided by main.py

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
currency_symbol = detect_currency_symbol_from_file(uploaded_file)

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
    q4.metric("Avg Close", fmt_price(df['Close'].mean(), currency_symbol))
    q5.metric("Avg Daily Return", f"{df['Daily_Return'].mean()*100:.3f}%")

    st.subheader("Top Price Change Days")
    col_g, col_l = st.columns(2)
    if direction in ("Max Gains", "Both"):
        gains = heapq.nlargest(N, enumerate(df_ml["Price_Change"]), key=lambda x: x[1])
        gain_data = [[df_ml.iloc[i]["Date"].strftime("%Y-%m-%d"),
                      fmt_price(df_ml.iloc[i]["Close"], currency_symbol),
                      fmt_price(df_ml.iloc[i]["Close_next"], currency_symbol),
                      f"+{c:.2f}"] for i, c in gains]
        with col_g:
            st.markdown("**Biggest Gains**")
            st.dataframe(pd.DataFrame(gain_data, columns=["Date","Close","Next Close","Change"]),
                         use_container_width=True, hide_index=True)

    if direction in ("Max Losses", "Both"):
        losses = heapq.nsmallest(N, enumerate(df_ml["Price_Change"]), key=lambda x: x[1])
        loss_data = [[df_ml.iloc[i]["Date"].strftime("%Y-%m-%d"),
                      fmt_price(df_ml.iloc[i]["Close"], currency_symbol),
                      fmt_price(df_ml.iloc[i]["Close_next"], currency_symbol),
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
                e1.metric("Purchase Price",       fmt_price(buy_p, currency_symbol),       f"on {buy_d.date()}")
                e2.metric("Best Exit Price",      fmt_price(best['Close'], currency_symbol),f"on {best['Date'].date()}")
                e3.metric("Max Potential Profit", fmt_price(profit, currency_symbol),      f"{pct_p:.2f}%")
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

