# Auto-extracted tab module from code.py
# Uses globals provided by main.py

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

