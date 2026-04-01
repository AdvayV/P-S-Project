# Auto-extracted tab module from code.py
# Uses globals provided by main.py

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
        f'<div class="hero-return" style="font-size:1.6rem;">{fmt_price(end_price, currency_symbol)}</div>'
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
    km1.metric("Current Price", fmt_price(end_price, currency_symbol),
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
            f'<div class="ac-row"><span>Support (Low)</span><span class="ac-val">{fmt_price(support_s, currency_symbol)}</span></div>'
            f'<div class="ac-row"><span>Distance to Support</span><span class="ac-val">{dist_sup:.1f}% below</span></div>'
            f'<div class="ac-row"><span>Resistance (High)</span><span class="ac-val">{fmt_price(resist_s, currency_symbol)}</span></div>'
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
            f'<div class="ac-row"><span>68% Range (latest)</span><span class="ac-val">{fmt_price(df["Expected_Low_68"].iloc[-1], currency_symbol)} — {fmt_price(df["Expected_High_68"].iloc[-1], currency_symbol)}</span></div>'
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
            f'<span class="mc-value">{fmt_price(next_p, currency_symbol)}</span></div>'
            f'<div class="mc-row"><span class="mc-icon">📊</span><div><div class="mc-label">Accuracy (R²)</div>'
            f'<div class="mc-desc">How well the model fits test data (1.0 = perfect)</div></div>'
            f'<span class="mc-value">{r2_ml:.4f}</span></div>'
            f'<div class="mc-row"><span class="mc-icon">📏</span><div><div class="mc-label">Error Margin (RMSE)</div>'
            f'<div class="mc-desc">Average error in price units</div></div>'
            f'<span class="mc-value">{fmt_price(rmse_s, currency_symbol)}</span></div>'
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
            n_sims_display = _mc.get("n_simulations", "N/A")
            n_days_display = _mc.get("n_days", "N/A")
            conf_display = _mc.get("confidence", "68%")
            median_display = _mc.get("median", _mc["expected"])
            expected_ret = _mc.get("expected_return", (((_mc["expected"] - _mc["last_close"]) / _mc["last_close"]) * 100))

            st.markdown(
                '<div class="model-compare-card">'
                '<div class="mc-title">🎲 Monte Carlo Simulation</div>'
                f'<div class="mc-subtitle">Ran {n_sims_display:,} simulations over {n_days_display} trading days</div>'
                f'<div class="mc-row"><span class="mc-icon">💰</span><div><div class="mc-label">Probability of Profit</div>'
                f'<div class="mc-desc">% of simulations ending above current price</div></div>'
                f'<span class="mc-value">{_mc["prob_profit"]:.1f}%</span></div>'
                f'<div class="mc-row"><span class="mc-icon">🎯</span><div><div class="mc-label">Expected Price</div>'
                f'<div class="mc-desc">Mean outcome ({expected_ret:+.2f}% return)</div></div>'
                f'<span class="mc-value">{fmt_price(_mc["expected"], currency_symbol)}</span></div>'
                f'<div class="mc-row"><span class="mc-icon">📊</span><div><div class="mc-label">Median Price</div>'
                f'<div class="mc-desc">50th percentile (most likely outcome)</div></div>'
                f'<span class="mc-value">{fmt_price(median_display, currency_symbol)}</span></div>'
                f'<div class="mc-row"><span class="mc-icon">🚀</span><div><div class="mc-label">Best Case (95th)</div>'
                f'<div class="mc-desc">Only 5% of simulations beat this</div></div>'
                f'<span class="mc-value">{fmt_price(_mc["best5"], currency_symbol)}</span></div>'
                f'<div class="mc-row"><span class="mc-icon">⚠️</span><div><div class="mc-label">Worst Case (5th)</div>'
                f'<div class="mc-desc">Only 5% of simulations were worse</div></div>'
                f'<span class="mc-value">{fmt_price(_mc["worst5"], currency_symbol)}</span></div>'
                f'<div class="mc-row"><span class="mc-icon">📈</span><div><div class="mc-label">Signal</div>'
                f'<div class="mc-desc">Based on probability of profit</div></div>'
                f'<span class="mc-value">{mc_signal}</span></div>'
                '</div>',
                unsafe_allow_html=True)

            # Additional MC details in expandable section
            with st.expander("📊 Full Monte Carlo Distribution Details"):
                mc_det1, mc_det2 = st.columns(2)
                with mc_det1:
                    st.markdown("**Percentile Breakdown**")
                    mc_percentile_data = {
                        "Percentile": ["5th (Worst)", "10th", "25th (Q1)", "50th (Median)", "75th (Q3)", "90th", "95th (Best)"],
                        "Price": [
                            fmt_price(_mc.get("worst5", 0), currency_symbol),
                            fmt_price(_mc.get("p10", _mc.get("worst5", 0)), currency_symbol),
                            fmt_price(_mc.get("p25", _mc.get("worst5", 0)), currency_symbol),
                            fmt_price(_mc.get("median", _mc["expected"]), currency_symbol),
                            fmt_price(_mc.get("p75", _mc.get("best5", 0)), currency_symbol),
                            fmt_price(_mc.get("p90", _mc.get("best5", 0)), currency_symbol),
                            fmt_price(_mc.get("best5", 0), currency_symbol),
                        ]
                    }
                    st.dataframe(pd.DataFrame(mc_percentile_data), use_container_width=True, hide_index=True)

                with mc_det2:
                    st.markdown("**Simulation Parameters**")
                    mc_params_data = {
                        "Parameter": ["Simulations", "Forecast Days", "Confidence Band", "Mean Return (μ)", "Volatility (σ)"],
                        "Value": [
                            f"{_mc.get('n_simulations', 'N/A'):,}" if isinstance(_mc.get('n_simulations'), int) else str(_mc.get('n_simulations', 'N/A')),
                            str(_mc.get("n_days", "N/A")),
                            str(_mc.get("confidence", "68%")),
                            f"{_mc.get('mu', 0)*100:.4f}%",
                            f"{_mc.get('sigma', 0)*100:.4f}%",
                        ]
                    }
                    st.dataframe(pd.DataFrame(mc_params_data), use_container_width=True, hide_index=True)

                # Probability metrics
                st.markdown("**Probability Metrics**")
                prob_col1, prob_col2, prob_col3, prob_col4 = st.columns(4)
                prob_col1.metric("Prob. of Profit", f"{_mc['prob_profit']:.1f}%")
                prob_col2.metric("Prob. of Loss", f"{_mc.get('prob_loss', 100-_mc['prob_profit']):.1f}%")
                prob_col3.metric("Start Price", fmt_price(_mc["last_close"], currency_symbol))
                prob_col4.metric("Price Spread (Std)", fmt_price(_mc.get("std", 0), currency_symbol))
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
                   fmt_price(_ml_r['last_pred'], currency_symbol), f"{_ml_r['diff']:+.2f} from current")
        fc2.metric("MC Expected Price",
                   fmt_price(_mc['expected'], currency_symbol),
                   f"{((_mc['expected'] - _mc['last_close']) / _mc['last_close'] * 100):+.2f}%")
        fc3.metric("Consensus Forecast",
                   fmt_price(consensus, currency_symbol),
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
        rr1.metric("Best Case (95th)", fmt_price(_mc['best5'], currency_symbol),
                   f"{((_mc['best5'] - _mc['last_close']) / _mc['last_close'] * 100):+.2f}%")
        rr2.metric("Expected Case", fmt_price(consensus, currency_symbol),
                   f"{((consensus - _ml_r['last_close']) / _ml_r['last_close'] * 100):+.2f}%")
        rr3.metric("Worst Case (5th)", fmt_price(_mc['worst5'], currency_symbol),
                   f"{((_mc['worst5'] - _mc['last_close']) / _mc['last_close'] * 100):+.2f}%")
        rr4.metric("Uncertainty (RMSE)", fmt_price(_ml_r['rmse'], currency_symbol),
                   help="ML model typical prediction error in price units.")

        st.markdown("---")

    # ── 7. PLAIN ENGLISH VERDICT ──────────────────────────────────────
    # Build verdict
    verdict_parts = []
    verdict_parts.append(f"Over the last <b>{len(df)} trading days</b>, this stock moved from "
                         f"<b>{fmt_price(start_price, currency_symbol)}</b> to <b>{fmt_price(end_price, currency_symbol)}</b>, "
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
        verdict_parts.append(f"The ML model predicts the next close at <b>{fmt_price(next_p, currency_symbol)}</b> (slightly up), "
                             f"with R² = {r2_ml:.3f} accuracy.")
    elif ml_available:
        verdict_parts.append(f"The ML model predicts the next close at <b>{fmt_price(next_p, currency_symbol)}</b> (slightly down), "
                             f"with R² = {r2_ml:.3f} accuracy.")

    if _mc:
        verdict_parts.append(f"Monte Carlo simulations show a <b>{_mc['prob_profit']:.0f}% chance of profit</b>, "
                             f"with the expected price at <b>{fmt_price(_mc['expected'], currency_symbol)}</b>.")

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
