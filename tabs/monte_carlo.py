# Auto-extracted tab module from code.py
# Uses globals provided by main.py

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
                           annotation_text=f"Start: {fmt_price(last_close, currency_symbol)}",
                           annotation_position="top right")
        fig_dist.add_vline(x=np.median(final_prices), line_color="#ab47bc",
                           line_dash="dash", line_width=1.5,
                           annotation_text=f"Median: {fmt_price(np.median(final_prices), currency_symbol)}",
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

        # Calculate additional statistics for detailed display
        median_final = np.median(final_prices)
        std_final = np.std(final_prices)
        min_final = np.min(final_prices)
        max_final = np.max(final_prices)
        p10 = np.percentile(final_prices, 10)
        p25 = np.percentile(final_prices, 25)
        p75 = np.percentile(final_prices, 75)
        p90 = np.percentile(final_prices, 90)
        expected_return = ((expected - last_close) / last_close) * 100
        median_return = ((median_final - last_close) / last_close) * 100
        conf_low_final = lo_band[-1]
        conf_high_final = hi_band[-1]

        # Store Monte Carlo results for Summary tab (expanded)
        st.session_state[f"mc_results_{active_stock}"] = {
            "prob_profit": prob_profit,
            "prob_loss": prob_loss,
            "expected": expected,
            "expected_return": expected_return,
            "median": median_final,
            "median_return": median_return,
            "worst5": worst5,
            "best5": best5,
            "p10": p10,
            "p25": p25,
            "p75": p75,
            "p90": p90,
            "min": min_final,
            "max": max_final,
            "std": std_final,
            "conf_low": conf_low_final,
            "conf_high": conf_high_final,
            "confidence": confidence,
            "last_close": last_close,
            "mu": mu,
            "sigma": sigma,
            "n_simulations": n_simulations,
            "n_days": n_days,
        }

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Prob. of Profit",  f"{prob_profit:.1f}%",
                  help="Percentage of simulations that ended above today's price.")
        k2.metric("Expected Price",   fmt_price(expected, currency_symbol),
                  f"{((expected - last_close)/last_close)*100:+.2f}%",
                  help="Average price across all simulations.")
        k3.metric("Best Case (95th)", fmt_price(best5, currency_symbol),
                  f"{((best5 - last_close)/last_close)*100:+.2f}%",
                  help="Only 5% of simulations ended above this price.")
        k4.metric("Worst Case (5th)", fmt_price(worst5, currency_symbol),
                  f"{((worst5 - last_close)/last_close)*100:+.2f}%",
                  help="Only 5% of simulations ended below this price.")

        st.markdown("---")
        st.markdown(
            f"**In plain English:** Based on {n_simulations:,} simulated scenarios over {n_days} trading days, "
            f"there is a **{prob_profit:.0f}% chance** the stock goes up and a **{prob_loss:.0f}% chance** it goes down. "
            f"The most likely outcome (median) is a price of **{median_final:.2f}**. "
            f"With {confidence} confidence, the price should land between "
            f"**{conf_low_final:.2f}** and **{conf_high_final:.2f}**.")

        # ── Detailed Monte Carlo Values Table ──
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Detailed Simulation Results</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="explain-box">'
            '<b>🔰 Understanding these values:</b> The table below shows the complete distribution of '
            f'simulated final prices after {n_days} trading days. Percentiles tell you what percentage of '
            'simulations ended below that price. For example, the 25th percentile means 25% of simulations '
            'ended below that price.'
            '</div>', unsafe_allow_html=True)

        # Create two columns for the detailed values
        det_col1, det_col2 = st.columns(2)

        with det_col1:
            st.markdown("**📈 Price Distribution Statistics**")
            price_stats_data = {
                "Metric": [
                    "Starting Price",
                    "Expected (Mean) Price",
                    "Median Price",
                    "Standard Deviation",
                    "Minimum Simulated",
                    "Maximum Simulated",
                    "Price Range (Max - Min)",
                ],
                "Value": [
                    fmt_price(last_close, currency_symbol),
                    fmt_price(expected, currency_symbol),
                    fmt_price(median_final, currency_symbol),
                    fmt_price(std_final, currency_symbol),
                    fmt_price(min_final, currency_symbol),
                    fmt_price(max_final, currency_symbol),
                    fmt_price(max_final - min_final, currency_symbol),
                ],
                "Change %": [
                    "—",
                    f"{expected_return:+.2f}%",
                    f"{median_return:+.2f}%",
                    "—",
                    f"{((min_final - last_close)/last_close)*100:+.2f}%",
                    f"{((max_final - last_close)/last_close)*100:+.2f}%",
                    "—",
                ]
            }
            st.dataframe(pd.DataFrame(price_stats_data), use_container_width=True, hide_index=True)

        with det_col2:
            st.markdown("**📊 Percentile Breakdown**")
            percentile_data = {
                "Percentile": [
                    "5th (Worst Case)",
                    "10th",
                    "25th (Q1)",
                    "50th (Median)",
                    "75th (Q3)",
                    "90th",
                    "95th (Best Case)",
                ],
                "Price": [
                    fmt_price(worst5, currency_symbol),
                    fmt_price(p10, currency_symbol),
                    fmt_price(p25, currency_symbol),
                    fmt_price(median_final, currency_symbol),
                    fmt_price(p75, currency_symbol),
                    fmt_price(p90, currency_symbol),
                    fmt_price(best5, currency_symbol),
                ],
                "Return %": [
                    f"{((worst5 - last_close)/last_close)*100:+.2f}%",
                    f"{((p10 - last_close)/last_close)*100:+.2f}%",
                    f"{((p25 - last_close)/last_close)*100:+.2f}%",
                    f"{median_return:+.2f}%",
                    f"{((p75 - last_close)/last_close)*100:+.2f}%",
                    f"{((p90 - last_close)/last_close)*100:+.2f}%",
                    f"{((best5 - last_close)/last_close)*100:+.2f}%",
                ]
            }
            st.dataframe(pd.DataFrame(percentile_data), use_container_width=True, hide_index=True)

        # Probability Analysis
        st.markdown("**🎯 Probability Analysis**")
        prob_col1, prob_col2, prob_col3 = st.columns(3)

        with prob_col1:
            st.markdown(
                f'<div style="background:#e8f5e9; border-radius:12px; padding:16px; text-align:center;">'
                f'<div style="font-size:0.85rem; color:#2e7d32; font-weight:600;">CHANCE OF PROFIT</div>'
                f'<div style="font-size:2rem; font-weight:800; color:#1b5e20;">{prob_profit:.1f}%</div>'
                f'<div style="font-size:0.8rem; color:#388e3c;">{int(prob_profit/100 * n_simulations):,} of {n_simulations:,} sims</div>'
                f'</div>', unsafe_allow_html=True)

        with prob_col2:
            st.markdown(
                f'<div style="background:#ffebee; border-radius:12px; padding:16px; text-align:center;">'
                f'<div style="font-size:0.85rem; color:#c62828; font-weight:600;">CHANCE OF LOSS</div>'
                f'<div style="font-size:2rem; font-weight:800; color:#b71c1c;">{prob_loss:.1f}%</div>'
                f'<div style="font-size:0.8rem; color:#d32f2f;">{int(prob_loss/100 * n_simulations):,} of {n_simulations:,} sims</div>'
                f'</div>', unsafe_allow_html=True)

        with prob_col3:
            # Calculate probability of >10% gain and >10% loss
            prob_10_gain = (final_prices > last_close * 1.10).mean() * 100
            prob_10_loss = (final_prices < last_close * 0.90).mean() * 100
            st.markdown(
                f'<div style="background:#fff3e0; border-radius:12px; padding:16px; text-align:center;">'
                f'<div style="font-size:0.85rem; color:#e65100; font-weight:600;">EXTREME MOVES</div>'
                f'<div style="font-size:1rem; font-weight:700; color:#ef6c00;">📈 >10% gain: {prob_10_gain:.1f}%</div>'
                f'<div style="font-size:1rem; font-weight:700; color:#ef6c00;">📉 >10% loss: {prob_10_loss:.1f}%</div>'
                f'</div>', unsafe_allow_html=True)

        # Simulation Parameters Used
        st.markdown("---")
        st.markdown("**⚙️ Simulation Parameters**")
        param_col1, param_col2, param_col3, param_col4 = st.columns(4)
        param_col1.metric("Simulations Run", f"{n_simulations:,}")
        param_col2.metric("Forecast Horizon", f"{n_days} days")
        param_col3.metric("Mean Daily Return (μ)", f"{mu*100:.4f}%")
        param_col4.metric("Daily Volatility (σ)", f"{sigma*100:.4f}%")

else:
    st.info("Upload a CSV in the CSV Analysis tab first.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
