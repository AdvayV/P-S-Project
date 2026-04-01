# Auto-extracted tab module from code.py
# Uses globals provided by main.py

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
