# Auto-extracted tab module from code.py
# Uses globals provided by main.py

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
    pm1.metric("RMSE (lower = better)",  fmt_price(rmse, currency_symbol))
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
    n1.metric("Today's Close",          fmt_price(last_close, currency_symbol))
    n2.metric("Predicted Next Close",   fmt_price(last_pred, currency_symbol), f"{diff:+.2f}")
    n3.metric("Signal", "Likely UP" if diff > 0 else "Likely DOWN")
else:
    st.info("Upload a CSV in the CSV Analysis tab first.")

