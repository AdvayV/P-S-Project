# This tab uses shared dashboard style and upload controls from main.py.

st.markdown('<div class="section-header">Machine Learning: Next-Session Close</div>', unsafe_allow_html=True)
active_stock = render_stock_selector("ml")
st.write(
    "Linear regression estimates the next observed closing price after each complete daily row. "
    "The persistence baseline uses that session's close as its forecast."
)

ml_file = uploaded_file_1 if active_stock == "Stock 1" else uploaded_file_2
ml_result = None
if ml_file:
    test_pct = st.slider("Holdout size (%)", 10, 40, 20, key="ml_holdout_pct")
    ml_frame, _ = load_and_clean_csv(ml_file)
    try:
        ml_result = evaluate_next_session(add_features(ml_frame), test_pct / 100)
    except (KeyError, ValueError) as error:
        st.info(f"Forecast unavailable. {error}")
else:
    st.info("Upload a CSV for the selected stock in the CSV Analysis tab.")

if ml_result:
    ml_currency = detect_currency_symbol_from_file(ml_file)
    holdout = ml_result["holdout"]
    pm1, pm2, pm3, pm4 = st.columns(4)
    pm1.metric("Regression RMSE", fmt_price(ml_result["rmse"], ml_currency))
    pm2.metric("Persistence RMSE", fmt_price(ml_result["persistence_rmse"], ml_currency))
    pm3.metric("Holdout R²", f'{ml_result["r2"]:.4f}')
    pm4.metric("Training rows", ml_result["train_count"])
    st.caption(
        "Both RMSE values use the same later sessions. RMSE is in price units. "
        "The two models are compared at each prior session's close."
    )

    fig_ml = go.Figure()
    for column, label, color in (
        ("actual", "Actual", "#1a73e8"),
        ("regression", "Regression", "#ff7043"),
        ("persistence", "Persistence", "#697386"),
    ):
        fig_ml.add_trace(go.Scatter(
            x=holdout["target_date"],
            y=holdout[column],
            name=label,
            mode="lines",
            line=dict(color=color),
        ))
    fig_ml.update_layout(
        **PLOTLY_LAYOUT,
        title="Holdout closes by target session",
        yaxis_title="Price",
        height=420,
    )
    st.plotly_chart(fig_ml, use_container_width=True)

    st.subheader("Next-session forecast")
    st.caption(
        f'As of {ml_result["as_of"]:%Y-%m-%d}, the latest session in the uploaded CSV. '
        "The forecast is refitted on all known target pairs."
    )
    latest, estimate = st.columns(2)
    latest.metric("Latest observed close", fmt_price(ml_result["last_close"], ml_currency))
    estimate.metric("Estimated next close", fmt_price(ml_result["prediction"], ml_currency))

    st.subheader("Fitted coefficients")
    st.caption("Inputs use different units, so coefficient magnitudes are not comparable as feature importance.")
    coefficients = ml_result["coefficients"].sort_values(key=abs)
    fig_coef = go.Figure(go.Bar(
        y=coefficients.index,
        x=coefficients.values,
        orientation="h",
        marker_color="#3487ce",
    ))
    fig_coef.update_layout(**PLOTLY_LAYOUT, xaxis_title="Coefficient", height=320)
    st.plotly_chart(fig_coef, use_container_width=True)
