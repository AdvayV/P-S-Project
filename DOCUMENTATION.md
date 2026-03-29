# Stock Analysis Dashboard — Project Documentation

## Overview

A **Streamlit-based stock analysis dashboard** that combines probability & statistics concepts with machine learning to analyze stock market data. The dashboard provides interactive visualizations, predictive modeling, anomaly detection, and Monte Carlo simulations.

### Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3 | Core language |
| Streamlit | Web application framework |
| Pandas / NumPy | Data manipulation & numerical computing |
| Plotly | Interactive charts (zoom, pan, hover, range selectors) |
| Scikit-Learn | ML models (Linear Regression, Isolation Forest, StandardScaler) |
| SciPy | Statistical tests & distributions |
| yfinance | Real-time & historical stock data from Yahoo Finance |
| heapq | Efficient top-N gain/loss computation |

### Files

| File | Description |
|------|-------------|
| `code.py` | Main dashboard application (7 interactive tabs) |
| `generate_csv.py` | Modernized tool to fetch & download Stock/Crypto CSV data |

### Design & User Experience
- **Modern UI:** Vibrant gradient headers, glassmorphism-style info boxes, and interactive metric cards with hover effects.
- **Beginner-Friendly:** Each tab includes "explain-boxes" (marked with 🔰) that clarify complex finance and statistics concepts in plain English.
- **Visual Feedback:** Emoji-labeled tabs and color-coded signals (🟢 Bullish, 🔴 Bearish, 🟡 Neutral) for intuitive results.
- **Dual-Stock Workflow:** Upload 2 CSVs and switch between **Stock 1 / Stock 2** from selector buttons available on every analysis tab.

---

## How to Run

```bash
# Install dependencies (one-time)
pip install streamlit pandas numpy plotly scikit-learn scipy yfinance

# Run the main dashboard
streamlit run code.py

# Run the CSV generator (separate page)
streamlit run generate_csv.py
```

---

## Feature Documentation

### Tab 1: Live Chart

**Purpose:** View real-time stock price data fetched from Yahoo Finance.

**Features:**
- Enter any stock ticker symbol (e.g., AAPL, RELIANCE.NS)
- Select time period (1 month to 5 years)
- Choose chart style: **Line Chart** or **Candlestick**
- Auto-refresh toggle for live updates
- **3-Panel Interactive Chart:**
  - **Panel 1 — Price:** Close price (or candlestick) with SMA-20 overlay and 68% Probable Range shaded band
  - **Panel 2 — Volume:** Color-coded volume bars (green = up day, red = down day)
  - **Panel 3 — RSI(14):** Relative Strength Index identifying overbought (>70) and oversold (<30) conditions.
- **Metrics Bar:** Last Price, Change %, Period High, Period Low, Total Volume.
- **Explain-Boxes (🔰):** Beginner guides for understanding Tickers, Candlesticks, RSI, and Volume.
- Range selector buttons (1W, 1M, 3M, 6M, 1Y, All).

**P&S Concepts:** Normal Distribution (68% rule), Moving Averages, Empirical Rule

---

### Tab 2: CSV Analysis

**Purpose:** Upload one or two stock CSV files and run in-depth analysis on the selected stock.

**Features:**
- Dual CSV upload:
  - **Stock 1 CSV** (primary)
  - **Stock 2 CSV** (optional, for side-by-side comparison in Summary)
- Stock selector buttons (**Stock 1 / Stock 2**) to choose which dataset feeds all analysis tabs
- CSV auto-detection of date and numeric columns
- Data cleaning and feature engineering:
  - `Daily_Return` = percentage change of Close price
  - `Volatility_10` = 10-day rolling standard deviation of returns
  - `RSI_14` = Relative Strength Index (14-period)
  - `Expected_High_68` / `Expected_Low_68` = 68% probability bounds (mean ± 1σ)
- **Interactive 3-Panel Chart** (same layout as Live Chart)
- Toggle between Line and Candlestick chart styles
- **Chart Analysis (Auto-Generated):**
  - 📈 **Trend Analysis** — Linear regression slope, R² score, trend classification (strong/mild up/downtrend, sideways)
  - 📊 **Volatility Assessment** — Average vs recent volatility, biggest gain/drop days
  - ⚡ **Momentum (RSI)** — Current RSI reading with plain-English interpretation (overbought / oversold / bullish / bearish / neutral)
  - 🔒 **Key Levels** — Support (60-day low) and Resistance (60-day high) with distance from current price
  - 🎯 **Win Rate & Risk** — Win rate percentage, average return, and 95% Value-at-Risk.
  - 📝 **Summary Verdict** — Consolidated signal detection (e.g., "uptrend, high volatility, positive historical bias").
- **Explain-Boxes (🔰):** Plain-English definitions of Win Rate, Max Drop, and statistical ranges.
- **Correlation Matrix** (expandable) with color-coded heatmap
- **Quick Statistics:** Top 5 gains/losses using Python's heapq module
- Descriptive statistics table (mean, std, min, max, quartiles)

**P&S Concepts:** Linear Regression, R², Standard Deviation, Correlation, Percentiles, Outliers, Z-scores, Value-at-Risk, Skewness, Kurtosis

---

### Tab 3: ML Prediction

**Purpose:** Predict the next day's closing price using Linear Regression.

**Features:**
- Uses the currently selected stock from the global Stock 1 / Stock 2 selector
- Adjustable train/test split ratio (10%–40% test set)
- **Model Training:** Linear Regression on 8 features:
  - Open, High, Low, Close, Volume
  - Expected_High_68, Expected_Low_68
  - RSI_14
- **Performance Metrics:** RMSE, R² Score
- **Actual vs Predicted Chart** — Interactive line chart comparing real vs predicted test prices
- **Tomorrow's Predicted Close** — Displays predicted price with UP/DOWN signal and delta from today.
- **Feature Importance Chart** — Visualizes which factors (price, volume, RSI) most influence the prediction.
- **Explain-Boxes (🔰):** Explains Linear Regression using a school-score analogy.

**P&S Concepts:** Regression Analysis, Train-Test Split, RMSE, R², Feature Importance, Prediction Intervals

---

### Tab 4: Statistics

**Purpose:** Deep statistical analysis of the stock's return distribution.

**Features:**
- Uses the currently selected stock from the global Stock 1 / Stock 2 selector
- **Descriptive Statistics Table** — Mean, Std Dev, Min, Max, Quartiles, Skewness, Kurtosis
- **Histogram of Daily Returns** — 60-bin histogram with Mean and Median vertical lines
- **Box Plot** — Shows quartiles, median, and outlier detection
- **Cumulative Return Over Time** — Area chart showing total return with color coding (green = positive, red = negative)
- **Rolling Volatility (10-Day)** — Explores price swing intensity over the last 10 trading days.
- **Explain-Boxes (🔰):** Simple interpretations of Histograms, Box Plots, Volatility, Skewness, and Kurtosis.

**P&S Concepts:** Descriptive Statistics, Normal Distribution, Skewness, Kurtosis, Quartiles, Cumulative Distribution, Rolling Statistics, Outlier Identification

---

### Tab 5: Monte Carlo Simulation

**Purpose:** Simulate thousands of possible future price scenarios using random sampling.

**Features:**
- Uses the currently selected stock from the global Stock 1 / Stock 2 selector
- **Configurable Parameters:**
  - Number of simulations (100 – 10,000)
  - Forecast horizon (5 – 252 trading days)
  - Confidence band (68% / 90% / 95%)
- **Fan Chart** — Overlays up to 100 individual simulated paths with:
  - Orange **median path** (most likely outcome)
  - Shaded **confidence band** (probability range)
  - Dashed upper/lower **percentile boundaries**
- **Final Price Distribution** — Histogram of all simulated end prices with start price and median markers
- **Key Takeaways Metrics:**
  - Probability of Profit (% of simulations ending above current price)
  - Expected Price (average across all simulations)
  - Best Case (95th percentile)
  - Worst Case (5th percentile)
- **Plain English Summary** — Explains results in simple words

**How it Works:**
1. Calculates historical mean daily return (μ) and standard deviation (σ)
2. Generates random daily returns from Normal Distribution N(μ, σ)
3. Compounds them forward: Price(t) = Price(t-1) × (1 + random_return)
4. Repeats for N simulations to build probability distribution

**P&S Concepts:** Monte Carlo Method, Random Sampling, Normal Distribution, Central Limit Theorem, Confidence Intervals, Percentiles

---

### Tab 6: Anomaly Detection

**Purpose:** Identify statistically unusual trading days using machine learning.

**Features:**
- Uses the currently selected stock from the global Stock 1 / Stock 2 selector
- **Algorithm:** Isolation Forest (scikit-learn ensemble method)
- **Configurable Parameters:**
  - Expected Anomaly Rate (1% – 10%)
  - Number of Trees (50 – 300)
- **Price Chart with Anomalies** — Close price line with red ✕ markers on anomalous days
- **Anomaly Score Chart** — Score timeline with threshold boundary line; negative scores = anomalies
- **Anomalous Days Detail Table** — Date, Close price, Daily Return, and Anomaly Score for each flagged day
- **Plain English Explanation** — Describes what the algorithm found

**How it Works:**
1. Prepares 8 trading features (OHLCV, returns, volatility, RSI)
2. Standardizes features using StandardScaler
3. Trains an Isolation Forest — isolates data points using random splits
4. Days that are easy to isolate (require fewer splits) are anomalies
5. Assigns anomaly scores; days scoring below threshold are flagged

**P&S Concepts:** Outlier Detection, Probability Thresholds, Ensemble Methods (Isolation Forest), Feature Scaling (Z-Score)

---

### Tab 7: Summary

**Purpose:** Consolidated report of all analyses in one place.

**Sections:**
1. **Data Overview** — Total days, date range, start/end price, total return
2. **Statistical Profile** — Mean return, std dev, skewness, kurtosis, win rate, VaR
3. **Trend Analysis** — Linear regression slope, R², trend direction
4. **Technical Indicators** — RSI reading, volatility (current vs average), support/resistance levels
5. **ML Prediction Results** — Model metrics, predicted next close, UP/DOWN signal.
6. **Key Probability Metrics** — 68% range, win rate, VaR, skewness interpretation.
7. **ML + Monte Carlo Forecast (New!)** — Integrates prediction and simulation for a balanced view:
   - **Combined Forecast Card:** Side-by-side view of ML prediction, MC expected price, and a "Consensus" average.
   - **Confidence-Weighted Signal:** A composite score combining ML model R² and MC probability of profit. Displays a colored card: **Bullish (🟢), Neutral (🟡), or Bearish (🔴).**
   - **30-Day Mini Fan Chart:** A compact MC simulation over the next 30 days with the ML prediction plotted on Day 1 to show where the regression sits within the probability fan.
   - **Risk-Reward Summary:** Best-case (95th), Worst-case (5th), and the ML uncertainty margin (RMSE).
8. **Methods Used Table** — Detailed mapping of techniques to P&S concepts.
9. **Dual-Stock Investor Pick (New!)** — When both CSVs are uploaded:
   - Computes Health/Behavior score for **Stock 1** and **Stock 2**
   - Shows side-by-side Total Return and Win Rate
   - Highlights a **Better Pick** based on higher Health score
10. **Appendix: Plain-English Glossary (New!)** — Beginner-friendly explanations of:
   - Rolling Mean, Std Dev, 68% band, Percentiles
   - Linear Regression, R², RSI, Support/Resistance
   - Monte Carlo terms (probability of profit, confidence bands)
   - Isolation Forest anomaly logic and anomaly score
   - Risk terms (VaR, skewness, kurtosis) and Health score components

---

## Standalone Tool: CSV Generator (`generate_csv.py`)

**Purpose:** Easily fetch historical data for Any Stock, Crypto, or Index.

**Features:**
- **Crypto-First Focus:** Highlighted Bitcoin (BTC) and 12+ other top cryptocurrencies (ETH, SOL, BNB, etc.).
- **Searchable List:** 100+ popular tickers across US Markets (Nasdaq/NYSE), Indian Markets (NSE), and Global Indices.
- **Modern Interface:** High-contrast design with step-by-step instructions for beginners.
- **Customization:** Select Time Period (1mo to Max) and Interval (Daily, Weekly, Monthly).
- **Data Preview:** Quick stats including start/end prices and total % change.
- **Ready for Dashboard:** One-click download as a pre-formatted CSV with OHLCV data.

---

## P&S Concepts Covered

| Method | Where Used | P&S Concept |
|--------|-----------|-------------|
| Rolling Mean & Std Dev | 68% Probability Band | Normal Distribution, Empirical Rule |
| Linear Regression | Trend Analysis, ML Prediction | Regression, R², RMSE |
| RSI (Relative Strength Index) | Momentum Analysis | Moving Averages |
| Monte Carlo Simulation | Future Price Scenarios | Random Sampling, CLT |
| Isolation Forest | Anomaly Detection | Ensemble Trees, Outlier Detection |
| Value-at-Risk (VaR) | Risk Assessment | Quantiles, Probability |
| Heap (nlargest/nsmallest) | Top Gain/Loss Days | Data Structures |
| Histogram & Box Plot | Return Distribution | Descriptive Statistics |
| Correlation Matrix | Feature Relationships | Correlation Coefficients |
| Skewness & Kurtosis | Distribution Shape | Higher-Order Moments |
| Composite Signal | ML + MC Forecast | Weighted Scoring, Composite Index |
| StandardScaler | Feature Normalization | Z-Score Standardization |
| Cumulative Return | Performance Tracking | Compound Growth |

---

## Dependencies

```
streamlit
pandas
numpy
plotly
scikit-learn
scipy
yfinance
```

Install all at once:
```bash
pip install streamlit pandas numpy plotly scikit-learn scipy yfinance
```
