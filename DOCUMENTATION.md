# Stock Analysis Dashboard — Project Documentation

## Overview

A **Streamlit-based stock analysis dashboard** that combines probability & statistics concepts with machine learning to analyze stock market data. The dashboard provides interactive visualizations, predictive modeling, anomaly detection, and Monte Carlo simulations.

This project serves as both a **practical financial analysis tool** and an **educational platform** demonstrating how statistical and machine learning techniques apply to real-world stock market data. It bridges the gap between theoretical probability & statistics concepts taught in academia and their practical applications in quantitative finance.

### Project Goals

1. **Educational Value**: Demonstrate real-world applications of probability & statistics concepts (normal distributions, regression analysis, Monte Carlo methods, etc.)
2. **Practical Utility**: Provide actionable insights for stock analysis through data-driven metrics
3. **Accessibility**: Make complex financial analysis accessible to beginners through intuitive UI and plain-English explanations
4. **Interactivity**: Enable hands-on exploration of statistical concepts through interactive visualizations

### Tech Stack

| Technology       | Purpose                   | Why We Chose It                                                                         |
| ---------------- | ------------------------- | --------------------------------------------------------------------------------------- |
| **Python 3**     | Core language             | Industry standard for data science and financial analysis                               |
| **Streamlit**    | Web application framework | Rapid prototyping of data apps with minimal frontend code                               |
| **Pandas**       | Data manipulation         | Powerful DataFrame operations for time-series financial data                            |
| **NumPy**        | Numerical computing       | Efficient array operations for statistical calculations                                 |
| **Plotly**       | Interactive charts        | Rich interactivity (zoom, pan, hover tooltips, range selectors)                         |
| **Scikit-Learn** | Machine learning          | Production-ready implementations of Linear Regression, Isolation Forest, StandardScaler |
| **SciPy**        | Statistical analysis      | Comprehensive statistical tests and probability distributions                           |
| **yfinance**     | Market data API           | Free, reliable access to Yahoo Finance historical & real-time data                      |
| **heapq**        | Data structures           | O(n log k) efficient computation of top-N gains/losses                                  |

### Project Structure

| File                 | Description                                                              | Lines of Code |
| -------------------- | ------------------------------------------------------------------------ | ------------- |
| `main.py`            | Main dashboard entrypoint integrating 7 tab modules                      | ~500+         |
| `tabs/`              | Modular tab files (`live.py`, `csv_analysis.py`, `ml_prediction.py`, etc.) | ~1900+        |
| `code.py`            | Legacy monolithic version retained for backward compatibility             | ~2400+        |
| `generate_csv.py`    | Standalone utility to fetch and download stock/crypto CSV data           | ~310          |
| `DOCUMENTATION.md`   | This comprehensive project documentation                                 | —             |
| `research_paper.tex` | LaTeX source for the accompanying research paper                         | —             |

### Design Philosophy & User Experience

The dashboard is designed with a **"progressive disclosure"** approach—beginners see simple summaries while advanced users can dive deeper into statistical details.

- **Modern UI Design**:
  - Vibrant gradient headers using CSS linear-gradients
  - Glassmorphism-style info boxes with backdrop blur effects
  - Interactive metric cards with hover animations (`transform: translateY(-2px)`)
  - Consistent color palette: Blue (#1a73e8) and Purple (#6c5ce7) gradients

- **Beginner-Friendly Explanations**:
  - Each tab includes yellow "explain-boxes" (marked with 🔰)
  - Complex finance/statistics concepts explained using everyday analogies
  - Example: Linear Regression explained as "predicting your test score based on hours studied"

- **Visual Feedback System**:
  - Color-coded signals: 🟢 Bullish (green), 🔴 Bearish (red), 🟡 Neutral (yellow)
  - Emoji-labeled tabs for quick visual scanning
  - Trend indicators with directional arrows (▲ up, ▼ down)

- **Dual-Stock Comparison Workflow**:
  - Upload up to 2 CSV files simultaneously
  - Global **Stock 1 / Stock 2** selector persists across all analysis tabs
  - Side-by-side comparison metrics in Summary tab

- **Smart Currency Detection**:
  - Automatically detects ticker region from filename
  - Indian tickers (`.NS`, `.BO`, `^NSEI`, `^BSESN`) → ₹ (Rupee)
  - US/International tickers → $ (Dollar)

---

## Installation & Setup

### Prerequisites

- **Python 3.8+** (tested on Python 3.10, 3.11)
- **pip** package manager
- **Internet connection** (required for yfinance API calls in Live Chart tab)

### Step-by-Step Installation

```bash
# 1. Clone or download the project
cd "P&S Project Finance"

# 2. (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Install all dependencies
pip install streamlit pandas numpy plotly scikit-learn scipy yfinance

# 4. Verify installation
python -c "import streamlit, pandas, numpy, plotly, sklearn, scipy, yfinance; print('All dependencies installed!')"
```

### Running the Applications

```bash
# Run the main dashboard (7 analysis tabs)
streamlit run main.py
# Opens at: http://localhost:8501

# Run the CSV generator (separate utility)
streamlit run generate_csv.py
# Opens at: http://localhost:8501
```

### Typical Workflow

1. **Generate Data**: Run `generate_csv.py` → Select a stock/crypto → Download CSV
2. **Analyze Data**: Run `main.py` → Upload CSV in "CSV Analysis" tab
3. **Explore**: Navigate through ML Prediction, Statistics, Monte Carlo, and Anomaly Detection tabs
4. **Compare**: Upload a second CSV to enable dual-stock comparison in Summary tab

---

## Feature Documentation

### Tab 1: 📈 Live Chart

**Purpose:** Fetch and visualize real-time stock price data directly from Yahoo Finance without needing to upload any files.

**When to Use:**

- Quick market checks for any ticker
- Real-time price monitoring during market hours
- Exploring new stocks before downloading historical data

**User Interface:**

1. **Ticker Input**: Enter any valid Yahoo Finance symbol (e.g., `AAPL`, `RELIANCE.NS`, `BTC-USD`)
2. **Period Selector**: Choose from 1 month to 5 years of historical data
3. **Chart Style Toggle**: Switch between smooth Line Chart or detailed Candlestick view
4. **Auto-Refresh**: Enable continuous updates for live market monitoring

**Three-Panel Interactive Chart:**
| Panel | Content | What It Shows |
|-------|---------|---------------|
| **Panel 1 — Price** | Close price line (or OHLC candlesticks) + SMA-20 + 68% Probability Band | Main price action with trend overlay and statistical range |
| **Panel 2 — Volume** | Color-coded volume bars | Trading activity; Green = price up, Red = price down |
| **Panel 3 — RSI(14)** | Relative Strength Index oscillator | Momentum indicator with overbought (>70) / oversold (<30) zones |

**Metrics Bar (displayed above chart):**

- **Last Price**: Most recent closing price
- **Change %**: Percentage change from start of selected period
- **Period High/Low**: Maximum and minimum prices in the range
- **Total Volume**: Cumulative shares/units traded

**Range Selector Buttons:** Quickly zoom to 1W, 1M, 3M, 6M, 1Y, or All data.

**🔰 Beginner Explainers Included:**

- What is a "Ticker Symbol"?
- How to read Candlestick charts
- Understanding RSI (Relative Strength Index)
- Why Volume matters in trading

**Statistical Concepts Demonstrated:**

- **Normal Distribution**: The 68% probability band assumes returns are normally distributed
- **Moving Averages**: SMA-20 smooths price data using rolling window statistics
- **Empirical Rule**: 68% of data falls within ±1 standard deviation from the mean

---

### Tab 2: 📂 CSV Analysis

**Purpose:** Upload historical stock data from CSV files for in-depth technical and statistical analysis. This is the entry point for offline data analysis.

**When to Use:**

- Analyzing downloaded historical data
- Comparing two different stocks side-by-side
- Running analysis on custom or pre-processed datasets

**CSV Requirements:**
Your CSV file should contain these columns (case-sensitive):
| Column | Required | Description |
|--------|----------|-------------|
| `Date` | ✅ Yes | Date in YYYY-MM-DD format |
| `Open` | ✅ Yes | Opening price of the trading day |
| `High` | ✅ Yes | Highest price during the day |
| `Low` | ✅ Yes | Lowest price during the day |
| `Close` | ✅ Yes | Closing price (most important!) |
| `Volume` | Optional | Number of shares/units traded |

**Dual-Stock Upload System:**

1. **Stock 1 CSV** (required): Primary dataset for all analysis tabs
2. **Stock 2 CSV** (optional): Enables side-by-side comparison in Summary tab
3. **Stock Selector Buttons**: Switch which dataset feeds into ML, Statistics, Monte Carlo, and Anomaly tabs

**Automatic Feature Engineering:**
When you upload a CSV, the system automatically calculates:

| Feature            | Formula                                                   | What It Measures                     |
| ------------------ | --------------------------------------------------------- | ------------------------------------ |
| `Daily_Return`     | `(Close_today - Close_yesterday) / Close_yesterday × 100` | Daily percentage price change        |
| `Volatility_10`    | `std(Daily_Return)` over rolling 10-day window            | Recent price swing intensity         |
| `RSI_14`           | 14-period Relative Strength Index                         | Momentum oscillator (0-100 scale)    |
| `Expected_High_68` | `mean(Close) + 1 × std(Close)`                            | Upper bound of 68% probability range |
| `Expected_Low_68`  | `mean(Close) - 1 × std(Close)`                            | Lower bound of 68% probability range |

**Auto-Generated Chart Analysis Cards:**
| Card | Key Metrics | Interpretation |
|------|-------------|----------------|
| 📈 **Trend Analysis** | Regression slope, R² score | "Strong uptrend (slope > 0, R² > 0.7)", "Sideways (R² < 0.3)" |
| 📊 **Volatility Assessment** | Average vs recent volatility | "High volatility" if current > 1.5× average |
| ⚡ **Momentum (RSI)** | RSI value | >70: Overbought, <30: Oversold, 45-55: Neutral |
| 🔒 **Key Levels** | 60-day High/Low | Support and Resistance zones with distance from price |
| 🎯 **Win Rate & Risk** | Win %, VaR(95%) | Historical success rate and worst-case daily loss |
| 📝 **Summary Verdict** | Combined signals | Plain-English market sentiment summary |

**Additional Analysis Tools:**

- **Correlation Matrix**: Interactive heatmap showing relationships between OHLCV features
- **Top 5 Gains/Losses**: Using Python's `heapq.nlargest()` and `heapq.nsmallest()` for O(n log k) efficiency
- **Descriptive Statistics Table**: Mean, Std Dev, Min, 25%, 50%, 75%, Max

**Statistical Concepts Demonstrated:**

- **Linear Regression**: Fitting a trend line to price data
- **R² (Coefficient of Determination)**: How much of price variance is explained by time
- **Standard Deviation**: Measuring price dispersion from the mean
- **Correlation Coefficients**: Pearson correlation between features (-1 to +1)
- **Percentiles & Quartiles**: Data distribution analysis
- **Value-at-Risk (VaR)**: 5th percentile of returns as risk measure

---

### Tab 3: 🤖 ML Prediction

**Purpose:** Train a Linear Regression model to predict the next trading day's closing price based on historical patterns.

**When to Use:**

- Getting a data-driven price forecast for short-term planning
- Understanding which factors most influence price movements
- Learning how ML models are trained and evaluated

**How the Model Works:**

```
Step 1: Feature Selection
┌─────────────────────────────────────────────────────────────┐
│  Input Features (X)              │  Target Variable (y)    │
│  ─────────────────               │  ──────────────────     │
│  • Open, High, Low, Close        │  • Next Day's Close     │
│  • Volume                        │    (shifted by 1 day)   │
│  • RSI_14                        │                         │
│  • Expected_High_68              │                         │
│  • Expected_Low_68               │                         │
└─────────────────────────────────────────────────────────────┘

Step 2: Train-Test Split
┌─────────────────────────────────────────────────────────────┐
│  Historical Data (e.g., 500 days)                           │
│  ├── Training Set (70-90%): Model learns patterns           │
│  └── Test Set (10-30%): Evaluate on unseen data             │
└─────────────────────────────────────────────────────────────┘

Step 3: Model Training
┌─────────────────────────────────────────────────────────────┐
│  LinearRegression.fit(X_train, y_train)                     │
│  Finds optimal coefficients: y = β₀ + β₁x₁ + β₂x₂ + ...    │
└─────────────────────────────────────────────────────────────┘

Step 4: Prediction
┌─────────────────────────────────────────────────────────────┐
│  Tomorrow's Price = Model.predict([today's features])       │
└─────────────────────────────────────────────────────────────┘
```

**Configurable Parameters:**

- **Test Split Ratio**: Slider from 10% to 40% (default: 20%)
  - Lower % = More training data, less validation
  - Higher % = More robust validation, less training data

**Output Visualizations:**

1. **Actual vs Predicted Chart**
   - Blue line: Real historical prices (test set)
   - Orange line: Model's predictions
   - Good fit: Lines closely overlap

2. **Tomorrow's Prediction Card**
   - Predicted price with currency symbol
   - Direction indicator: 🟢 UP / 🔴 DOWN
   - Delta: Expected change from today's close

3. **Feature Importance Bar Chart**
   - Shows which inputs have the highest regression coefficients
   - Helps understand what drives the model's decisions

**Model Evaluation Metrics:**
| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **RMSE** | √(Σ(actual - predicted)² / n) | Average prediction error in price units |
| **R² Score** | 1 - (SS_res / SS_tot) | 0.0 = useless, 1.0 = perfect predictions |

**🔰 Beginner Analogy:**

> "Think of it like predicting your exam score. If you studied 5 hours and scored 80%,
> studied 10 hours and scored 90%, Linear Regression finds the pattern:
> each hour of study ≈ +2 points. Tomorrow you study 7 hours → predicted: 84%."

**Statistical Concepts Demonstrated:**

- **Regression Analysis**: Finding linear relationships between variables
- **Train-Test Split**: Preventing overfitting by validating on unseen data
- **RMSE**: Root Mean Squared Error as loss function
- **R² (Coefficient of Determination)**: Explained variance ratio
- **Feature Importance**: Interpreting model coefficients

---

### Tab 4: 📊 Statistics

**Purpose:** Perform comprehensive statistical analysis on the stock's daily return distribution to understand its behavior, risk profile, and distribution characteristics.

**When to Use:**

- Understanding the risk/return profile of a stock
- Checking if returns follow a normal distribution
- Identifying unusual trading periods and outliers
- Comparing volatility across time periods

**Descriptive Statistics Table:**
| Statistic | What It Measures | Stock Market Context |
|-----------|------------------|---------------------|
| **Mean** | Average daily return | Expected daily gain/loss |
| **Std Dev** | Dispersion of returns | Daily risk/volatility measure |
| **Min** | Worst single-day return | Maximum historical daily loss |
| **25% (Q1)** | First quartile | 25% of days performed worse |
| **50% (Median)** | Middle value | Typical daily performance |
| **75% (Q3)** | Third quartile | 75% of days performed worse |
| **Max** | Best single-day return | Maximum historical daily gain |
| **Skewness** | Distribution asymmetry | Negative = more extreme losses, Positive = more extreme gains |
| **Kurtosis** | Tail heaviness | High = more extreme events ("fat tails") |

**Interactive Visualizations:**

1. **Histogram of Daily Returns (60 bins)**
   - Distribution shape of daily percentage changes
   - Vertical lines mark Mean (blue) and Median (green)
   - Bell-shaped curve indicates normal distribution
   - Fat tails suggest extreme events are more common than expected

2. **Box Plot (Box-and-Whisker)**

   ```
   ◄────────────── Outliers ──────────────►
                     ┌─────┐
   ├─────────────────┤     ├─────────────────┤
   Min               Q1   Q2   Q3           Max
                          │
                       Median
   ```

   - Box spans Interquartile Range (IQR = Q3 - Q1)
   - Whiskers extend to 1.5 × IQR
   - Points beyond whiskers are outliers

3. **Cumulative Return Over Time (Area Chart)**
   - Shows total return if you held the stock from day 1
   - Green shading: Positive cumulative return
   - Red shading: Negative cumulative return
   - Formula: `(1 + r₁) × (1 + r₂) × ... × (1 + rₙ) - 1`

4. **Rolling Volatility (10-Day)**
   - Time series of 10-day rolling standard deviation
   - Identifies periods of high and low market turbulence
   - Spikes often correspond to major news events

**🔰 Beginner Explainers:**

- **Histogram**: "Imagine sorting your exam scores into buckets: 60-70, 70-80, 80-90..."
- **Box Plot**: "The box shows where 50% of your grades cluster; outliers are unusually good/bad days"
- **Skewness**: "Negative skew = the stock has more sudden crashes than sudden spikes"
- **Kurtosis**: "High kurtosis = expect more 'surprising' days than a coin flip would suggest"

**Statistical Concepts Demonstrated:**

- **Descriptive Statistics**: Central tendency (mean, median) and spread (std, IQR)
- **Normal Distribution**: Checking if returns are bell-curve shaped
- **Skewness & Kurtosis**: Higher-order moments describing distribution shape
- **Quartiles & Percentiles**: Data distribution milestones
- **Cumulative Distribution**: Compound growth/decline visualization
- **Rolling Statistics**: Time-varying statistical measures
- **Outlier Identification**: Using IQR method (1.5 × IQR rule)

---

### Tab 5: 🎲 Monte Carlo Simulation

**Purpose:** Use random sampling to simulate thousands of possible future price scenarios, providing probability-based forecasts rather than single-point predictions.

**When to Use:**

- Understanding the range of possible future outcomes
- Quantifying the probability of achieving a target price
- Stress-testing portfolio under various market conditions
- Visualizing uncertainty in financial forecasts

**The Monte Carlo Method Explained:**

Instead of predicting "the stock will be at $150," Monte Carlo answers: "There's a 70% chance the stock will be between $140 and $160, with a 5% chance it falls below $120."

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MONTE CARLO SIMULATION PROCESS                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Step 1: Calculate Historical Parameters                               │
│  ┌──────────────────────────────────────┐                              │
│  │  μ (mu) = mean(daily_returns)        │  ← Average daily change     │
│  │  σ (sigma) = std(daily_returns)      │  ← Daily volatility          │
│  └──────────────────────────────────────┘                              │
│                                                                         │
│  Step 2: Generate Random Future (for each simulation)                  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  For day t = 1 to forecast_horizon:                              │  │
│  │      random_return = np.random.normal(μ, σ)                      │  │
│  │      price[t] = price[t-1] × (1 + random_return)                 │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  Step 3: Repeat N Times (1,000 to 10,000 simulations)                  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Simulation 1: $100 → $98 → $102 → $105 → ... → $112             │  │
│  │  Simulation 2: $100 → $103 → $99 → $97 → ... → $89               │  │
│  │  Simulation 3: $100 → $101 → $104 → $108 → ... → $125            │  │
│  │  ...                                                              │  │
│  │  Simulation N: $100 → $99 → $101 → $103 → ... → $107             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  Step 4: Analyze Distribution of Final Prices                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  5th percentile (worst case): $85                                │  │
│  │  50th percentile (median): $108                                  │  │
│  │  95th percentile (best case): $135                               │  │
│  │  Probability of Profit: 65% (simulations ending > $100)          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Configurable Parameters:**
| Parameter | Range | Effect |
|-----------|-------|--------|
| **Number of Simulations** | 100 – 10,000 | More simulations = smoother probability distribution |
| **Forecast Horizon** | 5 – 252 days | Longer horizon = wider probability fan |
| **Confidence Band** | 68% / 90% / 95% | Higher confidence = wider band |

**Output Visualizations:**

1. **Fan Chart (Price Paths)**
   - Up to 100 individual simulation paths (semi-transparent lines)
   - Orange **median path**: Most likely trajectory
   - Shaded region: Selected confidence band
   - Dashed lines: Upper/lower percentile boundaries

2. **Final Price Distribution (Histogram)**
   - Distribution of all simulated end prices
   - Vertical line at starting price (green)
   - Vertical line at median outcome (orange)
   - Shape indicates probability concentration

**Key Metrics Displayed:**
| Metric | Calculation | Interpretation |
|--------|-------------|----------------|
| **Probability of Profit** | `count(final_price > start_price) / N` | Chance of positive return |
| **Expected Price** | `mean(all final prices)` | Average outcome across simulations |
| **Best Case** | `95th percentile of final prices` | Optimistic but realistic scenario |
| **Worst Case** | `5th percentile of final prices` | Pessimistic but realistic scenario |

**Why This Matters:**

> "The Monte Carlo method acknowledges that the future is uncertain. Instead of pretending
> we can predict exactly where a stock will be, we map out the entire landscape of possibilities
> and assign probabilities to each region."

**Statistical Concepts Demonstrated:**

- **Monte Carlo Method**: Using randomness to solve problems that might be deterministic in principle
- **Random Sampling**: Drawing from a probability distribution (Normal Distribution here)
- **Central Limit Theorem**: Many random paths converge to predictable aggregate behavior
- **Confidence Intervals**: Quantifying uncertainty with probability bounds
- **Percentiles**: Finding the value below which X% of observations fall

---

### Tab 6: 🔍 Anomaly Detection

**Purpose:** Automatically identify statistically unusual trading days that deviate significantly from normal market behavior using unsupervised machine learning.

**When to Use:**

- Investigating unusual price movements or volume spikes
- Detecting potential market manipulation or news-driven events
- Quality checking data for errors or outliers
- Understanding which days were "abnormal" compared to typical patterns

**The Isolation Forest Algorithm:**

Unlike traditional outlier detection (which defines outliers as points far from the mean), Isolation Forest takes a different approach: **anomalies are easier to isolate**.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ISOLATION FOREST INTUITION                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Imagine a 2D scatter plot of Volume vs Daily_Return:                  │
│                                                                         │
│      Daily_Return                                                       │
│           │      ✕ (anomaly - isolated quickly!)                       │
│           │                                                             │
│           │     ●●●●●                                                   │
│           │    ●●●●●●●                                                  │
│           │   ●●●●●●●●●   ← Normal data cluster (hard to isolate)      │
│           │    ●●●●●●●                                                  │
│           │     ●●●●●                                                   │
│           │                                                             │
│           │                    ✕ (another anomaly)                      │
│           └────────────────────────────────────── Volume               │
│                                                                         │
│  The algorithm randomly picks a feature and split value.               │
│  Points far from the cluster get isolated in fewer splits.             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Algorithm Pipeline:**

```python
# Step 1: Prepare feature matrix (8 features per day)
features = ['Open', 'High', 'Low', 'Close', 'Volume',
            'Daily_Return', 'Volatility_10', 'RSI_14']

# Step 2: Standardize features (Z-score normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Step 3: Train Isolation Forest
from sklearn.ensemble import IsolationForest
model = IsolationForest(
    contamination=anomaly_rate,  # Expected % of anomalies (1-10%)
    n_estimators=n_trees,        # Number of isolation trees (50-300)
    random_state=42
)
model.fit(X_scaled)

# Step 4: Get anomaly scores and labels
scores = model.decision_function(X_scaled)  # Higher = more normal
labels = model.predict(X_scaled)             # -1 = anomaly, 1 = normal
```

**Configurable Parameters:**
| Parameter | Range | Effect |
|-----------|-------|--------|
| **Expected Anomaly Rate** | 1% – 10% | Higher = more days flagged as anomalies |
| **Number of Trees** | 50 – 300 | More trees = more stable/robust detection |

**Output Visualizations:**

1. **Price Chart with Anomaly Markers**
   - Close price line over time
   - Red **✕** markers on anomalous days
   - Hover for exact date and price

2. **Anomaly Score Timeline**
   - Score for each trading day (higher = more normal)
   - Horizontal threshold line (dashed)
   - Points below threshold are classified as anomalies

3. **Anomalous Days Detail Table**
   | Date | Close | Daily_Return | Anomaly_Score |
   |------|-------|--------------|---------------|
   | 2024-03-15 | $152.30 | -5.2% | -0.142 |
   | 2024-01-08 | $148.75 | +7.8% | -0.198 |

**Interpreting Anomaly Scores:**

- **Score > 0**: Normal behavior (closer to 0 = more borderline)
- **Score < 0**: Anomalous behavior (more negative = more extreme)
- The threshold is automatically set based on the contamination parameter

**Common Causes of Anomalies in Stock Data:**

- Earnings announcements causing large price gaps
- Market crashes or flash crashes
- Data errors or stock splits not adjusted
- Unusual volume spikes (institutional trading)
- Breaking news events

**Statistical Concepts Demonstrated:**

- **Outlier Detection**: Identifying data points that don't fit the pattern
- **Ensemble Methods**: Combining multiple decision trees for robust detection
- **Feature Scaling (Z-Score)**: Standardizing features to mean=0, std=1
- **Contamination Parameter**: Prior probability of anomalies
- **Decision Function**: Converting model output to interpretable scores

---

### Tab 7: 📋 Summary

**Purpose:** Consolidate all analysis results into a single executive-style report with actionable insights and investment signals.

**When to Use:**

- Getting a quick overview without navigating multiple tabs
- Comparing two stocks to make an investment decision
- Generating a summary for reports or presentations
- Understanding the overall market sentiment for a stock

**Report Sections:**

#### Section 1: Data Overview

| Metric             | Description                      |
| ------------------ | -------------------------------- |
| Total Trading Days | Number of data points in the CSV |
| Date Range         | First date → Last date           |
| Start Price        | Opening price of the first day   |
| End Price          | Closing price of the last day    |
| Total Return       | `(End - Start) / Start × 100%`   |

#### Section 2: Statistical Profile

| Metric             | What It Tells You                                |
| ------------------ | ------------------------------------------------ |
| Mean Daily Return  | Average daily gain/loss percentage               |
| Standard Deviation | Daily volatility (risk measure)                  |
| Skewness           | Distribution asymmetry (negative = more crashes) |
| Kurtosis           | Tail heaviness (high = more extreme days)        |
| Win Rate           | Percentage of positive return days               |
| VaR (95%)          | Worst expected daily loss at 95% confidence      |

#### Section 3: Trend Analysis

- **Regression Slope**: Direction and steepness of trend
- **R² Score**: How well the trend explains price movement
- **Trend Classification**: Strong/Mild Uptrend, Downtrend, or Sideways

#### Section 4: Technical Indicators

- **RSI Reading**: Current momentum status (Overbought/Oversold/Neutral)
- **Current vs Average Volatility**: Is the stock more or less volatile than usual?
- **Support/Resistance Levels**: Key price zones based on 60-day high/low

#### Section 5: ML Prediction Results

- **Model Accuracy**: R² score and RMSE from Linear Regression
- **Tomorrow's Predicted Close**: Forecasted price with UP/DOWN signal
- **Confidence Margin**: Prediction uncertainty (based on RMSE)

#### Section 6: Key Probability Metrics

| Metric                  | Calculation                | Investment Insight                               |
| ----------------------- | -------------------------- | ------------------------------------------------ |
| 68% Range               | Mean ± 1σ                  | "Most days, the price stays within this range"   |
| Win Rate                | Positive days / Total days | "Historical odds of a green day"                 |
| VaR (95%)               | 5th percentile of returns  | "On a bad day, expect to lose at most this much" |
| Skewness Interpretation | Plain English              | "Stock has more sudden drops than sudden spikes" |

#### Section 7: ML + Monte Carlo Forecast (Combined View)

This section synthesizes machine learning predictions with Monte Carlo simulations:

```
┌───────────────────────────────────────────────────────────────────┐
│                    COMBINED FORECAST CARD                         │
├───────────────┬───────────────────┬───────────────────────────────┤
│  ML Prediction │  MC Expected Price │  Consensus (Average)        │
│    $152.30     │      $149.85       │       $151.08               │
│  (Linear Reg)  │  (10,000 sims)     │  (Weighted blend)           │
└───────────────┴───────────────────┴───────────────────────────────┘
```

- **Confidence-Weighted Signal**: Combines ML R² score and MC Probability of Profit
  - 🟢 **Bullish** (composite > 60%): Both models agree on upside
  - 🟡 **Neutral** (composite 40-60%): Mixed signals or uncertainty
  - 🔴 **Bearish** (composite < 40%): Both models suggest downside

- **30-Day Mini Fan Chart**: Compact Monte Carlo visualization showing:
  - Multiple simulated price paths
  - ML prediction point plotted at Day 1
  - Visual comparison of where regression sits within probability fan

- **Risk-Reward Summary**:
  - Best Case (95th percentile from MC)
  - Worst Case (5th percentile from MC)
  - ML Uncertainty Margin (RMSE)

#### Section 8: Methods Used Table

Comprehensive mapping of techniques to P&S concepts (for academic reference).

#### Section 9: Dual-Stock Investor Pick

_Only appears when both Stock 1 and Stock 2 CSVs are uploaded_

Computes a **Health/Behavior Score** for each stock based on:

- Total Return
- Win Rate
- Volatility (lower is better)
- Trend strength (R² × slope direction)

Side-by-side comparison with a **"Better Pick"** recommendation highlighted.

#### Section 10: Appendix - Plain-English Glossary

Beginner-friendly definitions of all technical terms used in the dashboard:

- Rolling Mean, Std Dev, 68% Band, Percentiles
- Linear Regression, R², RSI, Support/Resistance
- Monte Carlo terms (Probability of Profit, Confidence Bands)
- Isolation Forest (Anomaly Score, Contamination)
- Risk terms (VaR, Skewness, Kurtosis, Health Score)

---

## Standalone Tool: CSV Generator (`generate_csv.py`)

**Purpose:** Fetch and download historical market data for any stock, cryptocurrency, or index from Yahoo Finance.

**When to Use:**

- Before using the main dashboard (need CSV data to upload)
- Building a local dataset for analysis
- Downloading data for multiple assets to compare

**Step-by-Step Usage:**

```bash
# Start the CSV generator
streamlit run generate_csv.py
```

**User Interface Workflow:**

```
┌──────────────────────────────────────────────────────────────────────────┐
│  📥 STOCK & CRYPTO CSV GENERATOR                                         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STEP 1: Select a Stock, Crypto, or Index                               │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  ○ Search from list    ● Type custom ticker                     │    │
│  │  ┌──────────────────────────────────────────┐                   │    │
│  │  │ BTC-USD - Bitcoin              ▼         │  ← Searchable     │    │
│  │  └──────────────────────────────────────────┘    dropdown       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  STEP 2: Choose Time Period & Interval                                  │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Period: [1y ▼]     Interval: [1d ▼]                             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  STEP 3: [🔍 Fetch Data]                                                │
│                                                                          │
│  STEP 4: Preview & Download                                             │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Date       | Open     | High     | Low      | Close    | Volume │   │
│  │  2023-04-01 | 28215.32 | 28456.78 | 28012.45 | 28345.67 | 12.3M  │   │
│  │  2023-04-02 | 28345.67 | 28678.90 | 28234.56 | 28567.89 | 14.5M  │   │
│  │  ...                                                              │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  [⬇️ Download BTC-USD_1y_1d.csv]                                        │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Pre-populated Ticker Categories:**

| Category                | Examples                                            | Count |
| ----------------------- | --------------------------------------------------- | ----- |
| **Cryptocurrency**      | BTC-USD, ETH-USD, SOL-USD, BNB-USD, DOGE-USD        | 12+   |
| **US Large Cap**        | AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA           | 35+   |
| **Indian Stocks (NSE)** | RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS           | 25+   |
| **Global Indices**      | ^GSPC (S&P 500), ^DJI (Dow Jones), ^NSEI (Nifty 50) | 4     |

**Configuration Options:**

| Option            | Choices                                 | Description                |
| ----------------- | --------------------------------------- | -------------------------- |
| **Time Period**   | 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max     | How far back to fetch data |
| **Data Interval** | 1d (daily), 1wk (weekly), 1mo (monthly) | Granularity of data points |

**Quick Stats Preview:**
Before downloading, you can see:

- Total number of rows
- Date range covered
- Start and end prices
- Total percentage change

**Output CSV Format:**

```csv
Date,Open,High,Low,Close,Volume
2024-01-02,185.45,186.23,184.67,185.89,45234567
2024-01-03,186.12,187.45,185.34,186.78,38765432
...
```

**Integration with Main Dashboard:**
After downloading, upload the CSV file to the main dashboard:

1. Run `streamlit run main.py`
2. Navigate to **CSV Analysis** tab
3. Upload the downloaded CSV file
4. All analysis tabs will populate with your data

---

## Probability & Statistics Concepts Reference

This section provides a comprehensive mapping of all P&S concepts used in the project, organized by category.

### Descriptive Statistics

| Concept                    | Formula/Definition           | Dashboard Usage                               |
| -------------------------- | ---------------------------- | --------------------------------------------- |
| **Mean (μ)**               | `Σx / n`                     | Average daily return, expected price level    |
| **Median**                 | Middle value when sorted     | More robust center measure in Statistics tab  |
| **Standard Deviation (σ)** | `√(Σ(x-μ)² / n)`             | Volatility measure, 68% band width            |
| **Variance**               | `σ²`                         | Squared volatility, used in many calculations |
| **Range**                  | `max - min`                  | Price spread, support/resistance distance     |
| **Quartiles (Q1, Q2, Q3)** | 25th, 50th, 75th percentiles | Box plot construction                         |
| **IQR**                    | `Q3 - Q1`                    | Interquartile range for outlier detection     |

### Probability Distributions

| Concept                         | Formula/Definition                 | Dashboard Usage                      |
| ------------------------------- | ---------------------------------- | ------------------------------------ |
| **Normal Distribution**         | `N(μ, σ²)`                         | Monte Carlo random returns, 68% band |
| **Empirical Rule (68-95-99.7)** | Data within 1σ, 2σ, 3σ             | Probability bands on charts          |
| **Percentiles**                 | Value below which X% of data falls | VaR, Monte Carlo bounds              |
| **Skewness**                    | Third moment: `E[(X-μ)³]/σ³`       | Distribution asymmetry indicator     |
| **Kurtosis**                    | Fourth moment: `E[(X-μ)⁴]/σ⁴ - 3`  | Tail heaviness ("fat tails")         |

### Regression Analysis

| Concept                               | Formula/Definition  | Dashboard Usage               |
| ------------------------------------- | ------------------- | ----------------------------- |
| **Linear Regression**                 | `y = β₀ + β₁x + ε`  | Trend analysis, ML prediction |
| **R² (Coefficient of Determination)** | `1 - SS_res/SS_tot` | Model fit quality (0 to 1)    |
| **RMSE**                              | `√(Σ(y - ŷ)² / n)`  | Prediction error magnitude    |
| **Coefficients (β)**                  | Regression weights  | Feature importance chart      |
| **Train-Test Split**                  | Holdout validation  | Preventing overfitting        |

### Time Series Analysis

| Concept                | Formula/Definition         | Dashboard Usage            |
| ---------------------- | -------------------------- | -------------------------- |
| **Rolling Mean (SMA)** | Moving average over window | SMA-20 on price charts     |
| **Rolling Std Dev**    | Moving std dev over window | Volatility_10 feature      |
| **Cumulative Return**  | `∏(1 + rᵢ) - 1`            | Performance tracking chart |
| **RSI Formula**        | `100 - 100/(1 + RS)`       | Momentum oscillator        |

### Monte Carlo Methods

| Concept                   | Formula/Definition                    | Dashboard Usage               |
| ------------------------- | ------------------------------------- | ----------------------------- |
| **Random Sampling**       | Drawing from probability distribution | Simulating future returns     |
| **Central Limit Theorem** | Sample means → Normal distribution    | Aggregate simulation behavior |
| **Confidence Intervals**  | `μ ± z × (σ/√n)`                      | Fan chart bands               |
| **Expected Value**        | `E[X] = Σ(xᵢ × P(xᵢ))`                | Mean of simulated outcomes    |

### Machine Learning Concepts

| Concept                       | Definition                   | Dashboard Usage                        |
| ----------------------------- | ---------------------------- | -------------------------------------- |
| **Feature Engineering**       | Creating derived variables   | Daily_Return, Volatility_10, RSI_14    |
| **Feature Scaling (Z-Score)** | `(x - μ) / σ`                | StandardScaler before Isolation Forest |
| **Isolation Forest**          | Tree-based anomaly detection | Unusual trading day detection          |
| **Contamination Parameter**   | Expected anomaly fraction    | User-configurable (1-10%)              |

### Risk Metrics

| Concept                 | Formula/Definition           | Dashboard Usage                |
| ----------------------- | ---------------------------- | ------------------------------ |
| **Value-at-Risk (VaR)** | Loss at X percentile         | 5th percentile of returns      |
| **Win Rate**            | `Positive days / Total days` | Historical success probability |
| **Sharpe-like Ratio**   | `(Mean Return) / Std Dev`    | Implicit in health score       |

### Data Structures & Algorithms

| Concept                   | Python Implementation                   | Dashboard Usage                 |
| ------------------------- | --------------------------------------- | ------------------------------- |
| **Heap (Priority Queue)** | `heapq.nlargest()`, `heapq.nsmallest()` | Top 5 gains/losses (O(n log k)) |
| **Correlation Matrix**    | `df.corr()` (Pearson correlation)       | Feature relationship heatmap    |

---

## Technical Architecture

### Code Organization (`main.py` + `tabs/`)

```
main.py (Main Dashboard Integrator)
├── Imports & Configuration (lines 1-25)
├── Custom CSS Styling (lines 25-240)
├── Helper Functions
│   ├── compute_rsi()          # RSI calculation
│   ├── detect_currency_symbol() # ₹ vs $ detection
│   ├── load_and_clean_csv()   # CSV parsing & feature engineering
│   ├── fmt_price()            # Currency formatting
│   └── render_stock_selector() # Shared stock selector state
├── Streamlit tab containers
└── Tab module loader (exec-based integration)

tabs/
├── live.py              # Tab 1: Live Chart
├── csv_analysis.py      # Tab 2: CSV Analysis
├── ml_prediction.py     # Tab 3: ML Prediction
├── statistics.py        # Tab 4: Statistics
├── monte_carlo.py       # Tab 5: Monte Carlo
├── anomaly_detection.py # Tab 6: Anomaly Detection
└── summary.py           # Tab 7: Summary
```

### Session State Management

Streamlit uses `st.session_state` to persist data across interactions:

```python
# Key session state variables
st.session_state.df1         # Stock 1 DataFrame
st.session_state.df2         # Stock 2 DataFrame
st.session_state.active_stock  # Currently selected stock (1 or 2)
st.session_state.ml_model    # Trained Linear Regression model
st.session_state.ml_rmse     # Model RMSE
st.session_state.ml_r2       # Model R² score
```

### Modular Architecture Notes

- `main.py` is now the recommended entrypoint (`streamlit run main.py`).
- Each analysis tab is isolated in its own file under `tabs/`.
- Shared state (uploaded files, active stock selector, ML/Monte Carlo outputs) is still preserved through `st.session_state`.
- `code.py` remains available as a legacy single-file version if needed for rollback/reference.

### Performance Considerations

| Operation          | Optimization        | Why                                    |
| ------------------ | ------------------- | -------------------------------------- |
| Top N gains/losses | `heapq.nlargest()`  | O(n log k) vs O(n log n) for full sort |
| Feature scaling    | `StandardScaler`    | Fit once, transform efficiently        |
| Monte Carlo        | NumPy vectorization | Avoid Python loops                     |
| Chart rendering    | Plotly WebGL        | GPU-accelerated for large datasets     |

---

## Dependencies

| Package        | Version | Purpose                                       |
| -------------- | ------- | --------------------------------------------- |
| `streamlit`    | ≥1.28   | Web application framework                     |
| `pandas`       | ≥2.0    | DataFrame operations                          |
| `numpy`        | ≥1.24   | Numerical computing                           |
| `plotly`       | ≥5.18   | Interactive visualizations                    |
| `scikit-learn` | ≥1.3    | ML models (LinearRegression, IsolationForest) |
| `scipy`        | ≥1.11   | Statistical functions                         |
| `yfinance`     | ≥0.2.31 | Yahoo Finance API wrapper                     |

**Installation:**

```bash
pip install streamlit pandas numpy plotly scikit-learn scipy yfinance
```

**Verify Installation:**

```bash
python -c "import streamlit, pandas, numpy, plotly, sklearn, scipy, yfinance; print('All dependencies installed!')"
```

---

## Troubleshooting

### Common Issues

| Problem                  | Cause                 | Solution                                     |
| ------------------------ | --------------------- | -------------------------------------------- |
| "yfinance not installed" | Missing dependency    | `pip install yfinance`                       |
| "No data returned"       | Invalid ticker symbol | Check Yahoo Finance for correct format       |
| Empty CSV Analysis       | Incorrect CSV format  | Ensure columns: Date, Open, High, Low, Close |
| Slow Monte Carlo         | Too many simulations  | Reduce to 1,000-5,000                        |
| RSI shows NaN            | Not enough data       | Need ≥14 rows for RSI calculation            |

### CSV Format Requirements

✅ **Valid CSV:**

```csv
Date,Open,High,Low,Close,Volume
2024-01-02,185.45,186.23,184.67,185.89,45234567
```

❌ **Invalid CSV:**

```csv
date,open,high,low,close,volume   # Wrong: lowercase headers
2024/01/02,185.45,...             # Wrong: slash date format
```

---

## Future Enhancements

Potential improvements for future versions:

1. **Additional ML Models**: Random Forest, LSTM neural networks
2. **Portfolio Analysis**: Multi-stock portfolio optimization
3. **Real-Time Alerts**: Price threshold notifications
4. **Sentiment Analysis**: News-based sentiment scoring
5. **Backtesting Engine**: Historical strategy testing
6. **Export Reports**: PDF/Excel report generation

---

## License & Credits

- **Data Source**: Yahoo Finance via yfinance library
- **Framework**: Streamlit by Snowflake
- **Visualization**: Plotly.js
- **ML Library**: scikit-learn

_This project was developed as part of a Probability & Statistics course at VIT._
