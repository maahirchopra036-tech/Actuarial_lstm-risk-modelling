# Stock Price Prediction & Risk Analysis Using LSTM Neural Networks
### Stochastic Modelling of Equity Returns — Apple (AAPL) & JPMorgan Chase (JPM)

---

## Overview

This project applies **Long Short-Term Memory (LSTM) neural networks** to model and forecast the closing prices of two major equities — Apple Inc. (AAPL) and JPMorgan Chase (JPM) — over a 10-year historical window. Beyond price prediction, the project incorporates **return distribution analysis**, **simulated trading strategy evaluation**, and **multi-horizon forecasting** (1-year and 2-year), producing a framework that sits at the intersection of machine learning and quantitative risk modelling.

The methodology draws on concepts central to both **actuarial science** and **quantitative finance**: stochastic process theory, time-series modelling, model error quantification, and scenario-based return simulation. It is applicable to contexts including asset-liability management, economic capital modelling, and investment strategy stress-testing.

---

## Why This Matters — Actuarial & Consulting Context

### Stock Returns as a Stochastic Process

Daily equity returns are commonly modelled as a **discrete-time stochastic process**. Under the classic assumption used in options pricing and actuarial investment models, log-returns follow:

```
ln(S_t / S_{t-1}) ~ N(μ, σ²)
```

This underpins the **lognormal model** used in Black-Scholes, and is the same distributional assumption applied in **with-profits actuarial reserving** and **unit-linked policy modelling** under Solvency II. The daily return series computed in this project directly represents this process empirically — without imposing the normality assumption, allowing the data to reveal fat tails, volatility clustering, and regime changes that simpler models miss.

LSTM networks are particularly suited to financial time series because they capture **temporal dependencies and non-stationarity** — properties that violate the i.i.d. assumptions of simpler regression models, but which are characteristic of real market data.

### LSTM as a Discrete-Time Sequential Model

An LSTM is a type of **recurrent neural network** that maintains a hidden state across time steps, allowing it to learn long-range dependencies in sequential data. Formally, at each time step *t*, the model updates:

```
Cell state:   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
Hidden state: h_t = o_t ⊙ tanh(c_t)
```

where *f*, *i*, *o*, *g* are learned gate vectors controlling what information to forget, input, and output. This structure is analogous to **autoregressive time-series models** (AR, ARIMA, VAR) used in actuarial economic scenario generation, but with the added capacity to learn non-linear relationships.

This project uses a **60-day lookback window** — meaning the model conditions each prediction on the prior 60 trading days, reflecting the kind of rolling window assumptions made in **chain-ladder reserving** and **dynamic financial analysis (DFA)** models.

### Model Architecture

```
Input Layer     →  shape (60 timesteps, 1 feature)
LSTM Layer 1    →  50 units, return_sequences=True
Dropout (20%)   →  regularisation to prevent overfitting
LSTM Layer 2    →  50 units
Dropout (20%)   →  regularisation
Dense Output    →  1 unit (next-day price)
Loss Function   →  Mean Squared Error (MSE)
Optimiser       →  Adam (adaptive learning rate)
```

Dropout regularisation is the neural network analogue of **credibility weighting** in actuarial models — it prevents the model from over-fitting to noise in the training data, improving out-of-sample generalisation.

---

## Project Structure

```
├── lstm_stock_prediction.ipynb   # Main notebook
├── README.md                     # This file
```

---

## Dependencies

| Library | Purpose |
|---|---|
| `yfinance` | Market data download (AAPL, JPM) |
| `pandas` | Data manipulation and time-series indexing |
| `numpy` | Numerical operations and array handling |
| `matplotlib` | Visualisation |
| `scikit-learn` | MinMaxScaler, RMSE computation |
| `tensorflow / keras` | LSTM model construction and training |

Install all dependencies:

```bash
pip install yfinance pandas numpy matplotlib scikit-learn tensorflow
```

---

## Methodology

### 1. Data Acquisition & Exploratory Analysis

Ten years of daily OHLCV (Open, High, Low, Close, Volume) data is retrieved for both tickers. Initial exploratory steps include:

- **Closing price time series** — visualises the long-run price trend, capturing major market events (2020 COVID crash, 2022 rate-hike selloff)
- **Daily returns** (`pct_change()`) — the empirical realisation of the stochastic return process; volatile clustering visible in the series reflects **heteroskedasticity**, a key feature motivating GARCH models in actuarial economic scenario generators
- **20-day moving average** — a smoothed signal that filters short-term noise; analogous to **graduated mortality rates** in actuarial graduation, where raw data is smoothed to reveal underlying trends

### 2. Data Preprocessing & Normalisation

Closing prices are normalised to [0, 1] using **Min-Max scaling**:

```
x_scaled = (x - x_min) / (x_max - x_min)
```

This is essential for gradient-based optimisation in neural networks. Predictions are subsequently inverse-transformed to recover actual USD prices — preserving the interpretability required for financial reporting.

Sequences of 60 consecutive trading days are constructed as inputs *X*, with the following day's price as the target *y*. This sliding-window approach is conceptually identical to the **development year triangles** used in claims reserving, where each diagonal represents a successive time window of observed data.

### 3. Model Training

Separate LSTM models are trained independently for Apple and JPMorgan. Key parameters:

- **Epochs:** 5 (sufficient for convergence given data volume; increasing epochs risks overfitting)
- **Batch size:** 32 (mini-batch stochastic gradient descent)
- **Validation:** Training loss monitored across epochs; loss converges steadily, indicating stable learning

Training on the full historical dataset allows the model to learn across multiple market regimes — bull markets, corrections, and recovery — mirroring the **multi-scenario** approach in actuarial stress testing under IFoA/IFRS 17 standards.

### 4. Performance Evaluation — Error Metrics

Two standard error metrics are computed:

**Root Mean Squared Error (RMSE):**
```
RMSE = √( (1/n) Σ (y_actual - y_predicted)² )
```
RMSE penalises large deviations heavily due to the squared term — analogous to **quadratic loss functions** used in actuarial credibility theory (Bühlmann-Straub model). In portfolio risk contexts, RMSE maps to **tracking error** against a benchmark.

**Mean Absolute Percentage Error (MAPE):**
```
MAPE = (1/n) Σ |( y_actual - y_predicted ) / y_actual| × 100%
```
MAPE provides a scale-independent relative error, directly comparable across the two tickers regardless of their price levels. This mirrors the use of **proportional error measures** in experience analysis and mortality investigations, where absolute deviations in large portfolios can be misleading without normalisation.

### 5. Simulated Trading Strategy

A simple **signal-based trading strategy** is implemented: the model goes long (holds the stock) on days where its prediction implies an upward move, and stays flat otherwise:

```python
if predicted[t+1] > predicted[t]:
    strategy_return = actual_daily_return[t]
else:
    strategy_return = 0
```

Cumulative returns are computed via the **geometric compounding** formula:

```
Cumulative Return = ∏ (1 + r_t)
```

This is directly analogous to the **accumulation factors** used in actuarial present value calculations and **asset share projections** for with-profits business. The comparison of strategy returns against buy-and-hold benchmarks provides a signal-to-noise assessment of model skill — a framework parallel to **information ratio** analysis in investment mandates.

> **Note:** This simulation is in-sample (trained and evaluated on the same data) and should be interpreted as a proof-of-concept rather than a live trading signal. Out-of-sample backtesting would be required for robust strategy validation — a distinction central to actuarial **model validation** standards under TAS 100.

### 6. Multi-Horizon Forecasting

The trained model is used to generate **rolling iterative forecasts** of 1 year (252 trading days) and 2 years (504 trading days) beyond the historical window. At each step, the model's own previous prediction is fed back as input:

```python
current_batch = np.append(current_batch[:,1:,:], [[pred]], axis=1)
```

This autoregressive forecasting approach is analogous to **stochastic projection models** in actuarial liability valuation — for example, projecting future mortality rates beyond observed data using a Lee-Carter or CBD model, where each projected value feeds into the next period's estimate. The compounding of prediction error over longer horizons is an inherent limitation, reflected in widening uncertainty intervals in practice.

---

## Results Summary

| Metric | Apple (AAPL) | JPMorgan (JPM) |
|---|---|---|
| RMSE | Reported at runtime | Reported at runtime |
| MAPE | Reported at runtime | Reported at runtime |

Both models achieve low RMSE relative to the price levels of each stock, with MAPE values indicating close tracking of the underlying price series on the training set.

---

## Key Visualisations

**1. Historical Closing Prices** — full 10-year price series for both equities, capturing long-run trend and major drawdown events.

**2. Daily Return Series** — empirical realisation of the stochastic return process; volatility clustering visible, consistent with GARCH-type behaviour.

**3. 20-Day Moving Average Overlay** — graduated price signal alongside raw closes, illustrating trend extraction from noisy data.

**4. Actual vs. Predicted Prices** — LSTM fit across the training window for both tickers on a common date axis, demonstrating model capture of price dynamics.

**5. Simulated Cumulative Returns** — strategy vs. buy-and-hold comparison for both stocks, illustrating the added value (or otherwise) of the directional signal.

**6. 1-Year & 2-Year Forecasts** — projected price paths for AAPL and JPM beyond the observed data, with uncertainty implicit in the autoregressive compounding.

---

## Limitations & Extensions

**Current limitations:**

- Model is trained and evaluated on the same dataset (in-sample) — a formal train/test split with hold-out validation would be required for rigorous out-of-sample assessment
- No external features (macroeconomic indicators, earnings data, sentiment) are incorporated; a multivariate LSTM could integrate these as additional input channels
- Forecast uncertainty is not quantified; a **Monte Carlo dropout** approach or **conformal prediction intervals** would provide probabilistic bounds on forecasts — a standard requirement in actuarial reporting under uncertainty

**Natural extensions for actuarial application:**

- **Covariance modelling:** With two assets modelled jointly, a natural extension is computing the rolling return covariance matrix between AAPL and JPM. This feeds directly into **portfolio variance** (σ²_p = w'Σw) and is the foundation of Markowitz mean-variance optimisation and actuarial **asset-liability matching** frameworks
- **VaR / CVaR estimation:** The empirical return distribution from this model can be used to estimate **Value at Risk** and **Conditional Value at Risk** — standard risk capital metrics under Solvency II internal models
- **Volatility modelling:** Combining LSTM price predictions with a **GARCH(1,1)** model on residuals would capture the heteroskedastic volatility structure apparent in the daily return series
- **Mortality/lapse analogue:** The same LSTM architecture can be applied to model policyholder behaviour time series (e.g., monthly lapse rates), making this methodology directly transferable to life insurance experience analysis

---

## Technical Notes

- `MinMaxScaler` is fitted on the full dataset before sequence creation; in a strict out-of-sample framework, the scaler should be fitted on training data only to prevent data leakage
- The `yfinance` library applies automatic price adjustment for splits and dividends (`auto_adjust=True` by default in recent versions), ensuring returns reflect true economic performance
- Future date generation uses `freq='B'` (business days), correctly excluding weekends from the forecast horizon

---

## Tools & Environment

- Python 3.10+
- TensorFlow 2.x / Keras
- Developed in Google Colab (GPU runtime recommended for faster training)

---

## Author

*Quantitative Finance & Risk Modelling Project*
*Built to demonstrate applied machine learning in an actuarial and financial analysis context*
