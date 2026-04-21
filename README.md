# Deep Learning Time-Series Forecasting with Stochastic Financial Foundations.

> **LSTM Neural Networks · Geometric Brownian Motion · Actuarial Risk Metrics · Portfolio Covariance Analysis**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This project constructs a production-grade **Long Short-Term Memory (LSTM) neural network** to forecast equity prices for **Apple Inc. (AAPL)** and **JPMorgan Chase & Co. (JPM)**, grounded entirely within an **actuarial and stochastic mathematics** framework.

Every component — from data preparation through forecast interpretation — is embedded in the mathematical theory underpinning actuarial practice: **Geometric Brownian Motion**, **Itô's Lemma**, **coherent risk measures (CVaR)**, **covariance decomposition**, and **Solvency II-style stress testing**.

This is not a black-box prediction exercise. It is a demonstration of how **deep learning can be applied responsibly** when the practitioner understands the stochastic system generating the data.

---

## Project Structure

```
actuarial_lstm_project/
│
├── actuarial_lstm_project.ipynb   ← Main Jupyter notebook (11 sections)
├── requirements.txt               ← Package dependencies
└── README.md                      ← This file
```

---

## Sections at a Glance

| # | Section | Key Methods |
|---|---------|-------------|
| 1 | Environment Setup & Data Acquisition | `yfinance`, 10-year OHLCV data |
| 2 | Stationarity Testing & Log Return Analysis | ADF test, Jarque-Bera, Q-Q plots, skew/kurtosis |
| 3 | Covariance Matrix & Portfolio Risk Decomposition | Σ matrix, Efficient Frontier, Marginal Risk Contributions |
| 4 | Geometric Brownian Motion — Monte Carlo Simulation | Itô-corrected GBM, 500-path fan charts, percentile bands |
| 5 | LSTM Architecture — Construction & Training | Stacked LSTM, BatchNorm, Dropout, EarlyStopping, Huber loss |
| 6 | Model Evaluation | RMSE, MAPE, Directional Accuracy, Theil's U |
| 7 | Actuarial Risk Metrics | Parametric & Historical VaR/CVaR (95%), Max Drawdown, Calmar Ratio |
| 8 | Multi-Horizon Forecasting | 1-Year & 2-Year LSTM forecasts vs GBM fan chart |
| 9 | Simulated Trading Strategy | Long-only signal with transaction costs, Sharpe ratio |
| 10 | Stress Testing & Scenario Analysis | Market Crash / Bull Run / High Volatility — GBM shocks |
| 11 | Summary Dashboard | Full metrics table across all frameworks |

---

## Mathematical Foundations

### Geometric Brownian Motion

Asset prices evolve under the stochastic differential equation:

$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$$

By **Itô's Lemma**, the exact discrete solution used in simulation is:

$$S_{t+\Delta t} = S_t \exp\!\left[\left(\mu - \tfrac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}\, Z_t\right], \quad Z_t \sim \mathcal{N}(0,1)$$

The $-\frac{1}{2}\sigma^2$ **Itô correction** ensures $\mathbb{E}[S_t] = S_0 e^{\mu t}$.

### Covariance & Portfolio Risk

For a two-asset portfolio with weight vector $w$:

$$\sigma_p^2 = w^\top \Sigma w, \qquad \Sigma = \begin{pmatrix} \sigma^2_{\text{AAPL}} & \sigma_{\text{AAPL,JPM}} \\ \sigma_{\text{JPM,AAPL}} & \sigma^2_{\text{JPM}} \end{pmatrix}$$

**Marginal Risk Contributions** decompose portfolio volatility to individual assets — the standard approach in Solvency II economic capital models.

### Coherent Risk Measures

**VaR** at confidence level $\alpha$:

$$\text{VaR}_\alpha = -\left(\hat{\mu} + z_\alpha \hat{\sigma}\right)$$

**CVaR / Expected Shortfall** — satisfies subadditivity (coherent):

$$\text{CVaR}_\alpha = -\hat{\mu} + \hat{\sigma} \cdot \frac{\phi(z_\alpha)}{1 - \alpha}$$

Both parametric and historical (non-parametric) versions are computed, exposing the gap created by fat tails in empirical return distributions.

### LSTM Gate Equations

| Gate | Equation |
|------|----------|
| Input | $i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$ |
| Forget | $f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$ |
| Cell update | $c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c x_t + U_c h_{t-1} + b_c)$ |
| Output | $h_t = o_t \odot \tanh(c_t)$ |

---

## Outputs

The notebook produces the following quantitative outputs for each asset:

**Statistical:**
- ADF test statistic, p-value, critical values at 1% / 5% / 10%
- Log return distribution statistics: μ, σ, skewness, excess kurtosis, Sharpe ratio
- Jarque-Bera normality test

**Risk Metrics:**
- Parametric VaR (95%) and CVaR (95%) — daily
- Historical VaR (95%) and CVaR (95%) — daily
- Maximum Drawdown and Calmar Ratio

**Portfolio:**
- Daily and annualised covariance matrix Σ
- Correlation matrix ρ
- Efficient Frontier curve with Minimum Variance and Maximum Sharpe portfolios
- Marginal Risk Contributions and diversification benefit

**Forecasting:**
- LSTM out-of-sample RMSE, MAPE, Directional Accuracy, Theil's U
- 1-Year and 2-Year LSTM price forecasts
- GBM Monte Carlo fan chart (500 paths, 5th/20th/35th/65th/80th/95th percentiles)
- GBM vs LSTM comparison overlay

**Stress Testing:**
- Four GBM scenarios: Base Case, Market Crash, Bull Run, High Volatility
- P5 / Median / P95 output table per scenario

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/actuarial-lstm-forecasting.git
cd actuarial-lstm-forecasting

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open in VS Code
code .
# Then open actuarial_lstm_project.ipynb and select .venv as your kernel

# 5. Run all cells top to bottom
```

> **Pylance red underlines?** This means VS Code cannot find the packages in your selected Python interpreter — not a code error. Press `Ctrl+Shift+P` → `Python: Select Interpreter` → choose the `.venv` environment.

---

## Key Limitations (Actuarial Perspective)

Understanding model limitations is as important as building the model. From an actuarial standpoint:

**Volatility clustering** — Empirical return series exhibit ARCH effects (time-varying σ). Rolling volatility charts in Section 11 make this visible. A GARCH(1,1) extension would model σ²_t = ω + αε²_{t-1} + βσ²_{t-1} explicitly — standard in general insurance pricing and capital modelling.

**Fat tails** — Both assets show excess kurtosis > 0 and negative skewness (verified via Jarque-Bera). GBM's log-normal assumption therefore understates tail risk. A Student-t or jump-diffusion (Merton) process would be more actuarially rigorous.

**Forecast horizon accumulation** — Recursive multi-step prediction compounds error with each step, analogous to the widening of actuarial reserve uncertainty over a run-off period.

**In-sample training risk** — The LSTM is trained on data drawn from the same distribution it is evaluated on. Real out-of-sample performance, particularly under market regime changes, will typically be lower.

---

## Possible Extensions

| Extension | Technique | Actuarial Application |
|---|---|---|
| Time-varying volatility | GARCH(1,1) | General insurance pricing, reserving |
| Fat-tail modelling | Student-t GBM / Merton jump-diffusion | Catastrophe risk, tail capital |
| Multivariate LSTM | Full Σ matrix as additional features | Portfolio risk, solvency capital |
| Extreme value theory | Generalised Pareto Distribution (GPD) | Reinsurance pricing, ICS 2.0 |
| Uncertainty quantification | Monte Carlo Dropout / Conformal Prediction | Stochastic liability modelling |

---

## Skills Demonstrated

- Stochastic process modelling (GBM, Brownian motion, Itô calculus)
- Actuarial risk measures (VaR, CVaR, Maximum Drawdown, Calmar Ratio)
- Portfolio covariance analysis (Σ matrix, Efficient Frontier, MRC decomposition)
- Formal statistical testing (ADF, Jarque-Bera, Q-Q analysis)
- Deep learning implementation (LSTM, BatchNorm, EarlyStopping, Huber loss)
- Monte Carlo simulation (500-path fan charts, percentile uncertainty quantification)
- Stress testing and scenario analysis (Solvency II ORSA-style)
- Critical model evaluation (Theil's U, directional accuracy, trading strategy Sharpe)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
