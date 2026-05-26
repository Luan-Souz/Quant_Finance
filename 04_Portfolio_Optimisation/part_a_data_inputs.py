####################     Northeastern University                                    ####################
####################     FINA 6339: Quantitative Portfolio Management               ####################
####################     Portfolio Optimisation — Data & Optimisation Inputs        ####################
####################     Prof. Dr. Milivoje Davidovic                               ####################
####################     Student: Luan Ferreira de Souza                            ####################

"""
Builds and characterises a six-asset portfolio universe:
  - ETH-USD, BTC-USD  (cryptocurrencies)
  - SPY, IEO          (US-listed ETFs)
  - ABEV, PBR         (Brazilian ADRs)

Outputs:
  - Aligned multi-index DataFrame of closing prices and log returns
  - Full descriptive statistics per asset
  - Sample covariance and correlation matrices
  - Annualised expected returns and volatility (×252 convention)
  - Optimisation inputs: μ̂, Σ̂, rf
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

ASSETS     = ["ETH-USD", "BTC-USD", "SPY", "IEO", "ABEV", "PBR"]
START_DATE = "2023-01-01"
RF_ANNUAL  = 0.0359          # 3-month US T-bill rate (constant proxy)
ANNUALISE  = 252             # trading days per year

# ─────────────────────────────────────────────────────────────────────────────
# DATA DOWNLOAD & CLEANING
# ─────────────────────────────────────────────────────────────────────────────

print("Downloading data …")
raw_data = yf.download(ASSETS, start=START_DATE, interval="1d", keepna=False)

# Build multi-index DataFrame: (Close | Log Returns) × Ticker
price_feat = ["Close", "Log Returns"]
cols = pd.MultiIndex.from_arrays(
    [
        [price_feat[0]] * len(ASSETS) + [price_feat[1]] * len(ASSETS),
        ASSETS * len(price_feat),
    ],
    names=["Price Feature", "Ticker"],
)

close   = raw_data["Close"][ASSETS]
log_ret = np.log(close / close.shift(1))

data = pd.concat([close, log_ret], axis=1).dropna()
data.columns = cols

print(f"Clean dataset: {len(data)} daily observations")
print("Log returns are used — time-additive and approximately normally distributed.\n")

# Save to Excel (uncomment on first run)
# data.to_excel("6_assets_quant_portfolio_simulations.xlsx")

returns = data["Log Returns"]   # shape: (T, 6)

# ─────────────────────────────────────────────────────────────────────────────
# DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

ret_mean     = returns.mean()      * 100
ret_median   = returns.median()    * 100
ret_variance = returns.var(ddof=1)
ret_std      = returns.std(ddof=1) * 100
ret_skew     = pd.Series(skew(returns),     index=ASSETS)
ret_kurt     = pd.Series(kurtosis(returns), index=ASSETS)
ret_min      = returns.min()       * 100
ret_max      = returns.max()       * 100

stats_df = pd.DataFrame({
    "Asset":           ASSETS,
    "Mean (%)":        ret_mean.values,
    "Median (%)":      ret_median.values,
    "Variance":        ret_variance.values,
    "Std Dev (%)":     ret_std.values,
    "Skewness":        ret_skew.values,
    "Exc. Kurtosis":   ret_kurt.values,
    "Min (%)":         ret_min.values,
    "Max (%)":         ret_max.values,
}).set_index("Asset")

SEP = "=" * 100
print(f"\n{SEP}")
print(" " * 39 + "Descriptive Statistics (daily log returns, %)")
print(SEP)
print(stats_df.round(4).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# COVARIANCE & CORRELATION MATRICES
# ─────────────────────────────────────────────────────────────────────────────

cov_matrix  = returns.cov()
corr_matrix = returns.corr()

print(f"\n{SEP}")
print(" " * 24 + "Covariance Matrix")
print(SEP)
print(cov_matrix.round(6).to_string())

print(f"\n{SEP}")
print(" " * 24 + "Correlation Matrix")
print(SEP)
print(corr_matrix.round(4).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# ANNUALISED EXPECTED RETURNS & VOLATILITY
# ─────────────────────────────────────────────────────────────────────────────

ann_ret = ret_mean * ANNUALISE
ann_vol = ret_std  * np.sqrt(ANNUALISE)

ann_df = pd.DataFrame({
    "Asset":               ASSETS,
    "Ann. Return (%)":     ann_ret.values,
    "Ann. Volatility (%)": ann_vol.values,
}).set_index("Asset")

print(f"\n{SEP}")
print(" " * 8 + "Annualised Expected Returns and Volatility (×252 / ×√252)")
print(SEP)
print(ann_df.round(4).to_string())

print(f"""
Key observations:
  - BTC-USD delivers the highest annualised return ({ann_ret['BTC-USD']:.2f}%)
    but at 43.5% volatility; ETH-USD has the highest volatility ({ann_vol['ETH-USD']:.2f}%)
    with only half BTC's return — the weakest risk-adjusted asset.
  - SPY is the lowest-volatility anchor ({ann_vol['SPY']:.2f}% annualised).
  - ETH-BTC correlation: {corr_matrix.loc['ETH-USD','BTC-USD']:.2f} → tight crypto cluster.
  - ABEV/PBR correlation with cryptos: {corr_matrix.loc['ABEV','ETH-USD']:.2f}–{corr_matrix.loc['PBR','BTC-USD']:.2f} → strong diversifiers.
  - All assets leptokurtic; SPY shows extreme excess kurtosis ({ret_kurt['SPY']:.2f}).
""")

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMISATION INPUTS
# ─────────────────────────────────────────────────────────────────────────────

mu_hat    = ann_ret                     # annualised expected returns (%)
sigma_hat = cov_matrix * ANNUALISE      # annualised covariance matrix
rf        = RF_ANNUAL                   # annualised risk-free rate (decimal)
rf_daily  = (1 + RF_ANNUAL) ** (1 / ANNUALISE) - 1

print(f"\n{SEP}")
print(" " * 35 + "Optimisation Inputs")
print(SEP)
print(f"  μ̂  (mu_hat)    = annualised sample mean log returns, vector length {len(ASSETS)}")
print(f"  Σ̂  (sigma_hat) = sample covariance × 252, matrix {len(ASSETS)}×{len(ASSETS)}")
print(f"  rf             = {RF_ANNUAL*100:.2f}% annual  ({rf_daily*100:.5f}% daily)")
print(f"  Source         = 3-month US T-bill rate, constant over sample")

print(f"""
Estimation risk discussion:
  Sample means have large standard errors — small perturbations in μ̂ can
  produce extreme, unstable weight allocations ("error maximisation").
  The covariance matrix converges faster and is estimated more reliably.

  Sensitivity hierarchy across downstream methods:
    Max-Return      — fully driven by μ̂ (highest risk)
    Max-Sharpe      — sensitive to both μ̂ and Σ̂
    Black-Litterman — shrinks μ̂ toward EWMA prior (moderate risk)
    Min-Variance    — ignores μ̂, uses only Σ̂ (robust)
    Risk-Parity     — ignores μ̂, uses only Σ̂ (robust)
""")

# Export clean returns for the optimisation scripts
returns.to_csv("clean_returns.csv")
print("Exported: clean_returns.csv")
