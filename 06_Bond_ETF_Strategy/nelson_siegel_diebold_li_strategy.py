####################     Northeastern University                                                    ####################
####################     FINA 6339: Quantitative Portfolio Management                               ####################
####################     Research Project — Dynamic Bond ETF Allocation                             ####################
####################     Nelson-Siegel / Diebold-Li VAR Bond ETF Strategy                          ####################
####################     Prof. Dr. Milivoje Davidovic                                               ####################
####################     Miles Choquette · Luan Ferreira de Souza · Luka Gabadadze                 ####################
####################     Jingying Gao · Zhe Chen                                                   ####################

"""
Systematic fixed income allocation strategy across seven U.S. Treasury ETFs
combining the Nelson-Siegel yield curve model with the Diebold-Li dynamic
factor extension.

Pipeline (13 steps):
  1.  Setup              — constants, ETF universe, decay parameter λ
  2.  Data               — FRED par yields + yfinance ETF prices, month-end alignment
  3.  Descriptive        — yield/ETF summary stats, correlation matrices, time-series plots
  4.  Nelson-Siegel      — cross-sectional OLS each month, condition number diagnostic
  5.  Diebold-Li VAR(1)  — expanding-window VAR, one-month-ahead factor forecasts
  6.  Helper Functions   — factor_loading, zscore, weighting schemes, backtest engine,
                           performance metrics, Jobson-Korkie test
  7.  Signal Decomp.     — NS mispricing-only vs DL directional-only in isolation
  8.  Factor Attribution — β₀ / β₁ / β₂ standalone Sharpe attribution
  9.  Signal Construction— 60/20/20 β₁-tilted directional + 50/50 composite signal
  10. Backtest           — lagged weights, monthly rebalancing
  11. Performance        — CAGR, Sharpe, Sortino, max drawdown, Jobson-Korkie z, TE
  12. Results            — full-period cumulative return chart (2009–2024)
  13. Sub-Period         — 2018–2021: 2019 rate-cut cycle + 2020 COVID shock

Nelson-Siegel model:
  y(τ) = β₀ + β₁·[(1−e^(−λτ))/(λτ)] + β₂·[(1−e^(−λτ))/(λτ) − e^(−λτ)]

Diebold-Li VAR(1):
  F_t = c + Φ·F_{t−1} + u_t   →   F̂_{t+1} = ĉ_t + Φ̂_t·F_t

Data sources:
  Treasury par yields — FRED (DGS2, DGS3, DGS5, DGS7, DGS10, DGS20, DGS30), daily CSVs
  ETF prices          — Yahoo Finance (adjusted close), month-end

NOTE: Update the FRED CSV file path in Section 2 before running.

References:
  Nelson, C.R. & Siegel, A.F. (1987). Parsimonious modeling of yield curves.
    Journal of Business, 60(4), 473–489.
  Diebold, F.X. & Li, C. (2006). Forecasting the term structure of government
    bond yields. Journal of Econometrics, 130(2), 337–364.
  Annaert, J. et al. (2010). Estimating the yield curve using the Nelson-Siegel
    model: A ridge regression approach. Working Paper, Universiteit Antwerpen.
  Jobson, J.D. & Korkie, B.M. (1981). Performance hypothesis testing with the
    Sharpe and Treynor measures. Journal of Finance, 36(4), 889–908.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.vector_ar.var_model import VAR
import warnings
import os
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. SETUP
# ─────────────────────────────────────────────────────────────────────────────

# Date range
START  = '2008-01-01'
END    = '2024-12-31'

# Nelson-Siegel
LAMBDA = 0.0609          # Fixed decay; matches ~30-month hump in NS curve

# Treasury yield maturities from FRED (years)
MATURITIES = [2, 3, 5, 7, 10, 20, 30]

# ETF universe & approximate effective durations (years)
ETFS = ['SHV', 'SHY', 'IEI', 'IEF', 'TLH', 'TLT', 'EDV']
ETF_MATS = {
    'SHV': 0.5,   # <1Y
    'SHY': 2.0,
    'IEI': 4.5,
    'IEF': 8.5,
    'TLH': 15.0,
    'TLT': 20.0,
    'EDV': 25.0,
}

# Backtest & weighting
TOP_N          = 3       # ETFs to hold in top-N strategy
VAR_INIT_MONTHS = 36    # Expanding-window burn-in for VAR(1)
FACTOR_WEIGHTS = {'b0': 0.20, 'b1': 0.60, 'b2': 0.20}  # β₁-tilt

# Sub-period analysis window
SUB_START = '2018-01-01'
SUB_END   = '2021-12-31'

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA
# Treasury par yields from FRED (daily CSVs, resampled to month-end),
# ETF prices via yfinance. Alignment drops any months where either series
# is missing.
# ─────────────────────────────────────────────────────────────────────────────

frames = []
for mat in MATURITIES:
    
    #CHANGE THE FILE PATH
    
    df = pd.read_csv(f'/Users/luanferreiradesouza/QPM/Project/DGS{mat}.csv', parse_dates=['observation_date'], index_col='observation_date')
    df.columns = [mat]
    df = df.replace('.', np.nan).astype(float)
    frames.append(df)

yields = pd.concat(frames, axis=1).dropna()
yields = yields / 100.0
yields = yields.resample('ME').last().dropna()

prices_raw = yf.download(ETFS, start=START, end=END, auto_adjust=True, progress=False)['Close']
prices = prices_raw.resample('ME').last()

# Drop tickers with less than 80% coverage (EDV launched 2007, so it should survive)
prices = prices.dropna(axis=1, thresh=int(0.8 * len(prices)))
ETFS     = [e for e in ETFS if e in prices.columns]
ETF_MATS = {k: v for k, v in ETF_MATS.items() if k in ETFS}
N_ETFS   = len(ETFS)

common = yields.index.intersection(prices.index)
yields = yields.loc[common]
prices = prices.loc[common]

print(f'ETFs loaded: {ETFS}')
print(f'Aligned months: {len(common)}  ({common[0].date()} to {common[-1].date()})')
yields.tail(3)

# ─────────────────────────────────────────────────────────────────────────────
# 3. DESCRIPTIVE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

# Treasury Yield Summary Statistics

print("=== Treasury Yield Summary Statistics (annualized, in %) ===\n")
yield_stats = (yields * 100).describe().T
yield_stats['skew'] = (yields * 100).skew()
yield_stats['kurtosis'] = (yields * 100).kurtosis()
yield_stats.index.name = 'Maturity (yr)'
print(yield_stats.round(3))

# ETF Return Summary Statistics

returns_all = prices.pct_change().dropna()
ann_ret   = returns_all.mean() * 12
ann_vol   = returns_all.std() * np.sqrt(12)
ann_sr    = ann_ret / ann_vol
skew_ret  = returns_all.skew()
kurt_ret  = returns_all.kurtosis()
max_dd    = ((1 + returns_all).cumprod().div(
              (1 + returns_all).cumprod().cummax()) - 1).min()

etf_stats = pd.DataFrame({
    'Ann. Return (%)':  (ann_ret * 100).round(2),
    'Ann. Vol (%)':     (ann_vol * 100).round(2),
    'Sharpe':            ann_sr.round(3),
    'Skewness':          skew_ret.round(3),
    'Kurtosis':          kurt_ret.round(3),
    'Max Drawdown (%)': (max_dd * 100).round(2),
})
print("\n=== ETF Monthly Return Summary Statistics ===\n")
print(etf_stats)

# Yield Curve Time Series

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Panel A: selected yield maturities
for mat in [2, 5, 10, 30]:
    axes[0].plot(yields.index, yields[mat] * 100, label=f'{mat}Y', linewidth=1.2)
axes[0].set_ylabel('Yield (%)')
axes[0].set_title('Panel A: Treasury Par Yields Over Time')
axes[0].legend(ncol=4, fontsize=9)
axes[0].grid(alpha=0.3)

# Panel B: term spread (10Y − 2Y)
spread = (yields[10] - yields[2]) * 100
axes[1].plot(spread.index, spread, color='#E91E63', linewidth=1.2)
axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[1].fill_between(spread.index, spread, 0, where=spread < 0,
                     color='#E91E63', alpha=0.15, label='Inverted')
axes[1].set_ylabel('Spread (bps)')
axes[1].set_title('Panel B: 10Y − 2Y Term Spread')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('term_spread.png', dpi=300, bbox_inches='tight')
plt.show()

# ETF Cumulative Returns

fig, ax = plt.subplots(figsize=(12, 5))
cum = (1 + returns_all[ETFS]).cumprod()
for etf in ETFS:
    ax.plot(cum.index, cum[etf], label=etf, linewidth=1.2)
ax.set_title('Cumulative Growth of $1 — Bond ETF Universe')
ax.set_ylabel('Growth of $1')
ax.legend(ncol=4, fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ETF Return Distributions

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
axes = axes.flatten()
for i, etf in enumerate(ETFS):
    ax = axes[i]
    ax.hist(returns_all[etf].dropna(), bins=40, edgecolor='white',
            color='#2196F3', alpha=0.75, density=True)
    mu, sigma = returns_all[etf].mean(), returns_all[etf].std()
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=1.2)
    ax.set_title(etf, fontsize=10)
    ax.tick_params(labelsize=8)
for j in range(len(ETFS), len(axes)):
    axes[j].set_visible(False)
fig.suptitle('Monthly Return Distributions vs. Normal Fit', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('monthly_ret_dist.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation Matrix ETF Returns

corr = returns_all[ETFS].corr()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(ETFS)))
ax.set_yticks(range(len(ETFS)))
ax.set_xticklabels(ETFS, fontsize=9, rotation=45, ha='right')
ax.set_yticklabels(ETFS, fontsize=9)
for i in range(len(ETFS)):
    for j in range(len(ETFS)):
        ax.text(j, i, f'{corr.values[i, j]:.2f}', ha='center', va='center', fontsize=8,
                color='white' if abs(corr.values[i, j]) > 0.6 else 'black')
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title('ETF Return Correlation Matrix')
plt.tight_layout()
plt.show()

# Correlation Matrix monthly yield changes

corr_y = np.log(yields.shift(1)/yields).dropna().corr()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr_y.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(MATURITIES)))
ax.set_yticks(range(len(MATURITIES)))
ax.set_xticklabels(MATURITIES, fontsize=9, rotation=45, ha='right')
ax.set_yticklabels(MATURITIES, fontsize=9)
for i in range(len(MATURITIES)):
    for j in range(len(MATURITIES)):
        ax.text(j, i, f'{corr_y.values[i, j]:.2f}', ha='center', va='center', fontsize=8,
                color='white' if abs(corr_y.values[i, j]) > 0.6 else 'black')
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title('Monthly Yield changes Correlation Matrix')
plt.tight_layout()
plt.savefig('yield_changes.png', dpi=300, bbox_inches='tight')
plt.show()

# Rolling Volatility (12-month)

fig, ax = plt.subplots(figsize=(12, 4))
for etf in ['SHY', 'IEF', 'TLT', 'EDV']:
    roll_vol = returns_all[etf].rolling(12).std() * np.sqrt(12) * 100
    ax.plot(roll_vol.index, roll_vol, label=etf, linewidth=1.2)
ax.set_title('12-Month Rolling Annualized Volatility (Selected ETFs)')
ax.set_ylabel('Volatility (%)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('volatility.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation Matrices side by side
corr = returns_all[ETFS].corr()
corr_y = np.log(yields.shift(1)/yields).dropna().corr()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, matrix, labels, title in zip(
    axes,
    [corr, corr_y],
    [ETFS, MATURITIES],
    ['ETF Return Correlation Matrix', 'Monthly Yield Changes Correlation Matrix']
):
    im = ax.imshow(matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha='right')
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{matrix.values[i, j]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(matrix.values[i, j]) > 0.6 else 'black')
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)

plt.tight_layout()
plt.savefig('correlation_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 4. NELSON-SIEGEL CURVE FITTING
# Diebold & Li (2006): fit β₀, β₁, β₂ each month via cross-sectional OLS
# with fixed λ. Multicollinearity checked via condition number diagnostic
# (Annaert et al., 2010). Residuals stored as the mispricing signal.
# ─────────────────────────────────────────────────────────────────────────────

# Nelson-Siegel OLS estimation with fixed λ (Diebold & Li, 2006)
def ns_yield(tau, b0, b1, b2):
    lt = LAMBDA * tau
    l1 = (1 - np.exp(-lt)) / lt
    l2 = l1 - np.exp(-lt)
    return b0 + b1 * l1 + b2 * l2

def ns_loadings(tau, lam):
    """Construct the NS regressor matrix for given maturities and fixed λ."""
    lt = lam * tau
    l1 = (1 - np.exp(-lt)) / lt
    l2 = l1 - np.exp(-lt)
    return np.column_stack([np.ones_like(tau), l1, l2])

mats = np.array(MATURITIES, dtype=float)
X = ns_loadings(mats, LAMBDA)

# Condition number diagnostic (Annaert et al., 2010)
# Check slope & curvature regressors only (columns 1 and 2)
kappa = np.linalg.cond(X[:, 1:])
print(f'Condition number of slope/curvature regressors: {kappa:.2f}')
print(f'Multicollinearity concern: {"Yes (consider ridge)" if kappa >= 10 else "No — OLS appropriate"} (threshold = 10)\n')

# Cross-sectional OLS each month: y = X @ beta
betas, residuals = [], []
for date, row in yields.iterrows():
    y_obs = row.values
    b, _, _, _ = np.linalg.lstsq(X, y_obs, rcond=None)
    fitted = X @ b
    betas.append({'date': date, 'b0': b[0], 'b1': b[1], 'b2': b[2]})
    residuals.append(dict(zip(mats, y_obs - fitted)))

factors_df   = pd.DataFrame(betas).set_index('date')
residuals_df = pd.DataFrame(residuals, index=yields.index)

print(f'Factors shape: {factors_df.shape}')
factors_df.tail(3)

# ─────────────────────────────────────────────────────────────────────────────
# 5. DIEBOLD-LI VAR(1) FORECASTS
# Treat three NS factors as a system and model joint dynamics with VAR(1).
# Expanding window — model refit on all data up to t, then forecast t+1.
# First VAR_INIT_MONTHS months initialize the window.
# ─────────────────────────────────────────────────────────────────────────────

forecasts = pd.DataFrame(index=factors_df.index, columns=['b0', 'b1', 'b2'], dtype=float)

for i in range(36, len(factors_df)):
    train = factors_df.iloc[:i]
    try:
        fcast = VAR(train).fit(1).forecast(train.values[-1:], steps=1)[0]
        forecasts.iloc[i] = fcast
    except Exception:
        pass  # skip months where VAR fails to converge (rare)

print(f'Forecast coverage: {forecasts.dropna().shape[0]} months')
forecasts.dropna().tail(3)

# ─────────────────────────────────────────────────────────────────────────────
# 6. HELPER FUNCTIONS
# factor_loading, zscore, weighting schemes, run_backtest, metrics
# ─────────────────────────────────────────────────────────────────────────────

# Factor loading (NS basis function value at a given maturity)
def factor_loading(factor, mat):
    lt = LAMBDA * mat
    l1 = (1 - np.exp(-lt)) / lt
    l2 = l1 - np.exp(-lt)
    if factor == 'b0': return 1.0
    if factor == 'b1': return l1
    if factor == 'b2': return l2

# Cross-sectional z-score
def zscore(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0)

# Weighting schemes
fallback = pd.Series(1 / len(ETFS), index=ETFS)

def signal_prop_weights(x):
    shifted = x - x.min()
    if shifted.sum() == 0:
        return pd.Series(1 / len(ETFS), index=x.index)
    return shifted / shifted.sum()

def top_n_weights(x, n=TOP_N):
    w = pd.Series(0.0, index=x.index)
    w[x.nlargest(n).index] = 1.0 / n
    return w

# Backtest engine
returns = prices.pct_change().dropna()

def run_backtest(weights):
    w_shifted = weights.shift(1).dropna()
    idx = w_shifted.index.intersection(returns.index)
    return (w_shifted.loc[idx] * returns.loc[idx, ETFS]).sum(axis=1)

# Performance metrics + Jobson-Korkie test
def metrics(r, label):
    cagr    = (1 + r).prod() ** (12 / len(r)) - 1
    vol     = r.std() * np.sqrt(12)
    sharpe  = (r.mean()*12) / (r.std() * np.sqrt(12))
    down    = r[r < 0].std() * np.sqrt(12)
    sortino = cagr / down if down > 0 else np.nan
    cum     = (1 + r).cumprod()
    max_dd  = ((cum - cum.cummax()) / cum.cummax()).min()
    print(f'{label:<30} CAGR: {cagr*100:5.2f}%  Vol: {vol*100:5.2f}%  '
          f'Sharpe: {sharpe:.3f}  Sortino: {sortino:.3f}  MaxDD: {max_dd*100:.2f}%')

def jk_test(r_s, r_b):
    n    = len(r_s)
    sr_s = r_s.mean() / r_s.std()
    sr_b = r_b.mean() / r_b.std()
    corr = np.corrcoef(r_s, r_b)[0, 1]
    se   = np.sqrt((1/n) * (2 - 2*corr + 0.5*(sr_s**2 + sr_b**2 - 2*sr_s*sr_b*corr**2)))
    z    = (sr_s - sr_b) / se
    p    = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p

# ─────────────────────────────────────────────────────────────────────────────
# 7. SIGNAL DECOMPOSITION: NELSON-SIEGEL vs DIEBOLD-LI
# Run each signal in isolation before combining. NS-only uses mispricing
# residuals; DL-only uses VAR(1) directional forecast. Establishes which
# component adds more value.
# ─────────────────────────────────────────────────────────────────────────────

# NS-only: mispricing residuals, no forecast
mis = pd.DataFrame({
    etf: residuals_df.apply(
        lambda r: float(np.interp(mat, residuals_df.columns.astype(float), r.values)), axis=1)
    for etf, mat in ETF_MATS.items()
})
weights_ns_only = zscore(mis).apply(
    lambda r: signal_prop_weights(r) if not r.isna().any() else fallback, axis=1)
strat_ns_only = run_backtest(weights_ns_only)

# DL-only: VAR(1) directional forecast, no mispricing
current_yield = pd.DataFrame({
    etf: factors_df.apply(
        lambda r: ns_yield(np.array([mat]), r['b0'], r['b1'], r['b2'])[0], axis=1)
    for etf, mat in ETF_MATS.items()
})
forecast_yield = pd.DataFrame({
    etf: forecasts.apply(
        lambda r: ns_yield(np.array([mat]), r['b0'], r['b1'], r['b2'])[0]
        if not r.isna().any() else np.nan, axis=1)
    for etf, mat in ETF_MATS.items()
})
dir_ = -(forecast_yield - current_yield)
weights_dl_only = zscore(dir_).apply(
    lambda r: signal_prop_weights(r) if not r.isna().any() else fallback, axis=1)
strat_dl_only = run_backtest(weights_dl_only)

# Benchmark
bench_ret = returns[ETFS].mean(axis=1).loc[strat_ns_only.index]

print('Signal Decomposition — Full Period')
for r, label in [
    (strat_ns_only, 'Nelson-Siegel only'),
    (strat_dl_only, 'Diebold-Li only'),
    (bench_ret,     'Equal Weight'),
]:
    metrics(r, label)

# ─────────────────────────────────────────────────────────────────────────────
# 8. INDIVIDUAL FACTOR ATTRIBUTION
# Run each Diebold-Li factor (β₀ level, β₁ slope, β₂ curvature) in isolation.
# Slope β₁ consistently dominates — motivates the 60/20/20 tilt in Section 9.
# ─────────────────────────────────────────────────────────────────────────────

factor_signals = {}
for factor in ['b0', 'b1', 'b2']:
    delta = forecasts[factor] - factors_df[factor]
    sig = pd.DataFrame({
        etf: -delta * factor_loading(factor, mat)
        for etf, mat in ETF_MATS.items()
    })
    factor_signals[factor] = sig

weights_b0 = factor_signals['b0'].apply(lambda r: signal_prop_weights(r) if not r.isna().any() else fallback, axis=1)
weights_b1 = factor_signals['b1'].apply(lambda r: signal_prop_weights(r) if not r.isna().any() else fallback, axis=1)
weights_b2 = factor_signals['b2'].apply(lambda r: signal_prop_weights(r) if not r.isna().any() else fallback, axis=1)

strat_b0 = run_backtest(weights_b0)
strat_b1 = run_backtest(weights_b1)
strat_b2 = run_backtest(weights_b2)

print('Individual Factor Signals — Full Period')
for r, label in [
    (strat_b0, 'β₀ Level'),
    (strat_b1, 'β₁ Slope'),
    (strat_b2, 'β₂ Curvature'),
    (bench_ret, 'Equal Weight'),
]:
    metrics(r, label)

fig, ax = plt.subplots(figsize=(12, 5))
for r, label, color in [
    (strat_b0, 'β₀ Level', '#E91E63'),
    (strat_b1, 'β₁ Slope', '#9C27B0'),
    (strat_b2, 'β₂ Curvature', '#00BCD4'),
    (bench_ret, 'Equal Weight', '#9E9E9E'),
]:
    cum = (1 + r).cumprod()
    ls  = '--' if label == 'Equal Weight' else '-'
    ax.plot(cum.index, cum, label=label, linewidth=1.8, linestyle=ls, color=color)

ax.set_title('Individual Factor Signals vs. Equal Weight')
ax.set_ylabel('Growth of $1')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('ind_factor.png', dpi=300, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 9. SIGNAL CONSTRUCTION & PORTFOLIO WEIGHTS
# Directional signal outperforms mispricing alone and β₁ carries the most
# predictive power — final signal applies 60/20/20 factor tilt and combines
# both signals with equal weight (50/50).
# ─────────────────────────────────────────────────────────────────────────────

# Directional signal: β₁-tilted VAR forecast
dir_tilted = pd.DataFrame(index=factors_df.index, columns=ETFS, dtype=float)
for etf, mat in ETF_MATS.items():
    weighted_delta = sum(
        FACTOR_WEIGHTS[f] * (forecasts[f] - factors_df[f]) * factor_loading(f, mat)
        for f in ['b0', 'b1', 'b2']
    )
    dir_tilted[etf] = -weighted_delta

# Combined signal
combined = 0.5 * zscore(mis) + 0.5 * zscore(dir_tilted)

# Portfolio weights
N_ETFS = len(ETFS)
fallback = pd.Series(1 / N_ETFS, index=ETFS)

weights_final = combined.apply(lambda r: signal_prop_weights(r) if not r.isna().any() else fallback, axis=1)
weights_top_n = combined.apply(lambda r: top_n_weights(r)       if not r.isna().any() else fallback, axis=1)

print(f'Average weights — Signal Proportional:')
print(weights_final.mean().round(3))
print(f'\nAverage weights — Top-{TOP_N}:')
print(weights_top_n.mean().round(3))

# ─────────────────────────────────────────────────────────────────────────────
# 10. BACKTEST
# Weights lagged one period to avoid look-ahead bias — signal at t goes into
# the portfolio held during t+1.
# ─────────────────────────────────────────────────────────────────────────────

strat_final = run_backtest(weights_final)
strat_topn  = run_backtest(weights_top_n)

idx = strat_final.index
bench_ret = returns.loc[idx, ETFS].mean(axis=1)

print(f'Backtest period: {idx[0].date()} to {idx[-1].date()}  ({len(idx)} months)')

# ─────────────────────────────────────────────────────────────────────────────
# 11. PERFORMANCE METRICS
# Jobson-Korkie (1981) test for statistical significance of Sharpe ratio
# differences. Tracking error is annualized.
# ─────────────────────────────────────────────────────────────────────────────

print('Full Period Performance')
for r, label in [
    (strat_final, f'Signal Proportional'),
    (strat_topn,  f'Top-{TOP_N}'),
    (bench_ret,   'Equal Weight'),
]:
    metrics(r, label)

print('\nJobson-Korkie vs Equal Weight')
for r, label in [(strat_final, 'Signal Proportional'), (strat_topn, f'Top-{TOP_N}')]:
    z, p = jk_test(r, bench_ret)
    te   = (r - bench_ret).std() * np.sqrt(12) * 100
    print(f'{label:<30} JK z: {z:6.3f}  p: {p:.3f}  TE: {te:.2f}%')

# ─────────────────────────────────────────────────────────────────────────────
# 12. RESULTS — FULL PERIOD (2009–2024)
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))
for r, label, color in [
    (strat_final, 'Signal Proportional', '#2196F3'),
    (strat_topn, f'Top-{TOP_N}', '#FF9800'),
    (bench_ret, 'Equal Weight', '#9E9E9E'),
]:
    cum = (1 + r).cumprod()
    ls  = '--' if label == 'Equal Weight' else '-'
    ax.plot(cum.index, cum, label=label, linewidth=1.8, linestyle=ls, color=color)

ax.set_title(f'NS/DL β₁-Tilted: Signal Proportional vs. Top-{TOP_N} vs. Equal Weight')
ax.set_ylabel('Growth of $1')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('full_period.png', dpi=300, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 13. SUB-PERIOD (2018–2021)
# Captures two distinct regimes: the 2019 Fed rate-cut cycle and the 2020
# COVID shock. Both had directional yield curve moves — the environment where
# a slope-forecasting model adds the most value.
# ─────────────────────────────────────────────────────────────────────────────

def sub(r):
    return r[(r.index >= SUB_START) & (r.index <= SUB_END)]

print(f'── {SUB_START[:4]}–{SUB_END[:4]}')
for r, label in [
    (strat_final, 'Signal Proportional'),
    (strat_topn,  f'Top-{TOP_N}'),
    (bench_ret,   'Equal Weight'),
]:
    metrics(sub(r), label)

fig, ax = plt.subplots(figsize=(12, 5))
for r, label, color in [
    (strat_final, 'Signal Proportional', '#2196F3'),
    (strat_topn,  f'Top-{TOP_N}',        '#FF9800'),
    (bench_ret,   'Equal Weight',        '#9E9E9E'),
]:
    s   = sub(r)
    cum = (1 + s).cumprod()
    cum = cum / cum.iloc[0]
    ls  = '--' if label == 'Equal Weight' else '-'
    ax.plot(cum.index, cum, label=label, linewidth=1.8, linestyle=ls, color=color)

ax.set_title(f'NS/DL β₁-Tilted — {SUB_START[:4]}–{SUB_END[:4]}')
ax.set_ylabel('Growth of $1')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('full_subper.png', dpi=300, bbox_inches='tight')
plt.show()
