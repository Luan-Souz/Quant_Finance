####################     Northeastern University                                    ####################
####################     FINA 6339: Quantitative Portfolio Management               ####################
####################     Descriptive Analysis of Market Index Returns               ####################
####################     Prof. Dr. Milivoje Davidovic                               ####################
####################     Student: Luan Ferreira de Souza                            ####################

"""
Profiles the full statistical behaviour of two equity indices across 500
trading days:
  - Descriptive statistics (mean, median, variance, std, skewness, kurtosis)
  - Correlation between the two return series
  - 20-day moving average and a sign-based momentum strategy
  - Empirical return distributions (histogram, KDE, normal Q-Q plot)

Data: daily closing prices loaded from HM1_QPM_database.xlsx (confidential).
      Indices are on sheets 3 and 4; stocks used in Part C are on sheets 5 and 6.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis, skew

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = "HM1_QPM_database.xlsx"   # update path as needed
MA_WINDOW  = 20                        # moving-average window (trading days)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

mkt_id1 = pd.read_excel(DATA_PATH, sheet_name=3)
mkt_id2 = pd.read_excel(DATA_PATH, sheet_name=4)

data_cols = mkt_id1.columns.tolist()
close_col = data_cols[1]              # column containing closing prices

id1 = mkt_id1[close_col].values
id2 = mkt_id2[close_col].values

# ─────────────────────────────────────────────────────────────────────────────
# PART A1 — DESCRIPTIVE STATISTICS & CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

# Simple daily returns
ret1 = (id1[1:] / id1[:-1]) - 1
ret2 = (id2[1:] / id2[:-1]) - 1

def descriptive_stats(ret, label):
    """Print descriptive statistics for a return series."""
    print("=" * 50)
    print(f" {label} STATISTICS")
    print("=" * 50)
    print(f" Mean:               {ret.mean() * 100:.3f}%")
    print(f" Median:             {np.median(ret) * 100:.3f}%")
    print(f" Minimum:            {ret.min() * 100:.2f}%")
    print(f" Maximum:            {ret.max() * 100:.2f}%")
    print(f" Standard Deviation: {ret.std():.4f}")
    print(f" Variance:           {ret.var(ddof=1):.4f}")
    print(f" Skewness:           {skew(ret):.2f}")
    print(f" Excess Kurtosis:    {kurtosis(ret):.2f}")
    print()

descriptive_stats(ret1, "INDEX 1")
descriptive_stats(ret2, "INDEX 2")

corr = np.corrcoef(ret1, ret2)[0, 1]
print(f" Correlation between Index 1 and Index 2: {corr:.2f}")
print()
print("""
INTERPRETATION:
  Central tendency  — Index 2 averaged ~double the daily return of Index 1.
  Dispersion        — Index 2 carries higher daily volatility.
  Tail behaviour    — Index 2 shows extreme excess kurtosis (18.62), making
                      normality-based risk models unreliable for that series.
  Co-movement       — Correlation of 0.27 offers substantial diversification.
""")

# ─────────────────────────────────────────────────────────────────────────────
# PART A2 — LOG RETURNS, MOVING AVERAGES & MOMENTUM STRATEGY
# ─────────────────────────────────────────────────────────────────────────────

id1_s = pd.Series(mkt_id1[close_col].values)
id2_s = pd.Series(mkt_id2[close_col].values)

ret1_log = np.log(id1_s / id1_s.shift(1)).dropna()
ret2_log = np.log(id2_s / id2_s.shift(1)).dropna()

ma1 = ret1_log.rolling(window=MA_WINDOW).mean()
ma2 = ret2_log.rolling(window=MA_WINDOW).mean()

# ── Plot 1: Index 1 log returns + MA ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ret1_log.values, alpha=0.7, color="steelblue", linewidth=1,
        label="Daily Log Returns")
ax.plot(ma1.values, color="darkred", linewidth=1.2, linestyle="--",
        label=f"{MA_WINDOW}-Day Moving Average")
ax.axhline(y=0, color="black", linewidth=0.7, linestyle="--")
ax.set_title("Index 1 — Daily Log Returns and 20-Day Moving Average")
ax.set_xlabel("Days")
ax.set_ylabel("Log Return")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("index1_log_returns_ma.png", dpi=150)
plt.show()

# ── Plot 2: Index 2 log returns + MA ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ret2_log.values, alpha=0.7, color="steelblue", linewidth=1,
        label="Daily Log Returns")
ax.plot(ma2.values, color="darkblue", linewidth=1.2, linestyle="--",
        label=f"{MA_WINDOW}-Day Moving Average")
ax.axhline(y=0, color="black", linewidth=0.7, linestyle="--")
ax.set_title("Index 2 — Daily Log Returns and 20-Day Moving Average")
ax.set_xlabel("Days")
ax.set_ylabel("Log Return")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("index2_log_returns_ma.png", dpi=150)
plt.show()

# ── Plot 3: Overlay of 20-day MAs ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ma1.values, color="darkred", linewidth=1.2,
        label="Index 1 — 20-Day MA")
ax.plot(ma2.values, color="darkblue", linewidth=1.2,
        label="Index 2 — 20-Day MA")
ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
ax.set_title("Comparison of 20-Day Moving Averages — Index 1 vs Index 2")
ax.set_xlabel("Days")
ax.set_ylabel("20-Day MA of Log Returns")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("ma_overlay.png", dpi=150)
plt.show()

# ── Trend-following strategy ──────────────────────────────────────────────────
print("""
TREND-FOLLOWING STRATEGY: MA Sign-Based Momentum
─────────────────────────────────────────────────
Decision rule:
  BUY  — when the 20-day MA crosses above zero (average returns turned positive)
  SELL — when the 20-day MA crosses below zero (average returns turned negative)
  HOLD — while the MA remains on the same side of zero
""")

signal1 = (ma1 > 0).astype(int)
signal2 = (ma2 > 0).astype(int)

# Lag signal by 1 day to avoid look-ahead bias
strat_ret1 = signal1.shift(1) * ret1_log
strat_ret2 = signal2.shift(1) * ret2_log

bh_cum1   = ret1_log.cumsum()
bh_cum2   = ret2_log.cumsum()
strat_cum1 = strat_ret1.cumsum()
strat_cum2 = strat_ret2.cumsum()

# ── Plot 4: Strategy vs Buy & Hold ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(bh_cum1.values, label="Buy & Hold", color="steelblue")
axes[0].plot(strat_cum1.values, label="MA Strategy", color="darkred")
axes[0].set_title("Index 1 — Strategy vs Buy & Hold")
axes[0].set_xlabel("Days")
axes[0].set_ylabel("Cumulative Log Return")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(bh_cum2.values, label="Buy & Hold", color="darkorange")
axes[1].plot(strat_cum2.values, label="MA Strategy", color="darkgreen")
axes[1].set_title("Index 2 — Strategy vs Buy & Hold")
axes[1].set_xlabel("Days")
axes[1].set_ylabel("Cumulative Log Return")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("strategy_vs_buyhold.png", dpi=150)
plt.show()

print("=" * 60)
print(" STRATEGY PERFORMANCE SUMMARY")
print("=" * 60)
n_trades1 = int((signal1.diff().abs() > 0).sum())
n_trades2 = int((signal2.diff().abs() > 0).sum())
print(f"\n Index 1:")
print(f"   Buy & Hold cumulative return:  {bh_cum1.iloc[-1]:.4f}")
print(f"   MA Strategy cumulative return: {strat_cum1.iloc[-1]:.4f}")
print(f"   Number of trades:              {n_trades1}")
print(f"\n Index 2:")
print(f"   Buy & Hold cumulative return:  {bh_cum2.iloc[-1]:.4f}")
print(f"   MA Strategy cumulative return: {strat_cum2.iloc[-1]:.4f}")
print(f"   Number of trades:              {n_trades2}")

# ─────────────────────────────────────────────────────────────────────────────
# PART A3 — RETURN DISTRIBUTIONS (Histogram, KDE, Q-Q Plot)
# ─────────────────────────────────────────────────────────────────────────────

ret1_sim = id1_s.pct_change().dropna()
ret2_sim = id2_s.pct_change().dropna()
ret1_log_s = np.log(id1_s / id1_s.shift(1)).dropna()
ret2_log_s = np.log(id2_s / id2_s.shift(1)).dropna()

def distribution_plots(simple_ret, log_ret, label, color="steelblue"):
    """
    Plot histogram, KDE, and normal Q-Q for both simple and log returns
    of a single index.
    """
    for ret, rtype in [(simple_ret, "Simple"), (log_ret, "Log")]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        x = np.linspace(ret.min(), ret.max(), 300)

        # (i) Histogram
        axes[0].hist(ret, bins=50, density=True, alpha=0.7, color=color,
                     edgecolor="white", label=f"{rtype} Returns")
        axes[0].plot(x, stats.norm.pdf(x, ret.mean(), ret.std()),
                     "r-", linewidth=2, label="Normal Fit")
        axes[0].set_title(f"{label} — {rtype}: Histogram")
        axes[0].set_xlabel("Return")
        axes[0].set_ylabel("Density")
        axes[0].legend()

        # (ii) KDE
        ret.plot.kde(ax=axes[1], color=color, linewidth=2,
                     label=f"{rtype} KDE")
        axes[1].plot(x, stats.norm.pdf(x, ret.mean(), ret.std()),
                     "r--", linewidth=2, label="Normal Fit")
        axes[1].set_title(f"{label} — {rtype}: KDE")
        axes[1].set_xlabel("Return")
        axes[1].set_ylabel("Density")
        axes[1].legend()

        # (iii) Q-Q Plot
        stats.probplot(ret, dist="norm", plot=axes[2])
        axes[2].set_title(f"{label} — {rtype}: Normal Q-Q Plot")
        axes[2].get_lines()[0].set(markerfacecolor=color, markersize=3)

        plt.suptitle(f"{label} — {rtype} Returns Distribution Analysis",
                     fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        fname = f"{label.lower().replace(' ', '_')}_{rtype.lower()}_dist.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.show()

distribution_plots(ret1_sim, ret1_log_s, "Index 1", color="steelblue")
distribution_plots(ret2_sim, ret2_log_s, "Index 2", color="darkorange")
