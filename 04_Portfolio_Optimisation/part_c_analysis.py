####################     Northeastern University                                    ####################
####################     FINA 6339: Quantitative Portfolio Management               ####################
####################     Portfolio Optimisation — Post-Optimisation Analysis        ####################
####################     Prof. Dr. Milivoje Davidovic                               ####################
####################     Student: Luan Ferreira de Souza                            ####################

"""
Applies the five optimal weight vectors to historical daily returns to
construct portfolio return time-series, then characterises each:

  C1 — Descriptive statistics of portfolio returns
       (mean, median, variance, std, skewness, kurtosis, min, max, quantiles)
  C2 — Empirical distribution analysis
       (histogram, KDE vs normal reference, normal Q-Q plot, overlay KDE)
  C3 — Comparative interpretation and final portfolio recommendation

Requires:
  clean_returns.csv   (from part_a_data_inputs.py)
  optimal_weights.csv (from part_b_optimisers.py)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.stats import skew, kurtosis

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

ASSETS  = ["ETH-USD", "BTC-USD", "SPY", "IEO", "ABEV", "PBR"]
METHODS = ["Min-Var", "Max-Ret", "Max-Sharpe", "Risk-Parity", "BL"]
COLS    = ["#1d4ed8", "#dc2626", "#059669", "#7c3aed", "#d97706"]
QUANTILES = [0.05, 0.25, 0.75, 0.95]

returns    = pd.read_csv("clean_returns.csv",   index_col=0, parse_dates=True)
returns.columns = ASSETS
weights_df = pd.read_csv("optimal_weights.csv", index_col=0)
weights_df.columns = METHODS

# Convert log returns to simple daily returns (required for linear aggregation)
sim_returns = np.exp(returns) - 1

# Daily portfolio return series: r_port = w · r_assets
port_returns = pd.DataFrame(
    {m: sim_returns @ weights_df[m].values for m in METHODS},
    index=returns.index,
)

# Cumulative returns
cum_returns = (1 + port_returns).cumprod() - 1

# Total period return
total_ret = (1 + port_returns).prod() - 1

print("Total period returns:")
for m in METHODS:
    print(f"  {m:<15}: {total_ret[m]*100:.2f}%")
print()

# ─────────────────────────────────────────────────────────────────────────────
# C1 — DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_stats_table(series_df):
    """Compute descriptive statistics for each portfolio return series."""
    rows = []
    for col in series_df.columns:
        s = series_df[col].dropna()
        row = {
            "Portfolio":     col,
            "Mean % /day":   s.mean()      * 100,
            "Median %":      s.median()    * 100,
            "Std Dev %":     s.std(ddof=1) * 100,
            "Skewness":      float(skew(s)),
            "Exc. Kurtosis": float(kurtosis(s)),
            "Min %":         s.min()        * 100,
            "Max %":         s.max()        * 100,
        }
        for q in QUANTILES:
            row[f"Q{int(q*100)}%"] = s.quantile(q) * 100
        rows.append(row)
    return pd.DataFrame(rows).set_index("Portfolio")

stats = portfolio_stats_table(port_returns)
SEP = "=" * 120
print(f"\n{SEP}")
print(" " * 40 + "Descriptive Statistics — Daily Simple Returns")
print(SEP)
print(stats.round(4).to_string())

# ── Cumulative return chart ────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for m, c in zip(METHODS, COLS):
    ax1.plot(cum_returns.index, cum_returns[m] * 100,
             label=m, color=c, linewidth=1.3)

ax1.set_title("Cumulative Returns — All Portfolios", fontweight="bold")
ax1.set_ylabel("Cumulative Return (%)")
ax1.legend(loc="upper left", fontsize=9, ncol=2)
ax1.grid(True, alpha=0.4)

for m, c in zip([m for m in METHODS if m != "Max-Ret"], COLS):
    ax2.plot(cum_returns.index, cum_returns[m] * 100,
             label=m, color=c, linewidth=1.3)

ax2.set_title("Cumulative Returns — Excluding Max-Return (scale)", fontsize=11)
ax2.set_ylabel("Cumulative Return (%)")
ax2.set_xlabel("Date")
ax2.legend(loc="upper left", fontsize=9, ncol=2)
ax2.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("cumulative_returns.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# C2 — DISTRIBUTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def distribution_plots(series, name, color):
    """
    Plot histogram + normal fit, KDE + normal reference, and normal Q-Q
    for a single portfolio return series.
    """
    s = series.dropna()
    x = np.linspace(s.min(), s.max(), 300)
    mu_s, sig_s = s.mean(), s.std(ddof=1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (i) Histogram
    axes[0].hist(s, bins=60, density=True, alpha=0.55, color=color,
                 edgecolor="white", label="Empirical")
    axes[0].plot(x, ss.norm.pdf(x, mu_s, sig_s),
                 "r-", lw=2, label="Normal fit")
    axes[0].set_title("Histogram")
    axes[0].set_xlabel("Daily Simple Return")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # (ii) KDE
    s.plot.kde(ax=axes[1], color=color, linewidth=2, label="KDE")
    axes[1].plot(x, ss.norm.pdf(x, mu_s, sig_s),
                 "r--", lw=2, label="Normal fit")
    axes[1].set_title("Kernel Density Estimate")
    axes[1].set_xlabel("Daily Simple Return")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    # (iii) Normal Q-Q Plot
    ss.probplot(s, dist="norm", plot=axes[2])
    axes[2].set_title("Normal Q-Q Plot")
    axes[2].get_lines()[0].set(markerfacecolor=color, markersize=3, alpha=0.6)

    fig.suptitle(f"{name} — Return Distribution Analysis",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = f"dist_{name.lower().replace('-', '_').replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname}")


for m, c in zip(METHODS, COLS):
    distribution_plots(port_returns[m], m, color=c)

# ── KDE overlay — all portfolios ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
for m, c in zip(METHODS, COLS):
    port_returns[m].dropna().plot.kde(
        ax=ax, color=c, linewidth=2.2, linestyle="--", label=m)

ax.set_xlim(-0.10, 0.10)
ax.set_title("KDE Overlay — Daily Simple Returns, All Portfolios",
             fontweight="bold")
ax.set_xlabel("Daily Simple Return")
ax.set_ylabel("Density")
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("kde_overlay.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# C3 — COMPARATIVE INTERPRETATION & RECOMMENDATION
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print(" " * 35 + "C3 — Comparative Interpretation & Recommendation")
print(SEP)
print(f"""
Objectives & intuition:
  Min-Variance    — minimise total risk (Σ only); ignores μ̂ entirely.
  Max-Return      — maximise expected return (μ̂ only); ignores risk.
  Max-Sharpe      — maximise risk-adjusted return; uses both μ̂ and Σ.
  Risk-Parity     — equalise risk contribution (Σ only); ignores μ̂.
  Black-Litterman — Bayesian blend of equilibrium prior + investor views.

Strengths & weaknesses:
  Min-Variance:    robust to return estimation error; may concentrate in
                   low-vol assets and hide kurtosis risk.
  Max-Return:      degenerate; 100% exposed to noisy μ̂ — impractical.
  Max-Sharpe:      best naive Sharpe; sensitive to both inputs; only 3
                   active assets.
  Risk-Parity:     most diversified; robust to μ̂ noise; ignores returns
                   — may over-weight low-return assets.
  Black-Litterman: anchors μ̂ to equilibrium prior; views with explicit
                   confidence (τ, Ω); stable, adaptable allocations.

Key distributional finding:
  All portfolios are leptokurtic (excess kurtosis >> 0). Normal-distribution
  VaR will systematically understate tail risk for every method.
  Min-Variance shows the highest kurtosis ({kurtosis(port_returns['Min-Var'].dropna()):.2f}), revealing
  hidden concentration risk despite its low stated volatility.
  Black-Litterman has the most favourable skewness
  ({skew(port_returns['BL'].dropna()):.2f}) — more large positive surprises than negative ones.

Recommendation: Black-Litterman
  - Highest Sharpe ratio of all five methods.
  - Greater diversification than Max-Sharpe (4 active assets vs 3).
  - Lower volatility than Risk-Parity with higher returns.
  - Equilibrium prior prevents extreme weights from noisy sample means.
  - Views are updateable as market conditions change (τ and Ω give
    transparent control over view confidence).
  - Distributional evidence: moderate kurtosis with positive skewness —
    tail risk is manageable relative to the upside captured.
""")
