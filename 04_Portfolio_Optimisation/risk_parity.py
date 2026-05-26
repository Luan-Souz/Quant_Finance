####################     Northeastern University                                    ####################
####################     FINA 6339: Quantitative Portfolio Management               ####################
####################     Risk-Parity Portfolio (Equal Risk Contribution)            ####################
####################     Prof. Dr. Milivoje Davidovic                               ####################
####################     Student: Luan Ferreira de Souza                            ####################

"""
Solves the equal risk contribution (ERC) optimisation problem on a
six-asset universe (ETH-USD, BTC-USD, SPY, IEO, ABEV, PBR).

Objective:  min  (1/N) Σᵢ (RC_i − target)²
where       RC_i   = w_i × (Σw)_i / (wᵀΣw)   [asset i's % of total variance]
            target = mean(RC) = 1/N ≈ 16.67%  [equal risk target]
Subject to: Σ wᵢ = 1   (fully invested)
            wᵢ ∈ [0,1] (long-only, no leverage)

Expected returns are not used — only Σ̂ is required. This makes Risk-Parity
robust to return estimation error and the only method to allocate across
all six assets simultaneously.

Requires: clean_returns.csv  (output of part_a_data_inputs.py)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

ASSETS    = ["ETH-USD", "BTC-USD", "SPY", "IEO", "ABEV", "PBR"]
RF_ANNUAL = 0.0359
N         = len(ASSETS)
ANN       = 252

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

returns    = pd.read_csv("clean_returns.csv", index_col=0, parse_dates=True)
returns.columns = ASSETS

ann_ret    = returns.mean() * 100 * ANN
ann_vol    = returns.std(ddof=1) * 100 * np.sqrt(ANN)
cov_matrix = returns.cov()

w0         = np.ones(N) / N
bounds     = [(0, 1)] * N
sum_to_one = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: RISK CONTRIBUTION VECTOR
# ─────────────────────────────────────────────────────────────────────────────

def risk_contrib(w, cov):
    """
    Compute each asset's percentage contribution to total portfolio variance.

    Parameters
    ----------
    w   : weight vector (N,)
    cov : daily covariance matrix (N×N)

    Returns
    -------
    RC : array (N,) where RC_i = w_i × (Σw)_i / (wᵀΣw), summing to 100%
    """
    port_var     = float(np.dot(w, np.dot(cov * ANN, w)) * 100)
    marg_contrib = np.dot(cov * ANN, w)
    return w * 100 * marg_contrib / port_var

# ─────────────────────────────────────────────────────────────────────────────
# OBJECTIVE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def obj_risk_parity(w, *args):
    """
    Mean squared deviation of each asset's risk contribution from the
    equal-risk target (1/N ≈ 16.67% for six assets).

    Parameters
    ----------
    w    : weight vector (N,)
    args : (ann_ret, cov_matrix, rf_pct) — only cov_matrix is used

    Returns
    -------
    float : mean((RC_i − target)²) → 0 at the equal-risk solution
    """
    cov    = args[1]
    rc     = risk_contrib(w, cov)
    target = np.mean(rc)
    return float(np.mean((rc - target) ** 2))

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────

result = sco.minimize(
    obj_risk_parity,
    w0,
    args=(ann_ret.values, cov_matrix.values, RF_ANNUAL * 100),
    method="SLSQP",
    bounds=bounds,
    constraints=sum_to_one,
)

w_opt  = result.x
rc_opt = risk_contrib(w_opt, cov_matrix.values)
p_ret  = float(np.dot(ann_ret.values, w_opt))
p_vol  = float(np.sqrt(np.dot(w_opt, np.dot(cov_matrix.values * ANN, w_opt))) * 100)
sharpe = (p_ret - RF_ANNUAL * 100) / p_vol

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────

SEP = "=" * 70
print(f"\n{SEP}")
print("  Risk-Parity Portfolio — Optimal Weights & Risk Contributions")
print(SEP)

rows = []
for i, a in enumerate(ASSETS):
    rows.append({
        "Asset":              a,
        "Weight (%)":         f"{w_opt[i]*100:.2f}",
        "Risk Contrib (%)":   f"{rc_opt[i]:.2f}",
        "Ann. Return (%)":    f"{ann_ret.iloc[i]:.4f}",
        "Ann. Vol (%)":       f"{ann_vol.iloc[i]:.4f}",
    })
rows.append({
    "Asset":              "PORTFOLIO",
    "Weight (%)":         "100.00",
    "Risk Contrib (%)":   "100.00",
    "Ann. Return (%)":    f"{p_ret:.4f}",
    "Ann. Vol (%)":       f"{p_vol:.4f}",
})
print(pd.DataFrame(rows).to_string(index=False))
print(f"\n  Sharpe Ratio:  {sharpe:.4f}")
print(f"  Target RC/asset: {100/N:.2f}%  |  Max deviation: {max(abs(rc_opt - 100/N)):.4f}%")

print(f"""
Interpretation:
  Each asset contributes ≈ 1/6 ≈ 16.67% of total portfolio variance —
  the defining characteristic of the ERC portfolio. High-volatility
  assets (ETH {w_opt[0]*100:.1f}%, BTC {w_opt[1]*100:.1f}%) receive the smallest weights
  to equalise their risk contribution; low-volatility SPY ({w_opt[2]*100:.1f}%)
  receives the largest.

  Risk-Parity is the only method to include all six assets.
  It requires no expected return estimates (μ̂ is entirely absent from
  the objective), making it the most robust to return estimation error
  alongside Min-Variance.

  Trade-off: lower Sharpe ({sharpe:.4f}) than Max-Sharpe and Black-Litterman
  because ignoring return information prevents the optimiser from
  concentrating weight in high-return assets.
""")

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

COLS = ["#dc2626", "#d97706", "#1d4ed8", "#059669", "#7c3aed", "#0891b2"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Weight allocation
ax1.pie(
    w_opt,
    labels=ASSETS,
    colors=COLS,
    autopct="%.1f%%",
    startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=2),
)
ax1.set_title("Portfolio Weights", fontweight="bold", pad=14)

# Risk contributions
ax2.pie(
    rc_opt,
    labels=ASSETS,
    colors=COLS,
    autopct="%.2f%%",
    startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=2),
)
ax2.set_title("Risk Contributions", fontweight="bold", pad=14)

plt.suptitle("Risk-Parity Portfolio", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("weights_risk_parity.png", dpi=150, bbox_inches="tight")
plt.show()
