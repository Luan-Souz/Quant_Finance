####################     Northeastern University                                    ####################
####################     FINA 6339: Quantitative Portfolio Management               ####################
####################     Minimum-Variance Portfolio                                 ####################
####################     Prof. Dr. Milivoje Davidovic                               ####################
####################     Student: Luan Ferreira de Souza                            ####################

"""
Solves the global minimum-variance optimisation problem on a six-asset
universe (ETH-USD, BTC-USD, SPY, IEO, ABEV, PBR).

Objective:  min  √(wᵀ Σ w)
Subject to: Σ wᵢ = 1   (fully invested)
            wᵢ ∈ [0,1] (long-only, no leverage)

The covariance matrix is the only input — expected returns are ignored
entirely, making this the most robust method to return estimation error.

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
RF_ANNUAL = 0.0359      # 3-month US T-bill (constant, annualised)
N         = len(ASSETS)
ANN       = 252         # trading days per year

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
# OBJECTIVE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def obj_min_var(w, *args):
    """
    Compute portfolio daily volatility (%).

    Parameters
    ----------
    w    : weight vector (N,)
    args : (ann_ret, cov_matrix) — only cov_matrix is used

    Returns
    -------
    float : √(wᵀ Σ w) × 100
    """
    cov = args[1]
    return float(np.sqrt(np.dot(w, np.dot(cov, w))) * 100)

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────

result = sco.minimize(
    obj_min_var,
    w0,
    args=(ann_ret, cov_matrix),
    method="SLSQP",
    bounds=bounds,
    constraints=sum_to_one,
)

w_opt  = result.x
p_ret  = float(np.dot(ann_ret.values, w_opt))
p_vol  = float(np.sqrt(np.dot(w_opt, np.dot(cov_matrix.values * ANN, w_opt))) * 100)
sharpe = (p_ret - RF_ANNUAL * 100) / p_vol

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────

SEP = "=" * 70
print(f"\n{SEP}")
print("  Minimum-Variance Portfolio — Optimal Weights")
print(SEP)

rows = []
for i, a in enumerate(ASSETS):
    rows.append({
        "Asset":           a,
        "Weight (%)":      f"{w_opt[i]*100:.2f}",
        "Ann. Return (%)": f"{ann_ret.iloc[i]:.4f}",
        "Ann. Vol (%)":    f"{ann_vol.iloc[i]:.4f}",
    })
rows.append({
    "Asset":           "PORTFOLIO",
    "Weight (%)":      "100.00",
    "Ann. Return (%)": f"{p_ret:.4f}",
    "Ann. Vol (%)":    f"{p_vol:.4f}",
})
print(pd.DataFrame(rows).to_string(index=False))
print(f"\n  Sharpe Ratio: {sharpe:.4f}")

print(f"""
Interpretation:
  Both crypto assets (ETH, BTC) receive zero weight — their extreme
  volatility would raise portfolio risk even at tiny allocations.
  SPY dominates ({w_opt[2]*100:.1f}%) as the lowest-volatility asset in the
  universe, supplemented by ABEV ({w_opt[4]*100:.1f}%) and PBR ({w_opt[5]*100:.1f}%).

  This is the lowest achievable annualised volatility ({p_vol:.2f}%) given
  the six-asset universe and long-only constraints. The Sharpe ratio
  ({sharpe:.4f}) exceeds Max-Return (0.5386) despite delivering a lower
  raw return, confirming the benefit of risk-aware allocation.

  Estimation risk: None — μ̂ is not used. The only input is Σ̂, which
  is estimated more reliably than expected returns from historical data.
""")

# ─────────────────────────────────────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────────────────────────────────────

COLS = ["#dc2626", "#d97706", "#1d4ed8", "#059669", "#7c3aed", "#0891b2"]
mask = w_opt > 0.001

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(
    w_opt[mask],
    labels=[ASSETS[i] for i in range(N) if mask[i]],
    colors=[COLS[i] for i in range(N) if mask[i]],
    autopct="%.1f%%",
    startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=2),
)
ax.set_title("Minimum-Variance Portfolio — Weights", fontweight="bold", pad=16)
plt.tight_layout()
plt.savefig("weights_min_variance.png", dpi=150, bbox_inches="tight")
plt.show()
