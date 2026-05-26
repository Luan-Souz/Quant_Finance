####################     Northeastern University                                    ####################
####################     FINA 6339: Quantitative Portfolio Management               ####################
####################     Maximum-Return Portfolio                                    ####################
####################     Prof. Dr. Milivoje Davidovic                               ####################
####################     Student: Luan Ferreira de Souza                            ####################

"""
Solves the maximum-return optimisation problem on a six-asset universe
(ETH-USD, BTC-USD, SPY, IEO, ABEV, PBR).

Objective:  max  μᵀ w   ≡   min  −μᵀ w
Subject to: Σ wᵢ = 1   (fully invested)
            wᵢ ∈ [0,1] (long-only, no leverage)

Expected returns are the only input — the covariance matrix is ignored
entirely. Under long-only constraints this degenerates to allocating
100% to the single asset with the highest μ̂, making it the method most
exposed to return estimation error.

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
# OBJECTIVE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def obj_max_ret(w, *args):
    """
    Negate portfolio expected return so the minimiser finds the maximum.

    Parameters
    ----------
    w    : weight vector (N,)
    args : (ann_ret, cov_matrix) — only ann_ret is used

    Returns
    -------
    float : −μᵀ w
    """
    mu = args[0]
    return -float(np.dot(mu, w))

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────

result = sco.minimize(
    obj_max_ret,
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
print("  Maximum-Return Portfolio — Optimal Weights")
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
  100% allocated to BTC-USD — the asset with the highest sample mean
  return ({ann_ret['BTC-USD']:.2f}%). This is a degenerate, fully concentrated
  portfolio that ignores risk, diversification, and estimation bias.

  The resulting annualised volatility ({p_vol:.2f}%) is nearly three times
  the Min-Variance portfolio's volatility. Despite delivering the highest
  raw return ({p_ret:.2f}%), the Sharpe ratio ({sharpe:.4f}) is the lowest of
  all five methods — confirming that return maximisation without any risk
  constraint is counterproductive on a risk-adjusted basis.

  Estimation risk: Extreme — the entire allocation is determined solely
  by μ̂, which carries large standard errors. A small change in the
  sample period would flip the allocation to a different asset entirely.
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
ax.set_title("Maximum-Return Portfolio — Weights", fontweight="bold", pad=16)
plt.tight_layout()
plt.savefig("weights_max_return.png", dpi=150, bbox_inches="tight")
plt.show()
