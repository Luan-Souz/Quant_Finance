####################     Northeastern University                                    ####################
####################     FINA 6339: Quantitative Portfolio Management               ####################
####################     Maximum-Sharpe Portfolio (Tangency Portfolio)              ####################
####################     Prof. Dr. Milivoje Davidovic                               ####################
####################     Student: Luan Ferreira de Souza                            ####################

"""
Solves the maximum-Sharpe (tangency portfolio) optimisation problem on a
six-asset universe (ETH-USD, BTC-USD, SPY, IEO, ABEV, PBR).

Objective:  max  (μ_p − rf) / σ_p
Subject to: Σ wᵢ = 1   (fully invested)
            wᵢ ∈ [0,1] (long-only, no leverage)

Both μ̂ and Σ̂ are used simultaneously. The portfolio sits at the point
on the efficient frontier where the capital market line is tangent —
maximising the excess return per unit of risk. This is the theoretically
optimal risky portfolio for a mean-variance investor who can combine it
with the risk-free asset.

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

def obj_max_sharpe(w, *args):
    """
    Negate the Sharpe ratio so the minimiser finds the maximum.

    Parameters
    ----------
    w    : weight vector (N,)
    args : (ann_ret, cov_matrix, rf_pct)

    Returns
    -------
    float : −(μ_p − rf) / σ_p
    """
    mu, cov, rf = args
    p_vol = float(np.sqrt(np.dot(w, np.dot(cov, w)) * ANN) * 100)
    p_ret = float(np.dot(mu, w))
    return -(p_ret - rf) / p_vol

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────

result = sco.minimize(
    obj_max_sharpe,
    w0,
    args=(ann_ret.values, cov_matrix.values, RF_ANNUAL * 100),
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
print("  Maximum-Sharpe Portfolio — Optimal Weights")
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
  SPY ({w_opt[2]*100:.1f}%) anchors the portfolio as the asset with the best
  risk-adjusted profile given the covariance structure. PBR ({w_opt[5]*100:.1f}%)
  adds return premium with limited additional risk due to its low
  correlation with US equities. BTC ({w_opt[1]*100:.1f}%) provides a modest
  crypto exposure. IEO and ETH receive zero weight.

  This is the highest Sharpe ratio achievable under naive mean-variance
  optimisation ({sharpe:.4f}), outperforming both B1 portfolios on a
  risk-adjusted basis. However, only 3 active assets reflect the
  concentration risk inherent in unconstrained MV optimisation.

  Estimation risk: High — the objective depends on both μ̂ (noisy) and
  Σ̂. A two-month change in the sample period would meaningfully shift
  the weights.
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
ax.set_title("Maximum-Sharpe Portfolio — Weights", fontweight="bold", pad=16)
plt.tight_layout()
plt.savefig("weights_max_sharpe.png", dpi=150, bbox_inches="tight")
plt.show()
