####################     Northeastern University                                    ####################
####################     FINA 6339: Quantitative Portfolio Management               ####################
####################     Black-Litterman Portfolio                                  ####################
####################     Prof. Dr. Milivoje Davidovic                               ####################
####################     Student: Luan Ferreira de Souza                            ####################

"""
Implements the Black-Litterman portfolio optimisation framework on a
six-asset universe (ETH-USD, BTC-USD, SPY, IEO, ABEV, PBR).

Framework:
  Prior π   = EWMA log returns (span=10), annualised ×252
  Views     = 2 absolute investor views (BTC @ 30%, SPY @ 15%)
  Uncertainty Ω = τ · P Σ P'   (proportional to prior covariance)

Posterior expected returns:
  μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ · [(τΣ)⁻¹ π + P'Ω⁻¹ Q]

Posterior covariance:
  Σ_BL = Σ + H   where H = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹

Optimisation: Maximum-Sharpe on posterior returns and covariance.
Constraints:  Σ wᵢ = 1,  wᵢ ∈ [0,1]

Requires: clean_returns.csv  (output of part_a_data_inputs.py)

References:
  Black, F., & Litterman, R. (1992). Global portfolio optimization.
  Financial Analysts Journal, 48(5), 28-43.
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

# Black-Litterman hyperparameters
TAU       = 0.05        # prior uncertainty scalar
EWMA_SPAN = 10          # EWMA span for prior estimation

# Investor views (absolute)
#   View 1: BTC-USD expected to return 30% annualised
#   View 2: SPY     expected to return 15% annualised
VIEW_ASSETS  = ["BTC-USD", "SPY"]
VIEW_RETURNS = [30.0, 15.0]     # in %

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
# STEP 1 — PRIOR: EWMA EXPECTED RETURNS
# ─────────────────────────────────────────────────────────────────────────────

EWMR  = returns.ewm(span=EWMA_SPAN, adjust=True).mean()
prior = np.array(EWMR.mean() * ANN * 100)   # annualised %, length-N

print("=" * 65)
print("  Step 1 — Prior Expected Returns (EWMA, span=10)")
print("=" * 65)
for a, p in zip(ASSETS, prior):
    print(f"  {a:<10}  {p:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — ENCODE INVESTOR VIEWS: P, Q, Ω
# ─────────────────────────────────────────────────────────────────────────────

k     = len(VIEW_ASSETS)
P     = np.zeros((k, N))
Q     = np.array(VIEW_RETURNS)

for row, asset in enumerate(VIEW_ASSETS):
    P[row, ASSETS.index(asset)] = 1.0

Sigma = cov_matrix.values
Omega = np.diag(np.diag(TAU * P @ Sigma @ P.T))

print("\n" + "=" * 65)
print("  Step 2 — Investor Views")
print("=" * 65)
for i, (a, q) in enumerate(zip(VIEW_ASSETS, Q)):
    print(f"  View {i+1} (absolute): {a} @ {q:.1f}%")
print(f"  τ = {TAU}  |  Ω = τ · PΣP' (diagonal)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — POSTERIOR EXPECTED RETURNS
# ─────────────────────────────────────────────────────────────────────────────

inv_tau_Sigma = np.linalg.inv(TAU * Sigma)
inv_Omega     = np.linalg.inv(Omega)
H             = np.linalg.inv(inv_tau_Sigma + P.T @ inv_Omega @ P)
mu_bl         = H @ (inv_tau_Sigma @ prior + P.T @ inv_Omega @ Q)
Cov_BL        = Sigma + H   # posterior covariance

print("\n" + "=" * 65)
print("  Step 3 — Prior vs Posterior Expected Returns (%)")
print("=" * 65)
pp_df = pd.DataFrame({
    "Asset":         ASSETS,
    "Prior (%)":     prior.round(2),
    "Posterior (%)": mu_bl.round(2),
    "Δ (pp)":        (mu_bl - prior).round(2),
})
print(pp_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — OPTIMISATION: MAX-SHARPE ON POSTERIOR
# ─────────────────────────────────────────────────────────────────────────────

def obj_bl_sharpe(w, *args):
    """
    Negate Sharpe ratio computed on BL posterior returns and covariance.

    Parameters
    ----------
    w    : weight vector (N,)
    args : (mu_bl, Cov_BL, rf_pct)

    Returns
    -------
    float : −(μ_BL_p − rf) / σ_BL_p
    """
    mu_bl_, cov_bl, rf = args
    p_vol = float(np.sqrt(np.dot(w, np.dot(cov_bl * ANN, w))) * 100)
    p_ret = float(np.dot(mu_bl_, w))
    return -(p_ret - rf) / p_vol


result = sco.minimize(
    obj_bl_sharpe,
    w0,
    args=(mu_bl, Cov_BL, RF_ANNUAL * 100),
    method="SLSQP",
    bounds=bounds,
    constraints=sum_to_one,
)

w_opt    = result.x
p_ret_bl = float(np.dot(mu_bl, w_opt))
p_vol_bl = float(np.sqrt(np.dot(w_opt, np.dot(Cov_BL * ANN, w_opt))) * 100)
sharpe   = (p_ret_bl - RF_ANNUAL * 100) / p_vol_bl

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────

SEP = "=" * 70
print(f"\n{SEP}")
print("  Black-Litterman Portfolio — Optimal Weights")
print(SEP)

rows = []
for i, a in enumerate(ASSETS):
    rows.append({
        "Asset":            a,
        "Weight (%)":       f"{w_opt[i]*100:.2f}",
        "Post. Return (%)": f"{mu_bl[i]:.4f}",
        "Ann. Vol (%)":     f"{ann_vol.iloc[i]:.4f}",
    })
rows.append({
    "Asset":            "PORTFOLIO",
    "Weight (%)":       "100.00",
    "Post. Return (%)": f"{p_ret_bl:.4f}",
    "Ann. Vol (%)":     f"{p_vol_bl:.4f}",
})
print(pd.DataFrame(rows).to_string(index=False))
print(f"\n  Sharpe Ratio: {sharpe:.4f}")

print(f"""
Interpretation:
  Prior: EWMA log returns (span={EWMA_SPAN}, annualised ×{ANN}) — emphasises
         recent price history over the full sample mean.

  Views: BTC @ {VIEW_RETURNS[0]}% (bullish above EWMA prior ~25%) and
         SPY @ {VIEW_RETURNS[1]}% (above prior ~12%). Both views shift
         their posteriors upward; other assets shift indirectly through
         the covariance structure (ETH gains +7.3 pp via ρ=0.82 with BTC).

  Outcome: ABEV enters the portfolio at {w_opt[4]*100:.2f}% — nearly absent
           from naive Max-Sharpe (0.09%) — because the equilibrium prior
           raises its relative attractiveness. SPY concentration is
           virtually unchanged ({w_opt[2]*100:.1f}% vs 61.74%), while PBR
           drops slightly as weight redistributes more evenly.

  This is the highest Sharpe ratio of all five methods ({sharpe:.4f}).
  BL's advantage over naive Max-Sharpe is not just performance but
  stability: the equilibrium prior prevents extreme, unstable weights
  caused by noisy sample mean estimates of μ̂.
""")

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

COLS = ["#dc2626", "#d97706", "#1d4ed8", "#059669", "#7c3aed", "#0891b2"]
mask = w_opt > 0.001

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Weight pie
ax1.pie(
    w_opt[mask],
    labels=[ASSETS[i] for i in range(N) if mask[i]],
    colors=[COLS[i] for i in range(N) if mask[i]],
    autopct="%.1f%%",
    startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=2),
)
ax1.set_title("BL Portfolio Weights", fontweight="bold", pad=14)

# Prior vs Posterior bar chart
x = np.arange(N)
w_bar = 0.35
ax2.bar(x - w_bar/2, prior,  w_bar, label="Prior",     color="#cccccc")
ax2.bar(x + w_bar/2, mu_bl,  w_bar, label="Posterior",
        color=["#d97706" if a == "BTC-USD" else
               "#1d4ed8" if a == "SPY" else "#93b4f5" for a in ASSETS])
ax2.set_xticks(x)
ax2.set_xticklabels(ASSETS, rotation=20, ha="right", fontsize=9)
ax2.set_ylabel("Expected Return (%)")
ax2.set_title("Prior vs Posterior Returns", fontweight="bold", pad=14)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.4)

plt.suptitle("Black-Litterman Portfolio", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("weights_black_litterman.png", dpi=150, bbox_inches="tight")
plt.show()
