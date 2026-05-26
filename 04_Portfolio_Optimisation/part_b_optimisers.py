####################     Northeastern University                                    ####################
####################     FINA 6339: Quantitative Portfolio Management               ####################
####################     Portfolio Optimisation — Five Optimisers                   ####################
####################     Prof. Dr. Milivoje Davidovic                               ####################
####################     Student: Luan Ferreira de Souza                            ####################

"""
Implements and compares five portfolio optimisation methods on a six-asset
universe (ETH-USD, BTC-USD, SPY, IEO, ABEV, PBR):

  B1 — Minimum-Variance Portfolio  (global minimum variance)
  B1 — Maximum-Return Portfolio    (highest expected return)
  B2 — Maximum-Sharpe Portfolio    (tangency portfolio)
  B2 — Risk-Parity Portfolio       (equal risk contribution)
  B3 — Black-Litterman Portfolio   (Bayesian equilibrium + investor views)

Constraints applied to all methods:
  Long-only:       w_i ∈ [0, 1]
  Fully invested:  Σ w_i = 1

Requires: clean_returns.csv  (output of part_a_data_inputs.py)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & SHARED INPUTS
# ─────────────────────────────────────────────────────────────────────────────

ASSETS    = ["ETH-USD", "BTC-USD", "SPY", "IEO", "ABEV", "PBR"]
RF_ANNUAL = 0.0359          # 3-month US T-bill (constant, annualised)
N         = len(ASSETS)
ANN       = 252             # trading days per year

returns    = pd.read_csv("clean_returns.csv", index_col=0, parse_dates=True)
returns.columns = ASSETS

ann_ret    = returns.mean() * 100 * ANN          # annualised return, % Series
ann_vol    = returns.std(ddof=1) * 100 * np.sqrt(ANN)
cov_matrix = returns.cov()                       # daily covariance matrix

# Shared solver settings
w0          = np.ones(N) / N                     # equal-weight initialisation
bounds      = [(0, 1)] * N                       # long-only
sum_to_one  = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_stats(w, mu_ann, cov_daily, rf_ann_pct):
    """
    Compute annualised portfolio return, volatility, and Sharpe ratio.

    Parameters
    ----------
    w           : weight vector (N,)
    mu_ann      : annualised expected returns, % (N,)
    cov_daily   : daily covariance matrix (N×N)
    rf_ann_pct  : annualised risk-free rate in % (e.g. 3.59)

    Returns
    -------
    (port_return %, port_vol %, sharpe_ratio)
    """
    p_ret = float(np.dot(mu_ann, w))
    p_vol = float(np.sqrt(np.dot(w, np.dot(cov_daily * ANN, w))) * 100)
    sharpe = (p_ret - rf_ann_pct) / p_vol
    return p_ret, p_vol, sharpe


def print_portfolio(label, weights, stats_tuple):
    """Print a formatted portfolio summary table."""
    p_ret, p_vol, sharpe = stats_tuple
    print("=" * 70)
    print(f"  {label}")
    print("=" * 70)
    rows = []
    for i, a in enumerate(ASSETS):
        rows.append({
            "Asset":           a,
            "Weight (%)":      f"{weights[i]*100:.2f}",
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
    print(f"  Sharpe Ratio: {sharpe:.4f}\n")

# ─────────────────────────────────────────────────────────────────────────────
# B1-i — MINIMUM-VARIANCE PORTFOLIO
# Objective: min √(wᵀΣw)   s.t.  Σwᵢ=1, wᵢ∈[0,1]
# ─────────────────────────────────────────────────────────────────────────────

def obj_min_var(w, *args):
    """Minimise portfolio daily volatility (%)."""
    cov = args[1]
    return float(np.sqrt(np.dot(w, np.dot(cov, w))) * 100)


res_mv = sco.minimize(obj_min_var, w0,
                      args=(ann_ret, cov_matrix),
                      method="SLSQP",
                      bounds=bounds,
                      constraints=sum_to_one)
w_mv   = res_mv.x
s_mv   = portfolio_stats(w_mv, ann_ret.values, cov_matrix.values, RF_ANNUAL * 100)

print_portfolio("Minimum-Variance Portfolio", w_mv, s_mv)
print(f"""  Interpretation:
  Both crypto assets (ETH, BTC) receive zero weight — their extreme
  volatility would raise portfolio risk even at tiny allocations.
  SPY dominates ({w_mv[2]*100:.1f}%) as the lowest-volatility asset.
""")

# ─────────────────────────────────────────────────────────────────────────────
# B1-ii — MAXIMUM-RETURN PORTFOLIO
# Objective: max μᵀw  ≡  min −μᵀw   s.t.  Σwᵢ=1, wᵢ∈[0,1]
# ─────────────────────────────────────────────────────────────────────────────

def obj_max_ret(w, *args):
    """Negate portfolio return — minimiser finds the maximum."""
    mu = args[0]
    return -float(np.dot(mu, w))


res_mr = sco.minimize(obj_max_ret, w0,
                      args=(ann_ret, cov_matrix),
                      method="SLSQP",
                      bounds=bounds,
                      constraints=sum_to_one)
w_mr   = res_mr.x
s_mr   = portfolio_stats(w_mr, ann_ret.values, cov_matrix.values, RF_ANNUAL * 100)

print_portfolio("Maximum-Return Portfolio", w_mr, s_mr)
print(f"""  Interpretation:
  100% allocated to BTC-USD — the highest expected-return asset.
  This is a degenerate, fully concentrated portfolio that ignores
  risk entirely. Sharpe ratio ({s_mr[2]:.4f}) is the lowest of all five
  methods, confirming that return maximisation without risk control
  is counterproductive on a risk-adjusted basis.
""")

# ─────────────────────────────────────────────────────────────────────────────
# B2-i — MAXIMUM-SHARPE PORTFOLIO (Tangency Portfolio)
# Objective: max (μ_p − rf) / σ_p   s.t.  Σwᵢ=1, wᵢ∈[0,1]
# ─────────────────────────────────────────────────────────────────────────────

def obj_max_sharpe(w, *args):
    """Negate Sharpe ratio — minimiser finds the maximum."""
    mu, cov, rf = args
    p_vol = float(np.sqrt(np.dot(w, np.dot(cov, w)) * ANN) * 100)
    p_ret = float(np.dot(mu, w))
    return -(p_ret - rf) / p_vol


res_ms = sco.minimize(obj_max_sharpe, w0,
                      args=(ann_ret.values, cov_matrix.values, RF_ANNUAL * 100),
                      method="SLSQP",
                      bounds=bounds,
                      constraints=sum_to_one)
w_ms   = res_ms.x
s_ms   = portfolio_stats(w_ms, ann_ret.values, cov_matrix.values, RF_ANNUAL * 100)

print_portfolio("Maximum-Sharpe Portfolio", w_ms, s_ms)
print(f"""  Interpretation:
  SPY ({w_ms[2]*100:.1f}%) anchors the portfolio. PBR ({w_ms[5]*100:.1f}%) adds return
  without proportionate risk due to low US-equity correlation.
  BTC ({w_ms[1]*100:.1f}%) provides modest crypto exposure.
  Best naive mean-variance Sharpe: {s_ms[2]:.4f}.
""")

# ─────────────────────────────────────────────────────────────────────────────
# B2-ii — RISK-PARITY PORTFOLIO (Equal Risk Contribution)
# Objective: min mean((RC_i − target)²)   s.t.  Σwᵢ=1, wᵢ∈[0,1]
# where RC_i = w_i × (Σw)_i / (wᵀΣw)  and  target = 1/N
# ─────────────────────────────────────────────────────────────────────────────

def risk_contrib(w, cov):
    """Return the percentage risk contribution vector for weight array w."""
    port_var    = float(np.dot(w, np.dot(cov * ANN, w)) * 100)
    marg_contrib = np.dot(cov * ANN, w)
    return w * 100 * marg_contrib / port_var


def obj_risk_parity(w, *args):
    """Mean squared deviation from equal risk contribution target."""
    cov    = args[1]
    rc     = risk_contrib(w, cov)
    target = np.mean(rc)          # = 100/N ≈ 16.67% for 6 assets
    return float(np.mean((rc - target) ** 2))


res_rp = sco.minimize(obj_risk_parity, w0,
                      args=(ann_ret.values, cov_matrix.values, RF_ANNUAL * 100),
                      method="SLSQP",
                      bounds=bounds,
                      constraints=sum_to_one)
w_rp   = res_rp.x
s_rp   = portfolio_stats(w_rp, ann_ret.values, cov_matrix.values, RF_ANNUAL * 100)
rc_rp  = risk_contrib(w_rp, cov_matrix.values)

print_portfolio("Risk-Parity Portfolio", w_rp, s_rp)
print("  Risk Contributions:")
for a, rc in zip(ASSETS, rc_rp):
    print(f"    {a:<10}  {rc:.2f}%")
print(f"""
  Interpretation:
  Each asset contributes ≈1/6 ≈ 16.67% of total portfolio variance.
  All six assets enter the portfolio — the only method to achieve this.
  ETH ({w_rp[0]*100:.1f}%) and BTC ({w_rp[1]*100:.1f}%) receive the smallest weights
  to compensate for their high volatility. μ̂ is not used — making
  this the most robust method to return estimation error.
""")

# ─────────────────────────────────────────────────────────────────────────────
# B3 — BLACK-LITTERMAN PORTFOLIO
# Prior: EWMA log returns (span=10), annualised
# Views: BTC @ 30%,  SPY @ 15%  (both absolute)
# Posterior: μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹π + P'Ω⁻¹Q]
# Optimisation: Max-Sharpe on posterior returns
# ─────────────────────────────────────────────────────────────────────────────

# Prior: EWMA expected returns
EWMR  = returns.ewm(span=10, adjust=True).mean()
prior = np.array(EWMR.mean() * ANN * 100)       # annualised %, length-N array

# Investor views
tau = 0.05
k   = 2
P   = np.zeros((k, N))
P[0, ASSETS.index("BTC-USD")] = 1.0    # View 1: BTC @ 30%
P[1, ASSETS.index("SPY")]     = 1.0    # View 2: SPY @ 15%
Q   = np.array([30.0, 15.0])           # view returns (%)

# View uncertainty matrix: proportional to prior covariance
Sigma = cov_matrix.values
Omega = np.diag(np.diag(tau * P @ Sigma @ P.T))

# Posterior expected returns (Black-Litterman formula)
inv_tau_Sigma = np.linalg.inv(tau * Sigma)
inv_Omega     = np.linalg.inv(Omega)
H             = np.linalg.inv(inv_tau_Sigma + P.T @ inv_Omega @ P)
mu_bl         = H @ (inv_tau_Sigma @ prior + P.T @ inv_Omega @ Q)

# Updated covariance (adds posterior uncertainty)
Cov_BL = Sigma + H

def obj_bl_sharpe(w, *args):
    """Max-Sharpe objective using BL posterior returns and covariance."""
    mu_bl_, cov_bl, rf = args
    p_vol = float(np.sqrt(np.dot(w, np.dot(cov_bl * ANN, w))) * 100)
    p_ret = float(np.dot(mu_bl_, w))
    return -(p_ret - rf) / p_vol


res_bl = sco.minimize(obj_bl_sharpe, w0,
                      args=(mu_bl, Cov_BL, RF_ANNUAL * 100),
                      method="SLSQP",
                      bounds=bounds,
                      constraints=sum_to_one)
w_bl   = res_bl.x

p_ret_bl = float(np.dot(mu_bl, w_bl))
p_vol_bl = float(np.sqrt(np.dot(w_bl, np.dot(Cov_BL * ANN, w_bl))) * 100)
sr_bl    = (p_ret_bl - RF_ANNUAL * 100) / p_vol_bl

# Prior vs Posterior table
print("=" * 65)
print("  Black-Litterman — Prior vs Posterior Expected Returns (%)")
print("=" * 65)
pp_df = pd.DataFrame({
    "Asset":        ASSETS,
    "Prior (%)":    prior.round(2),
    "Posterior (%)": mu_bl.round(2),
    "Δ (pp)":       (mu_bl - prior).round(2),
})
print(pp_df.to_string(index=False))

print("\n  BL Optimal Weights:")
for a, w in zip(ASSETS, w_bl):
    print(f"    {a:<10}  {w*100:.2f}%")
print(f"\n  Portfolio Return:    {p_ret_bl:.4f}%")
print(f"  Portfolio Vol:       {p_vol_bl:.4f}%")
print(f"  Sharpe Ratio:        {sr_bl:.4f}")
print(f"""
  Interpretation:
  Prior: EWMA log returns (span=10, annualised ×252) — emphasises
         recent price history over the full sample mean.
  Views: BTC @ 30% (bullish above prior ~25%), SPY @ 15% (above ~12%).
  Posterior: All six returns move upward. ETH gains the most (+7.3 pp)
             via its high covariance with BTC (ρ=0.82), despite having
             no direct view assigned.
  Outcome: ABEV enters the portfolio at 6.45% — nearly absent from
           naive Max-Sharpe (0.09%) — because the equilibrium prior
           raises its relative attractiveness. Highest Sharpe of all
           five methods: {sr_bl:.4f}.
""")

# ─────────────────────────────────────────────────────────────────────────────
# CROSS-COMPARISON SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

portfolios = {
    "Min-Variance":    (w_mv, ann_ret.values),
    "Max-Return":      (w_mr, ann_ret.values),
    "Max-Sharpe":      (w_ms, ann_ret.values),
    "Risk-Parity":     (w_rp, ann_ret.values),
    "Black-Litterman": (w_bl, mu_bl),
}

rows = []
for name, (w, mu) in portfolios.items():
    pr, pv, sr = portfolio_stats(w, mu, cov_matrix.values, RF_ANNUAL * 100)
    herfindahl  = float(np.sum(w ** 2))
    rows.append({
        "Portfolio":       name,
        "Ann. Return (%)": f"{pr:.2f}",
        "Ann. Vol (%)":    f"{pv:.2f}",
        "Sharpe":          f"{sr:.4f}",
        "Herfindahl":      f"{herfindahl:.4f}",
        "Active Assets":   int(np.sum(w > 0.01)),
    })

print("=" * 100)
print(" " * 30 + "FULL COMPARISON — All Five Portfolios")
print("=" * 100)
print(pd.DataFrame(rows).to_string(index=False))

# ── Weight allocation bar chart ────────────────────────────────────────────────
weights_dict = {
    "Min-Var":  w_mv,
    "Max-Ret":  w_mr,
    "Max-Sharpe": w_ms,
    "Risk-Parity": w_rp,
    "BL":       w_bl,
}

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
COLS = ["#dc2626", "#d97706", "#1d4ed8", "#059669", "#7c3aed", "#0891b2"]

for ax, (title, w) in zip(axes, weights_dict.items()):
    mask = w > 0.001
    ax.pie(w[mask],
           labels=[ASSETS[i] for i in range(N) if mask[i]],
           colors=[COLS[i] for i in range(N) if mask[i]],
           autopct="%.1f%%", startangle=90,
           wedgeprops=dict(edgecolor="white", linewidth=1.5))
    ax.set_title(title, fontweight="bold")

axes[-1].axis("off")
plt.suptitle("Portfolio Weights — All Five Methods", fontsize=13)
plt.tight_layout()
plt.savefig("portfolio_weights.png", dpi=150, bbox_inches="tight")
plt.show()

# Export weights for post-optimisation analysis
weights_df = pd.DataFrame(weights_dict, index=ASSETS)
weights_df.to_csv("optimal_weights.csv")
print("Exported: optimal_weights.csv")
