####################     Northeastern University                                    ####################
####################     FINA 6339: Quantitative Portfolio Management               ####################
####################     Homework 3 — Heston Stochastic Volatility Model           ####################
####################     Prof. Dr. Milivoje Davidovic                               ####################
####################     Aluno: Luan Ferreira de Souza                              ####################

"""
Prices a European call option using the Heston (1993) stochastic volatility
model simulated via Monte Carlo with a full-truncation Euler scheme.

Unlike Black-Scholes, volatility here is itself a stochastic process that
mean-reverts, clusters, and correlates with the stock — reproducing three
empirically observed phenomena: volatility clustering, mean-reversion, and
the leverage effect (ρ < 0 → stock drops coincide with vol spikes).

Heston variance dynamics (CIR process):
  dv(t) = κ(θ − v(t))dt + ξ√v(t) dW₂

Stock dynamics (risk-neutral measure):
  dS(t) = r·S(t)dt + √v(t)·S(t) dW₁

Correlated Brownians:
  dW₁·dW₂ = ρ·dt

Discretization (full-truncation Euler scheme, daily steps):
  v(t+dt) = max(v + κ(θ−v)dt + ξ·√max(v,0)·dt·Z₂, 0)
  S(t+dt) = S · exp((r − ½v)dt + √max(v,0)·dt·Z₁)

Full truncation prevents negative variance — a known instability of the
Euler scheme for the CIR process. The stock uses exact log-Euler given v.

Parameters:
  Core   : S0=100, K=105, T=1, r=0.04
  Heston : v0=0.09, κ=2.0, θ=0.09, ξ=0.30, ρ=−0.70
  Paths  : 50,000  |  Steps: 252 (daily)

Parameter justification:
  v0 = θ = 0.09  → σ₀ = σ∞ = 30%, matching the BS baseline for comparability
  κ  = 2.0       → moderate mean-reversion; half-life ≈ ln2/κ ≈ 4.2 months
  ξ  = 0.30      → meaningful vol-of-vol without violating Feller condition
  ρ  = −0.70     → strong leverage effect; empirically typical for equities

Feller condition for guaranteed v > 0 in the continuous limit: 2κθ > ξ²
  2 × 2.0 × 0.09 = 0.36  >  0.09 = 0.30²  ✓

Requires: numpy, pandas, matplotlib, scipy

References:
  Heston, S. L. (1993). A closed-form solution for options with stochastic
  volatility with applications to bond and currency options.
  Review of Financial Studies, 6(2), 327–343.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

S0      = 100          # current stock price
K       = 105          # strike price (slightly OTM)
T       = 1            # time to maturity (years)
r       = 0.04         # continuously compounded risk-free rate

# Heston parameters
V0      = 0.09         # initial variance (σ₀ = 30%)
KAPPA   = 2.0          # mean-reversion speed
THETA   = 0.09         # long-run variance (σ∞ = 30%)
XI      = 0.30         # volatility of variance (vol-of-vol)
RHO     = -0.70        # stock-vol correlation (leverage effect)

N_PATHS = 50_000       # Monte Carlo paths
N_STEPS = 252          # daily time steps

# ─────────────────────────────────────────────────────────────────────────────
# PRICING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def heston_call(S0, K, T, r, v0=V0, kappa=KAPPA, theta=THETA,
                xi=XI, rho=RHO, n_paths=N_PATHS, n_steps=N_STEPS):
    """
    European call price via Heston stochastic volatility model (Monte Carlo).
    Full-truncation Euler scheme for the variance process.

    Parameters
    ----------
    S0, K, T, r          : standard option parameters
    v0                   : initial variance (σ₀ = √v0)
    kappa                : mean-reversion speed of variance
    theta                : long-run variance (σ∞ = √θ)
    xi                   : volatility of variance (vol-of-vol)
    rho                  : correlation between stock and variance Brownians
    n_paths              : number of Monte Carlo paths
    n_steps              : number of time steps (252 = daily for T=1yr)

    Returns
    -------
    float : discounted expected payoff under Heston measure
    """
    dt = T / n_steps
    S  = np.full(n_paths, float(S0))
    v  = np.full(n_paths, float(v0))

    for _ in range(n_steps):
        W1    = np.random.standard_normal(n_paths)
        W2    = np.random.standard_normal(n_paths)
        Z1    = W1
        Z2    = rho * W1 + np.sqrt(1 - rho**2) * W2    # correlated Brownian

        v_pos = np.maximum(v, 0)                        # full truncation
        v     = np.maximum(
                    v + kappa * (theta - v) * dt
                    + xi * np.sqrt(v_pos * dt) * Z2,
                    0)
        S     = S * np.exp((r - 0.5 * v_pos) * dt
                           + np.sqrt(v_pos * dt) * Z1)

    return float(np.exp(-r * T) * np.mean(np.maximum(S - K, 0)))


def bs_call(S0, K, T, r, sigma):
    """Black-Scholes call price — used as analytical benchmark."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

# ─────────────────────────────────────────────────────────────────────────────
# BASE CASE
# ─────────────────────────────────────────────────────────────────────────────

C_heston = heston_call(S0, K, T, r)
C_bs     = bs_call(S0, K, T, r, np.sqrt(V0))    # BS at σ = √v0 = 30%

feller = 2 * KAPPA * THETA
xi_sq  = XI**2

SEP = "=" * 70
print(f"\n{SEP}")
print("  Heston Model — Parameter Summary")
print(SEP)
print(f"  S0={S0}  |  K={K}  |  T={T}  |  r={r}")
print(f"  v0={V0}  (σ₀ = {np.sqrt(V0)*100:.0f}%)  |  κ={KAPPA}  |  θ={THETA}  "
      f"(σ∞ = {np.sqrt(THETA)*100:.0f}%)  |  ξ={XI}  |  ρ={RHO}")
print(f"  Paths = {N_PATHS:,}  |  Steps = {N_STEPS} (daily)  |  Seed = 42")
print(f"  Feller condition 2κθ > ξ²: {feller:.2f} > {xi_sq:.2f}  "
      f"{'✓' if feller > xi_sq else '✗'}")
print(SEP)

print(f"\n{SEP}")
print("  Results")
print(SEP)
print(f"  Heston call price         : {C_heston:.4f}")
print(f"  Black-Scholes (σ=30%)     : {C_bs:.4f}")
print(f"  Difference (Heston − BS)  : {C_heston - C_bs:+.4f}")

print(f"""
Interpretation:
  The Heston price ({C_heston:.4f}) is lower than Black-Scholes ({C_bs:.4f})
  due to the leverage effect: ρ = {RHO} means stock drops coincide with
  volatility spikes, shifting the risk-neutral distribution toward the
  left tail. This increases the probability of finishing out-of-the-money
  and reduces the expected call payoff.

  Black-Scholes cannot capture this asymmetry — it prices all paths
  symmetrically under constant σ. Heston's ability to model the
  stock-vol feedback is its core advantage over BS for equity options.
""")

# ─────────────────────────────────────────────────────────────────────────────
# SENSITIVITY ANALYSIS — ρ
# ─────────────────────────────────────────────────────────────────────────────

rho_vals   = [-0.9, -0.7, -0.5, -0.3, 0.0]
heston_rho = [heston_call(S0, K, T, r, rho=rho) for rho in rho_vals]

print(f"\n{SEP}")
print("  Sensitivity to Correlation ρ")
print(SEP)
print(f"  Black-Scholes reference: {C_bs:.4f}")
print()
rho_rows = [{"ρ": rho, "Heston Call": round(p, 4),
             "vs BS": round(p - C_bs, 4)}
            for rho, p in zip(rho_vals, heston_rho)]
print(pd.DataFrame(rho_rows).to_string(index=False))

print(f"""
  As ρ increases from −0.9 toward 0, the call price rises monotonically.
  A strongly negative ρ amplifies downside scenarios: when the stock
  falls, variance surges, making recovery less likely and the call
  payoff lower in expectation. At ρ = 0 the stock and vol are
  uncorrelated and the Heston price approaches the BS level.
""")

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Chart 1 — Sensitivity to ρ
ax1.plot(rho_vals, heston_rho, "o-", lw=2, color="#059669",
         markersize=8, markerfacecolor="white", markeredgewidth=2,
         label="Heston call")
ax1.axhline(C_bs, color="#dc2626", ls="--", lw=1.5,
            label=f"Black-Scholes: {C_bs:.4f}")
ax1.set_xlabel("Correlation ρ (stock-vol)")
ax1.set_ylabel("Call Price (USD)")
ax1.set_title("Heston Sensitivity to ρ", fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.4)

# Chart 2 — Sensitivity to ξ (vol-of-vol)
xi_vals    = np.linspace(0.05, 0.80, 20)
heston_xi  = [heston_call(S0, K, T, r, xi=x) for x in xi_vals]

ax2.plot(xi_vals, heston_xi, "o-", lw=2, color="#7c3aed",
         markersize=6, markerfacecolor="white", markeredgewidth=2,
         label="Heston call")
ax2.axhline(C_bs, color="#dc2626", ls="--", lw=1.5,
            label=f"Black-Scholes: {C_bs:.4f}")
ax2.axvline(XI, color="gray", ls=":", lw=1.2, label=f"Base ξ = {XI}")
ax2.set_xlabel("Vol-of-Vol ξ")
ax2.set_ylabel("Call Price (USD)")
ax2.set_title("Heston Sensitivity to ξ (Vol-of-Vol)", fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.4)

plt.suptitle("Heston Stochastic Volatility Model — Sensitivity Analysis",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("heston_sensitivity.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# HESTON vs BLACK-SCHOLES DISCUSSION
# ─────────────────────────────────────────────────────────────────────────────

print(f"""
{SEP}
  Heston vs Black-Scholes
{SEP}

Why Heston is more realistic:
  Black-Scholes assumes volatility is constant, which contradicts what
  we observe in financial markets. Equity volatility clusters in time
  (high-vol periods follow high-vol periods), spikes during drawdowns,
  and gradually reverts toward a long-run average. Heston captures all
  three phenomena through five parameters:
    v0  — where volatility starts
    κ   — how quickly it mean-reverts (half-life ≈ ln2/κ)
    θ   — where it ultimately settles
    ξ   — how erratically it moves (vol-of-vol)
    ρ   — whether stock drops amplify vol spikes (leverage effect)

  The Heston model is also consistent with the implied volatility smile
  observed in options markets — something a flat BS σ can never reproduce.

One practical difficulty — calibration:
  Fitting all five parameters jointly to market option prices is a
  nonlinear optimisation problem. It is computationally expensive,
  can converge to local minima, and is sensitive to starting values.
  Small differences in calibrated parameters produce materially different
  prices, making Heston harder to deploy reliably in production compared
  to the one-parameter Black-Scholes.
""")
