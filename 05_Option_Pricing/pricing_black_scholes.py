####################     Northeastern University                                    ####################
####################     FINA 6339: Quantitative Portfolio Management               ####################
####################     Homework 3 — Black-Scholes Option Pricing                 ####################
####################     Prof. Dr. Milivoje Davidovic                               ####################
####################     Aluno: Luan Ferreira de Souza                              ####################

"""
Prices a European call option using the Black-Scholes (1973) closed-form
analytical solution under the assumptions of constant volatility and
continuously compounded risk-free rate.

Black-Scholes formula:
  C = S₀·N(d₁) − K·e^(−rT)·N(d₂)

  d₁ = [ln(S₀/K) + (r + ½σ²)T] / (σ√T)
  d₂ = d₁ − σ√T

where N(·) is the standard normal CDF.

Greeks computed:
  Δ (Delta)  = N(d₁)                         — price sensitivity to S
  Γ (Gamma)  = φ(d₁) / (S₀σ√T)              — delta sensitivity to S
  V (Vega)   = S₀·φ(d₁)·√T                  — price sensitivity to σ
  Θ (Theta)  = −[S₀·φ(d₁)·σ/(2√T)] − r·K·e^(−rT)·N(d₂)
  ρ (Rho)    = K·T·e^(−rT)·N(d₂)

Output:
  - Call price and full Greeks table
  - BS call price as a function of σ (vol sensitivity)
  - BS call price as a function of S₀ (spot sensitivity)

Parameters:
  S0 = 100, K = 105, T = 1, r = 0.04, σ = 0.30

Requires: numpy, pandas, matplotlib, scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

S0    = 100       # current stock price
K     = 105       # strike price (slightly OTM)
T     = 1         # time to maturity (years)
r     = 0.04      # continuously compounded risk-free rate
sigma = 0.30      # annualised volatility (constant under BS)

# ─────────────────────────────────────────────────────────────────────────────
# PRICING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def bs_call(S0, K, T, r, sigma):
    """
    European call price via Black-Scholes closed-form formula.

    Parameters
    ----------
    S0    : float — current stock price
    K     : float — strike price
    T     : float — time to maturity (years)
    r     : float — continuously compounded risk-free rate
    sigma : float — annualised volatility

    Returns
    -------
    float : call price
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_greeks(S0, K, T, r, sigma):
    """
    Compute the five standard Black-Scholes Greeks for a European call.

    Parameters
    ----------
    S0, K, T, r, sigma : standard option parameters

    Returns
    -------
    dict : {Delta, Gamma, Vega, Theta, Rho}
    """
    d1    = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2    = d1 - sigma * np.sqrt(T)
    phi   = norm.pdf(d1)           # standard normal PDF at d1

    delta = norm.cdf(d1)
    gamma = phi / (S0 * sigma * np.sqrt(T))
    vega  = S0 * phi * np.sqrt(T)
    theta = (-(S0 * phi * sigma) / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2))
    rho   = K * T * np.exp(-r * T) * norm.cdf(d2)

    return {"Delta": delta, "Gamma": gamma, "Vega": vega,
            "Theta": theta, "Rho": rho}

# ─────────────────────────────────────────────────────────────────────────────
# BASE CASE
# ─────────────────────────────────────────────────────────────────────────────

d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
C0 = bs_call(S0, K, T, r, sigma)
greeks = bs_greeks(S0, K, T, r, sigma)

SEP = "=" * 65
print(f"\n{SEP}")
print("  Black-Scholes — Parameter Summary")
print(SEP)
print(f"  S0={S0}  |  K={K}  |  T={T}  |  r={r}  |  σ={sigma}")
print(f"  d₁ = {d1:.4f}  |  d₂ = {d2:.4f}")
print(f"  N(d₁) = {norm.cdf(d1):.4f}  |  N(d₂) = {norm.cdf(d2):.4f}")
print(SEP)

print(f"\n{SEP}")
print("  Results")
print(SEP)
print(f"  Call price : {C0:.4f}")

print(f"\n{SEP}")
print("  Option Greeks")
print(SEP)
greek_rows = [{"Greek": name, "Symbol": sym, "Value": round(val, 4), "Interpretation": interp}
    for (name, sym, interp), val in zip([
        ("Delta", "Δ", "Change in call price per $1 move in S"),
        ("Gamma", "Γ", "Change in Delta per $1 move in S"),
        ("Vega",  "V", "Change in call price per 1pp move in σ (÷100)"),
        ("Theta", "Θ", "Daily time decay in call price (÷365)"),
        ("Rho",   "ρ", "Change in call price per 1pp move in r (÷100)"),
    ], greeks.values())]
print(pd.DataFrame(greek_rows).to_string(index=False))

print(f"""
Interpretation:
  The call is slightly out-of-the-money (S₀={S0} < K={K}), reflected
  in Δ = {greeks['Delta']:.4f} — for every $1 rise in the stock the call gains
  roughly {greeks['Delta']*100:.0f} cents. N(d₂) = {norm.cdf(d2):.4f} is the risk-neutral
  probability of finishing in the money at expiry.

  Vega = {greeks['Vega']:.4f} means a 1 percentage-point increase in volatility
  raises the call price by ~${greeks['Vega']/100:.2f}. This confirms that higher
  uncertainty always benefits the call buyer — the plot below shows
  this relationship is strictly increasing and convex in σ.
""")

# ─────────────────────────────────────────────────────────────────────────────
# SENSITIVITY CHARTS
# ─────────────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Chart 1 — Call price vs volatility
sigmas    = np.linspace(0.05, 0.80, 200)
bs_vs_vol = [bs_call(S0, K, T, r, s) for s in sigmas]

ax1.plot(sigmas, bs_vs_vol, lw=2, color="#1d4ed8")
ax1.axvline(sigma, color="gray",   ls=":", lw=1.2, label=f"σ = {sigma} (base)")
ax1.axhline(C0,    color="#dc2626", ls="--", lw=1.0,
            label=f"C₀ = {C0:.4f}")
ax1.set_xlabel("Volatility (σ)")
ax1.set_ylabel("Call Price (USD)")
ax1.set_title("Call Price vs Volatility", fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.4)

# Chart 2 — Call price vs spot price
spots     = np.linspace(60, 160, 200)
bs_vs_S   = [bs_call(s, K, T, r, sigma) for s in spots]
intrinsic = np.maximum(spots - K, 0)

ax2.plot(spots, bs_vs_S,   lw=2, color="#1d4ed8", label="BS call price")
ax2.plot(spots, intrinsic, lw=1.5, ls="--", color="#059669",
         label="Intrinsic value max(S−K, 0)")
ax2.axvline(S0, color="gray", ls=":", lw=1.2, label=f"S₀ = {S0}")
ax2.axvline(K,  color="gray", ls=":", lw=1.2, label=f"K  = {K}")
ax2.set_xlabel("Stock Price S₀ (USD)")
ax2.set_ylabel("Call Price (USD)")
ax2.set_title("Call Price vs Spot (Moneyness)", fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.4)

plt.suptitle("Black-Scholes Sensitivity Analysis", fontsize=13,
             fontweight="bold")
plt.tight_layout()
plt.savefig("bs_sensitivity.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# MONEYNESS TABLE
# ─────────────────────────────────────────────────────────────────────────────

strike_range = [80, 90, 95, 100, 105, 110, 115, 120]
rows = []
for k in strike_range:
    c = bs_call(S0, k, T, r, sigma)
    moneyness = "ITM" if k < S0 else ("ATM" if k == S0 else "OTM")
    rows.append({"K": k, "Moneyness": moneyness, "Call Price": round(c, 4)})

print(f"\n{SEP}")
print("  BS Call Price by Strike (Moneyness Table)")
print(SEP)
print(f"  S₀ = {S0}  |  T = {T}  |  r = {r}  |  σ = {sigma}")
print()
print(pd.DataFrame(rows).to_string(index=False))
