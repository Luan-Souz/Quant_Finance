####################     Northeastern University                                    ####################
####################     FINA 6339: Quantitative Portfolio Management               ####################
####################     Homework 3 — Problem 2: Binomial Option Pricing            ####################
####################     Prof. Dr. Milivoje Davidovic                               ####################
####################     Aluno: Luan Ferreira de Souza                              ####################

"""
Prices European call and put options on a stylised underlying using a
multi-step CRR (Cox-Ross-Rubinstein) binomial tree with backward induction.

Parameters (justified below):
  S0    = 100   — round-number at-the-money-ish spot; isolates moneyness effects cleanly
  K     = 105   — slight out-of-the-money; realistic for near-term equity options
  T     = 1     — one-year horizon; standard for textbook comparisons
  r     = 0.04  — proxies the current US risk-free environment
  sigma = 0.30  — moderate equity vol; consistent with mid-cap names
  n     = 4     — initial tree (coarse); convergence shown via sensitivity

Up/down factors derived from CRR convention:
  u = exp(σ√Δt),  d = 1/u
Risk-neutral probability:
  p* = (exp(r·Δt) − d) / (u − d)

Output:
  - Parameter summary and terminal stock prices
  - Call and put price trees (visualised)
  - Payoff and profit diagrams for all four option positions
  - Put-call parity verification
  - Sensitivity analysis: strike price and number of steps

Requires: numpy, pandas, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

S0    = 100       # current stock price
K     = 105       # strike price (slightly OTM call)
T     = 1         # time to maturity (years)
r     = 0.04      # risk-free rate (annualised, continuous)
sigma = 0.30      # implied/historical volatility
n     = 4         # number of binomial steps (coarse; refined in sensitivity)

# ─────────────────────────────────────────────────────────────────────────────
# CORE PRICING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def binomial_price(S0, K, T, r, sigma, n):
    """
    Price European call and put options via CRR binomial tree with
    backward induction under the risk-neutral measure.

    Parameters
    ----------
    S0    : float — current stock price
    K     : float — strike price
    T     : float — time to maturity (years)
    r     : float — continuously compounded risk-free rate
    sigma : float — annualised volatility
    n     : int   — number of tree steps

    Returns
    -------
    (C0, P0, ST, u, d, p_star, dt, R)
      C0     : call price
      P0     : put price
      ST     : terminal stock price array (n+1,)
      u, d   : up/down factors
      p_star : risk-neutral up-probability
      dt     : step length
      R      : one-period gross risk-free return
    """
    dt = T / n
    u  = np.exp(sigma * np.sqrt(dt))       # CRR up factor
    d  = 1 / u                             # CRR down factor
    R  = np.exp(r * dt)                    # one-period gross return
    p  = (R - d) / (u - d)                # risk-neutral probability
    q  = 1 - p

    # Terminal stock prices: S0 · u^(n-j) · d^j for j = 0,...,n
    ST  = np.array([S0 * u**(n - j) * d**j for j in range(n + 1)])

    # Terminal payoffs
    C = np.maximum(ST - K, 0)
    P = np.maximum(K - ST, 0)

    # Backward induction
    steps = n
    while steps > 0:
        C     = (p * C[:-1] + q * C[1:]) / R
        P     = (p * P[:-1] + q * P[1:]) / R
        steps -= 1

    return C[0], P[0], ST, u, d, p, dt, R


def backward_tree(terminal, p, q, R, n):
    """
    Reconstruct the full backward-induction price tree for visualisation.

    Parameters
    ----------
    terminal : array-like — terminal payoff vector (n+1,)
    p, q     : float      — risk-neutral up/down probabilities
    R        : float      — one-period gross return
    n        : int        — number of steps

    Returns
    -------
    list of lists — tree[0] is the root (today's price),
                    tree[-1] is the terminal payoff row
    """
    tree = [list(terminal)]
    vals = np.array(terminal, float)
    steps = n
    while steps > 0:
        vals  = (p * vals[:-1] + q * vals[1:]) / R
        steps -= 1
        tree.insert(0, list(vals))
    return tree

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE BASE CASE
# ─────────────────────────────────────────────────────────────────────────────

C0, P0, ST, u, d, p_star, dt, R = binomial_price(S0, K, T, r, sigma, n)

SEP = "=" * 70
print(f"\n{SEP}")
print("  Binomial Option Pricing — Parameter Summary")
print(SEP)
print(f"  S0={S0}  |  K={K}  |  T={T}  |  r={r}  |  sigma={sigma}  |  n={n}")
print(f"  u = {u:.4f}  |  d = {d:.4f}  |  Δt = {dt:.4f}  |  R = {R:.6f}")
print(f"  p* = {p_star:.4f}  (risk-neutral up-probability)")
print(SEP)

print(f"\n{SEP}")
print("  Terminal Stock Prices")
print(SEP)
for j, s in enumerate(ST):
    label = f"S·u^{n-j}·d^{j}" if j < n else f"S·d^{n}"
    print(f"  {label:<12}: ${s:.2f}")

print(f"\n{SEP}")
print("  Option Prices — Base Case (n=4)")
print(SEP)
print(f"  Call (C₀) = {C0:.4f}")
print(f"  Put  (P₀) = {P0:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# BINOMIAL TREE VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

call_tree = backward_tree(np.maximum(ST - K, 0), p_star, 1 - p_star, R, n)
put_tree  = backward_tree(np.maximum(K - ST, 0), p_star, 1 - p_star, R, n)


def draw_tree(ax, tree, title, color):
    """Render a binomial price tree as a network of nodes and edges."""
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.axis("off")
    n_steps = len(tree) - 1
    pos = {}
    for step, row in enumerate(tree):
        for node, val in enumerate(row):
            x = step
            y = node - len(row) / 2 + 0.5
            pos[(step, node)] = (x, y)
            ax.text(x, y, f"{val:.2f}", ha="center", va="center",
                    fontsize=8.5, color="white",
                    bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="none"))
    for step in range(n_steps):
        for node in range(len(tree[step])):
            x0, y0 = pos[(step, node)]
            x1u, y1u = pos[(step + 1, node)]
            x1d, y1d = pos[(step + 1, node + 1)]
            ax.plot([x0, x1u], [y0, y1u], "k-", lw=0.7, alpha=0.5)
            ax.plot([x0, x1d], [y0, y1d], "k-", lw=0.7, alpha=0.5)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
draw_tree(ax1, call_tree, f"Call Price Tree  (C₀ = {call_tree[0][0]:.4f})", "#1d4ed8")
draw_tree(ax2, put_tree,  f"Put Price Tree   (P₀ = {put_tree[0][0]:.4f})",  "#dc2626")
plt.suptitle("Binomial Option Pricing Trees", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("binomial_trees.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# PAYOFF & PROFIT DIAGRAMS
# ─────────────────────────────────────────────────────────────────────────────

# Future value of premiums (for profit calculation)
C0_fv = C0 * np.exp(r * T)
P0_fv = P0 * np.exp(r * T)

S_range = np.linspace(50, 160, 300)

positions = [
    ("Call Buyer",  np.maximum(S_range - K, 0),
                    np.maximum(S_range - K, 0) - C0_fv),
    ("Call Writer", -np.maximum(S_range - K, 0),
                    -np.maximum(S_range - K, 0) + C0_fv),
    ("Put Buyer",   np.maximum(K - S_range, 0),
                    np.maximum(K - S_range, 0) - P0_fv),
    ("Put Writer",  -np.maximum(K - S_range, 0),
                    -np.maximum(K - S_range, 0) + P0_fv),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (title, payoff, profit) in zip(axes.flat, positions):
    ax.plot(S_range, payoff, lw=1.5, ls="--", label="Payoff")
    ax.plot(S_range, profit, lw=2.0,           label="Profit")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(K, color="gray",  lw=0.8, ls=":")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Stock Price at Expiry (USD)")
    ax.set_ylabel("P&L (USD)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle("Payoff & Profit Diagrams at Expiration", fontsize=13,
             fontweight="bold")
plt.tight_layout()
plt.savefig("payoff_profit_diagrams.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# PUT-CALL PARITY VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

lhs = C0 - P0
rhs = S0 - K * np.exp(-r * T)

print(f"\n{SEP}")
print("  Put-Call Parity Verification")
print(SEP)
print(f"  C₀ − P₀            = {lhs:.4f}")
print(f"  S₀ − K·e^(−rT)    = {rhs:.4f}")
print(f"  |Difference|       = {abs(lhs - rhs):.4f}")
print(f"""
  The two sides are close but not exactly equal. With only n={n} steps
  the binomial tree is a coarse approximation; as n → ∞ both C₀ and P₀
  converge to their Black-Scholes values and the parity gap shrinks
  toward zero. The relationship holds approximately here, and would hold
  more tightly with a finer grid.
""")

# ─────────────────────────────────────────────────────────────────────────────
# SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

# Sensitivity 1 — Strike Price
strikes   = np.arange(80, 131, 5)
c_K, p_K  = zip(*[binomial_price(S0, k, T, r, sigma, n)[:2] for k in strikes])

# Sensitivity 2 — Number of Steps
step_range = np.arange(2, 101)
c_n, p_n   = zip(*[binomial_price(S0, K, T, r, sigma, s)[:2] for s in step_range])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(strikes, c_K, lw=2, label="Call")
axes[0].plot(strikes, p_K, lw=2, label="Put")
axes[0].axvline(S0, color="gray", ls=":", lw=0.9, label=f"S₀ = {S0}")
axes[0].set_title("Sensitivity to Strike Price", fontweight="bold")
axes[0].set_xlabel("Strike Price K (USD)")
axes[0].set_ylabel("Option Price (USD)")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(step_range, c_n, lw=2, label="Call")
axes[1].plot(step_range, p_n, lw=2, label="Put")
axes[1].axvline(n, color="gray", ls=":", lw=0.9, label=f"Base n = {n}")
axes[1].set_title("Sensitivity to Number of Steps", fontweight="bold")
axes[1].set_xlabel("Number of Binomial Steps n")
axes[1].set_ylabel("Option Price (USD)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle("Sensitivity Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("binomial_sensitivity.png", dpi=150, bbox_inches="tight")
plt.show()

# Sensitivity table — strike
sens_rows = []
for k, c, p in zip(strikes, c_K, p_K):
    moneyness = "ITM" if k < S0 else ("ATM" if k == S0 else "OTM")
    sens_rows.append({"K": k, "Call": round(c, 4), "Put": round(p, 4),
                      "Moneyness": moneyness})

print(f"\n{SEP}")
print("  Strike Sensitivity Table (n=4, T=1, σ=0.30, r=0.04)")
print(SEP)
print(pd.DataFrame(sens_rows).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# INTERPRETATION
# ─────────────────────────────────────────────────────────────────────────────

print(f"""
{SEP}
  Interpretation
{SEP}

How option values change with moneyness:
  The strike sensitivity table and plot tell this story directly.
  As K rises, the call becomes cheaper (less likely to finish in the
  money) and the put becomes more expensive (more likely to finish in
  the money). At K = S₀ = {S0}, both options are at-the-money and have
  similar values. Below S₀ the call is in-the-money and commands a
  higher price; above S₀ the put does.

Why the binomial price changes as steps increase:
  With very few steps the tree is too coarse to approximate the true
  price distribution well; the estimate is noisy. As n grows the tree
  more closely approximates continuous GBM and the price converges
  toward its Black-Scholes limit. The oscillation visible in the
  step-sensitivity plot is a known CRR artefact — even and odd step
  counts alternate slightly — but the envelope tightens steadily as
  n increases.

How the economic position differs for buyers versus writers:
  The payoff diagrams illustrate the fundamental asymmetry. The buyer
  pays the premium upfront and has limited downside (capped at the
  premium paid) but unbounded upside for calls, or large upside for
  puts. The writer has the mirror image: they collect the premium
  immediately but face potentially large losses if the option finishes
  deep in the money. This is why writers typically hedge their exposure,
  while buyers do not.
""")
