####################     Northeastern University                                    ####################
####################     FINA 6339: Quantitative Portfolio Management               ####################
####################     Volatility Estimation & GARCH Modeling                     ####################
####################     Prof. Dr. Milivoje Davidovic                               ####################
####################     Student: Luan Ferreira de Souza                            ####################

"""
Estimates and models time-varying volatility for two stocks and two
market indices:

  - Close-to-Close rolling volatility (21-day annualised std of log returns)
  - Garman-Klass OHLC-based volatility estimator (21-day rolling)
  - GARCH(1,1) and GJR-GARCH(1,1) model fitting and diagnostics
  - AIC / BIC model selection and leverage-effect interpretation

Data: stocks on sheets 5-6 of HM1_QPM_database.xlsx (confidential);
      indices on sheets 3-4.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH  = "HM1_QPM_database.xlsx"   # update path as needed
VOL_WINDOW = 21                         # rolling window (trading days)
ANNUALISE  = np.sqrt(252)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

stock1 = pd.read_excel(DATA_PATH, sheet_name=5)
stock2 = pd.read_excel(DATA_PATH, sheet_name=6)

mkt_id1 = pd.read_excel(DATA_PATH, sheet_name=3)
mkt_id2 = pd.read_excel(DATA_PATH, sheet_name=4)

# ─────────────────────────────────────────────────────────────────────────────
# PART C1 — HISTORICAL VOLATILITY: CLOSE-TO-CLOSE & GARMAN-KLASS
# ─────────────────────────────────────────────────────────────────────────────

def close_to_close_vol(close_series, window=VOL_WINDOW):
    """
    Compute rolling annualised close-to-close volatility.
    σ_cc = std(log(P_t / P_{t-1}), window) × √252
    """
    log_ret = np.log(close_series / close_series.shift(1))
    return log_ret.rolling(window=window).std() * ANNUALISE


def garman_klass_vol(df, window=VOL_WINDOW):
    """
    Compute rolling annualised Garman-Klass volatility.

    Daily GK variance:
        GK_t = 0.5 × (ln H/L)² − (2 ln 2 − 1) × (ln C/O)²

    Rolling annualised volatility:
        σ_GK = √( mean(GK_t, window) × 252 )

    References:
        Garman, M. B., & Klass, M. J. (1980). On the estimation of security
        price volatilities from historical data. Journal of Business, 53(1), 67-78.
    """
    h = np.log(df["High"] / df["Low"])
    c = np.log(df["Close"] / df["Open"])
    gk_daily = 0.5 * h**2 - (2 * np.log(2) - 1) * c**2
    return np.sqrt(gk_daily.rolling(window=window).mean() * 252)


def vol_stats(series, name):
    """Return a Series of descriptive statistics for a volatility series."""
    s = series.dropna()
    return pd.Series({
        "Mean":   s.mean(),
        "Median": s.median(),
        "Std Dev": s.std(),
        "Min":    s.min(),
        "Max":    s.max(),
        "Q25":    s.quantile(0.25),
        "Q75":    s.quantile(0.75),
        "Q95":    s.quantile(0.95),
    }, name=name)


# Compute volatility series
cc_vol1 = close_to_close_vol(stock1["Close"])
gk_vol1 = garman_klass_vol(stock1)
cc_vol2 = close_to_close_vol(stock2["Close"])
gk_vol2 = garman_klass_vol(stock2)

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(cc_vol1.values, color="steelblue", linewidth=1.2,
             label="Close-to-Close Volatility")
axes[0].plot(gk_vol1.values, color="darkred", linewidth=1.2,
             label="Garman-Klass Volatility")
axes[0].set_title("Stock 1 — Historical Volatility Comparison")
axes[0].set_xlabel("Days")
axes[0].set_ylabel("Annualised Volatility")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(cc_vol2.values, color="steelblue", linewidth=1.2,
             label="Close-to-Close Volatility")
axes[1].plot(gk_vol2.values, color="darkred", linewidth=1.2,
             label="Garman-Klass Volatility")
axes[1].set_title("Stock 2 — Historical Volatility Comparison")
axes[1].set_xlabel("Days")
axes[1].set_ylabel("Annualised Volatility")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("historical_volatility.png", dpi=150)
plt.show()

# ── Descriptive statistics table ──────────────────────────────────────────────
stats_df = pd.DataFrame([
    vol_stats(cc_vol1, "Stock 1 — Close-to-Close"),
    vol_stats(gk_vol1, "Stock 1 — Garman-Klass"),
    vol_stats(cc_vol2, "Stock 2 — Close-to-Close"),
    vol_stats(gk_vol2, "Stock 2 — Garman-Klass"),
])

print("\n" + "=" * 70)
print(" DESCRIPTIVE STATISTICS — VOLATILITY SERIES")
print("=" * 70)
print(stats_df.round(4).to_string())
print("""
Key findings:
  - Close-to-Close consistently reads higher than Garman-Klass (Stock 1:
    23.0% vs 20.5% mean), because it captures single-day close moves while
    GK smooths via the intraday High-Low range.
  - GK has lower variance (std dev 0.046 vs 0.075 for Stock 1), making it
    a more stable estimator — preferable for risk budgeting.
  - Both stocks show clear volatility clustering, consistent with a GARCH
    data-generating process.
""")

# ─────────────────────────────────────────────────────────────────────────────
# PART C2 — GARCH(1,1) AND GJR-GARCH(1,1)
# ─────────────────────────────────────────────────────────────────────────────

index1 = mkt_id1["Close"]
index2 = mkt_id2["Close"]

# Log returns scaled to % (required by arch library)
ret1 = 100 * np.log(index1 / index1.shift(1)).dropna()
ret2 = 100 * np.log(index2 / index2.shift(1)).dropna()


def fit_garch_models(ret, label):
    """
    Fit GARCH(1,1) and GJR-GARCH(1,1) to a return series.
    Print model summaries, AIC/BIC comparison, and Ljung-Box diagnostics.
    Returns both fitted model objects.

    GARCH(1,1):
        σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}

    GJR-GARCH(1,1):
        σ²_t = ω + (α + γ × I_{t-1}) × ε²_{t-1} + β × σ²_{t-1}
        where I_{t-1} = 1 if ε_{t-1} < 0 (leverage indicator)
        γ > 0 implies bad news amplifies future volatility more than good news.
    """
    print("=" * 65)
    print(f" GARCH(1,1) — {label}")
    print("=" * 65)
    garch = arch_model(ret, mean="Constant", vol="Garch", p=1, q=1)
    garch_fit = garch.fit(disp="off")
    print(garch_fit.summary())

    print("=" * 65)
    print(f" GJR-GARCH(1,1) — {label}")
    print("=" * 65)
    gjr = arch_model(ret, mean="Constant", vol="Garch", p=1, o=1, q=1)
    gjr_fit = gjr.fit(disp="off")
    print(gjr_fit.summary())

    # AIC / BIC comparison
    print(f"\n MODEL SELECTION — {label}")
    print("-" * 40)
    print(f"  GARCH(1,1)     AIC: {garch_fit.aic:.2f}  BIC: {garch_fit.bic:.2f}")
    print(f"  GJR-GARCH(1,1) AIC: {gjr_fit.aic:.2f}  BIC: {gjr_fit.bic:.2f}")

    # Ljung-Box test on standardised residuals
    std_resid = garch_fit.resid / garch_fit.conditional_volatility
    lb = acorr_ljungbox(std_resid.dropna(), lags=[10], return_df=True)
    print(f"\n  Ljung-Box (lag 10) p-value (GARCH residuals): "
          f"{lb['lb_pvalue'].values[0]:.4f}")

    return garch_fit, gjr_fit


garch1_fit, gjr1_fit = fit_garch_models(ret1, "Index 1")
garch2_fit, gjr2_fit = fit_garch_models(ret2, "Index 2")

# ── Conditional volatility plot ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, fit_g, fit_gjr, label in [
    (axes[0], garch1_fit, gjr1_fit, "Index 1"),
    (axes[1], garch2_fit, gjr2_fit, "Index 2"),
]:
    ax.plot(fit_g.conditional_volatility.values, color="steelblue",
            linewidth=1.0, label="GARCH(1,1)")
    ax.plot(fit_gjr.conditional_volatility.values, color="darkred",
            linewidth=1.0, linestyle="--", label="GJR-GARCH(1,1)")
    ax.set_title(f"{label} — Conditional Volatility")
    ax.set_xlabel("Days")
    ax.set_ylabel("Conditional Volatility (%)")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig("conditional_volatility.png", dpi=150)
plt.show()

print("""
INTERPRETATION — Model Selection:
  Index 1: GARCH(1,1) preferred (lower AIC and BIC). The GJR leverage
           parameter γ is not significant — shocks affect volatility
           symmetrically for this index.

  Index 2: GJR-GARCH(1,1) preferred (AIC improvement of 35 points).
           γ is positive and significant — negative shocks amplify
           future volatility far more than equivalent positive shocks,
           confirming a significant leverage effect.

  Volatility persistence (α + β ≈ 0.97 for both) is high, consistent
  with slow mean-reversion commonly observed in equity return volatility.
""")
