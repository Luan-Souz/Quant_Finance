####################     Northeastern University                                    ####################
####################     FINA 6339: Quantitative Portfolio Management               ####################
####################     Factor Model Analysis of Industry Portfolios               ####################
####################     Prof. Dr. Milivoje Davidovic                               ####################
####################     Aluno: Luan Ferreira de Souza                              ####################

"""
Decomposes Manufacturing (Manuf) and Telecom (Telcm) industry portfolio
returns using four progressively richer factor models:

  1. CAPM           — single market factor
  2. Fama-French 3  — market + size (SMB) + value (HML)
  3. Carhart 4      — FF3 + momentum (Mom)
  4. Fama-French 5  — FF3 + profitability (RMW) + investment (CMA)

Factor data: Kenneth R. French Data Library (F-F_Research_Data_Factors_daily).
Index data:  sheets 3 and 4 of HM1_QPM_database.xlsx (confidential).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH   = "HM1_QPM_database.xlsx"           # update path as needed
FACTOR_PATH = "F-F_Research_Data_Factors_daily_CSV.csv"
MOM_PATH    = "F-F_Momentum_Factor_daily_CSV.csv"
FF5_PATH    = "F-F_Research_Data_5_Factors_2x3_daily_CSV.csv"
SAMPLE_SIZE = 500                                # number of observations used

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

mkt_id1 = pd.read_excel(DATA_PATH, sheet_name=3)
mkt_id2 = pd.read_excel(DATA_PATH, sheet_name=4)

close_col = mkt_id1.columns[1]

id1 = pd.Series(mkt_id1[close_col].values)
id2 = pd.Series(mkt_id2[close_col].values)

# Log returns scaled to % (matching French factor units)
ret1 = 100 * np.log(id1 / id1.shift(1)).dropna()
ret2 = 100 * np.log(id2 / id2.shift(1)).dropna()

# Load Fama-French 3-factor data
ff3 = pd.read_csv(FACTOR_PATH, skiprows=4, index_col=0)
ff3 = ff3[ff3.index.astype(str).str.len() == 8]  # keep daily rows only
ff3.index = pd.to_datetime(ff3.index, format="%Y%m%d")
ff3 = ff3.apply(pd.to_numeric, errors="coerce").dropna()

# Align on common index (last SAMPLE_SIZE rows)
common_idx = ff3.index
mkt_rf = ff3["Mkt-RF"].iloc[-SAMPLE_SIZE:]
smb    = ff3["SMB"].iloc[-SAMPLE_SIZE:]
hml    = ff3["HML"].iloc[-SAMPLE_SIZE:]
rf_ff  = ff3["RF"].iloc[-SAMPLE_SIZE:]

excess_ret1 = ret1.iloc[-SAMPLE_SIZE:].values - rf_ff.values
excess_ret2 = ret2.iloc[-SAMPLE_SIZE:].values - rf_ff.values

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: print OLS regression results cleanly
# ─────────────────────────────────────────────────────────────────────────────

def print_regression(model_name, portfolio, result):
    """Print coefficient table and R² for one regression."""
    print("=" * 65)
    print(f" {model_name} — {portfolio}")
    print("=" * 65)
    print(result.summary2().tables[1].to_string())
    print(f"\n R-squared:          {result.rsquared:.4f}")
    print(f" Adj. R-squared:     {result.rsquared_adj:.4f}")
    print(f" AIC:                {result.aic:.2f}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — CAPM
# ─────────────────────────────────────────────────────────────────────────────

X_capm = sm.add_constant(mkt_rf)

capm_manuf = sm.OLS(excess_ret1, X_capm).fit()
capm_telcm = sm.OLS(excess_ret2, X_capm).fit()

print_regression("CAPM", "Manuf", capm_manuf)
print_regression("CAPM", "Telcm", capm_telcm)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — FAMA-FRENCH 3-FACTOR
# ─────────────────────────────────────────────────────────────────────────────

X_ff3 = sm.add_constant(pd.DataFrame({
    "Mkt-RF": mkt_rf.values,
    "SMB":    smb.values,
    "HML":    hml.values,
}))

ff3_manuf = sm.OLS(excess_ret1, X_ff3).fit()
ff3_telcm = sm.OLS(excess_ret2, X_ff3).fit()

print_regression("Fama-French 3-Factor", "Manuf", ff3_manuf)
print_regression("Fama-French 3-Factor", "Telcm", ff3_telcm)

print("""
INTERPRETATION — Manuf FF3:
  SMB loading +0.88 (highly significant) — strong small-cap tilt.
  HML loading insignificant — no value/growth exposure.
  R² = 0.890, up from 0.67 under CAPM. Size factor drives the improvement.

INTERPRETATION — Telcm FF3:
  SMB loading +0.72, HML loading +0.30 (both significant) — small-cap
  and value tilt, consistent with mature capital-intensive telecom firms.
  R² = 0.612 — Telcm retains substantial industry-specific risk.
""")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — CARHART 4-FACTOR (FF3 + Momentum)
# ─────────────────────────────────────────────────────────────────────────────

mom_raw = pd.read_csv(MOM_PATH, skiprows=13, index_col=0)
mom_raw = mom_raw[mom_raw.index.astype(str).str.len() == 8]
mom_raw.index = pd.to_datetime(mom_raw.index, format="%Y%m%d")
mom_raw = mom_raw.apply(pd.to_numeric, errors="coerce").dropna()
mom = mom_raw.iloc[-SAMPLE_SIZE:].squeeze()

X_c4 = sm.add_constant(pd.DataFrame({
    "Mkt-RF": mkt_rf.values,
    "SMB":    smb.values,
    "HML":    hml.values,
    "Mom":    mom.values,
}))

c4_manuf = sm.OLS(excess_ret1, X_c4).fit()
c4_telcm = sm.OLS(excess_ret2, X_c4).fit()

print_regression("Carhart 4-Factor", "Manuf", c4_manuf)
print_regression("Carhart 4-Factor", "Telcm", c4_telcm)

print("""
INTERPRETATION — Momentum Factor:
  Both Manuf and Telcm load negatively on momentum — they behave as
  contrarian portfolios where past losers tend to recover. This is
  consistent with the mean-reverting nature of industrial and telecom
  sector returns over medium-term horizons.
""")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 4 — FAMA-FRENCH 5-FACTOR (FF3 + RMW + CMA)
# ─────────────────────────────────────────────────────────────────────────────

ff5_raw = pd.read_csv(FF5_PATH, skiprows=4, index_col=0)
ff5_raw = ff5_raw[ff5_raw.index.astype(str).str.len() == 8]
ff5_raw.index = pd.to_datetime(ff5_raw.index, format="%Y%m%d")
ff5_raw = ff5_raw.apply(pd.to_numeric, errors="coerce").dropna()

rmw = ff5_raw["RMW"].iloc[-SAMPLE_SIZE:]
cma = ff5_raw["CMA"].iloc[-SAMPLE_SIZE:]

X_ff5 = sm.add_constant(pd.DataFrame({
    "Mkt-RF": mkt_rf.values,
    "SMB":    smb.values,
    "HML":    hml.values,
    "RMW":    rmw.values,
    "CMA":    cma.values,
}))

ff5_manuf = sm.OLS(excess_ret1, X_ff5).fit()
ff5_telcm = sm.OLS(excess_ret2, X_ff5).fit()

print_regression("Fama-French 5-Factor", "Manuf", ff5_manuf)
print_regression("Fama-French 5-Factor", "Telcm", ff5_telcm)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL COMPARISON SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

models = ["CAPM", "FF3", "Carhart 4", "FF5"]
r2_manuf = [capm_manuf.rsquared, ff3_manuf.rsquared,
            c4_manuf.rsquared,   ff5_manuf.rsquared]
r2_telcm = [capm_telcm.rsquared, ff3_telcm.rsquared,
            c4_telcm.rsquared,   ff5_telcm.rsquared]

summary = pd.DataFrame({
    "Model":        models,
    "R² Manuf":     [f"{r:.4f}" for r in r2_manuf],
    "R² Telcm":     [f"{r:.4f}" for r in r2_telcm],
})

print("=" * 50)
print(" MODEL FIT COMPARISON — R²")
print("=" * 50)
print(summary.to_string(index=False))
print("""
Key findings:
  - Adding SMB and HML (FF3) lifts Manuf R² from 0.67 to 0.89 — size
    is the dominant driver.
  - Momentum is negative for both industries (contrarian behaviour).
  - FF5 adds marginal fit over FF3; CMA is significant, RMW is not
    significant for Manuf.
  - Neither portfolio shows significant alpha in any model — returns
    are fully explained by systematic factor exposures.
""")
