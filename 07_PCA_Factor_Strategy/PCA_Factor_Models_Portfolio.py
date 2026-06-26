"""
####################   Northeastern University                   ####################
####################   FINA 6334: Empirical Methods in Finance   ####################
####################   PCA Factor Strategy                       ####################
####################   Student: Luan Ferreira de Souza          ####################

Finding Factors with PCA: A Statistical Alternative to Fama-French
===================================================================

This module builds a PCA-driven long-only equity portfolio from CRSP
monthly stock return data (NYSE / AMEX / NASDAQ common stocks, exchcd
∈ {1, 2, 3}, shrcd ∈ {10, 11}).  The pipeline follows four stages:

  Stage 1 — Return Matrix Construction
      Filters the CRSP universe, pivots to a (T × N) return matrix,
      and standardises each column to (μ = 0, σ = 1) using training-
      period statistics only.

  Stage 2 — PCA on the Training Period  (Jan 2010 – Dec 2024)
      Fits sklearn PCA on the standardised matrix; retains k = 4
      components (PC1–PC4) that together explain 46.30 % of total
      variance.  Produces a scree plot and a loadings table showing
      the five highest- and five lowest-loading stocks per component.

  Stage 3 — Factor Interpretation & Views
      PC1  Broad Market / Systematic Risk        → Positive view
      PC2  Defensive Consumption vs Innovation   → Neutral
      PC3  Semiconductor Investment Cycle        → Positive (low PC3)
      PC4  Innovation Platform vs Maturity       → Neutral

  Stage 4 — Portfolio Construction
      Scoring rule:  score = 0.5 × PC1_loading − 0.5 × PC3_loading
      Top 30 stocks by score form an equal-weighted, long-only
      portfolio submitted ahead of the live evaluation period.

Mathematical Notation
---------------------
Let R ∈ ℝ^{T×N} be the standardised return matrix.  PCA solves

    R = U Σ Vᵀ   (economy SVD)

where columns of V are the principal directions (loadings).  The
j-th PC score for stock i is  φ_{ij} = V_{ij}, the i-th element of
the j-th eigenvector.

Portfolio scoring:

    s_i = 0.5 · φ_{i,1}  −  0.5 · φ_{i,3}

Stocks are ranked by sᵢ; the top 30 receive equal weight 1/30.
"""

# ── Standard library ──────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition  import PCA

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
DATA_PATH        = "crsp_monthly_returns.csv"   # CRSP monthly return file
TRAIN_START      = "2010-01-01"
TRAIN_END        = "2024-12-31"
MIN_HISTORY_MOS  = 60                           # ≥ 60 months required
VALID_EXCHCD     = [1, 2, 3]                    # NYSE, AMEX, NASDAQ
VALID_SHRCD      = [10, 11]                     # common stocks only
N_COMPONENTS     = 4                            # retained PCs
PORTFOLIO_SIZE   = 30                           # equal-weighted stocks
TOP_N_LOADINGS   = 5                            # shown per component
PC_WEIGHTS       = {1: +0.5, 3: -0.5}          # scoring formula
COLORS           = ["#dc2626", "#d97706",
                    "#1d4ed8", "#059669",
                    "#7c3aed", "#0891b2"]
FIG_DPI          = 150
SEP              = "=" * 70

# ═══════════════════════════════════════════════════════════════════════
# STAGE 1 — RETURN MATRIX CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════

def load_and_filter(path: str) -> pd.DataFrame:
    """
    Load and filter the CRSP monthly return file.

    Parameters
    ----------
    path : str
        Path to the CRSP CSV with columns: date, permno, ticker,
        exchcd, shrcd, ret, dlret.

    Returns
    -------
    pd.DataFrame
        Filtered long-format frame with columns [date, ticker, ret].
    """
    df = pd.read_csv(path, parse_dates=["date"], low_memory=False)

    # Exchange and share-type filters
    df = df[df["exchcd"].isin(VALID_EXCHCD)]
    df = df[df["shrcd"].isin(VALID_SHRCD)]

    # Merge delisting returns where available (avoids survivorship bias)
    df["ret"] = np.where(
        df["dlret"].notna(),
        (1 + df["ret"].fillna(0)) * (1 + df["dlret"]) - 1,
        df["ret"]
    )

    # Training window
    mask = (df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)
    df = df[mask][["date", "ticker", "ret"]].dropna(subset=["ret"])
    return df


def build_return_matrix(df: pd.DataFrame,
                        min_history: int = MIN_HISTORY_MOS
                        ) -> pd.DataFrame:
    """
    Pivot long-format returns into a (T × N) matrix and drop
    stocks with fewer than `min_history` non-missing months.

    Parameters
    ----------
    df           : long-format return frame [date, ticker, ret]
    min_history  : minimum months of history required

    Returns
    -------
    pd.DataFrame
        Month-indexed return matrix; columns are ticker symbols.
    """
    R = df.pivot(index="date", columns="ticker", values="ret")
    counts = R.notna().sum()
    R = R[counts[counts >= min_history].index]
    R = R.sort_index()
    return R


def standardise(R: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """
    Standardise columns of R to (μ = 0, σ = 1) using training
    statistics only.

    Parameters
    ----------
    R : return matrix (T × N), NaN-filled rows are dropped.

    Returns
    -------
    Z      : standardised numpy array (T × N)
    scaler : fitted StandardScaler (for out-of-sample use)
    """
    R_filled = R.ffill().bfill()
    scaler   = StandardScaler()
    Z        = scaler.fit_transform(R_filled)
    return Z, scaler

# ═══════════════════════════════════════════════════════════════════════
# STAGE 2 — PCA
# ═══════════════════════════════════════════════════════════════════════

def run_pca(Z: np.ndarray,
            n_components: int = N_COMPONENTS) -> tuple[PCA, np.ndarray]:
    """
    Fit PCA on the standardised return matrix.

    Parameters
    ----------
    Z            : standardised return array (T × N)
    n_components : number of principal components to retain

    Returns
    -------
    pca      : fitted sklearn PCA object
    loadings : (N × n_components) array of component loadings
    """
    pca      = PCA(n_components=n_components)
    pca.fit(Z)
    loadings = pca.components_.T   # shape (N, k)
    return pca, loadings


def scree_plot(pca: PCA, save_path: str = "scree_plot.png") -> None:
    """
    Produce and save a scree plot with cumulative variance line.

    Parameters
    ----------
    pca       : fitted PCA object
    save_path : output file path
    """
    ev  = pca.explained_variance_ratio_ * 100
    cum = np.cumsum(ev)
    ks  = np.arange(1, len(ev) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()

    ax1.bar(ks, ev, color=COLORS[2], alpha=0.75, label="Individual")
    ax2.plot(ks, cum, color=COLORS[0], marker="o",
             linewidth=2, label="Cumulative")
    ax2.axhline(y=cum[N_COMPONENTS - 1], color=COLORS[0],
                linestyle="--", linewidth=1, alpha=0.6)

    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained (%)")
    ax2.set_ylabel("Cumulative Variance Explained (%)")
    ax1.set_xticks(ks)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.title("Scree Plot — PCA on CRSP Monthly Returns (2010–2024)",
              fontsize=11, fontweight="bold")
    fig.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()
    plt.close()


def loadings_table(loadings: np.ndarray,
                   tickers: list,
                   top_n: int = TOP_N_LOADINGS) -> pd.DataFrame:
    """
    Build a summary table of the highest and lowest loading stocks
    for each retained principal component.

    Parameters
    ----------
    loadings : (N × k) loading array
    tickers  : list of N ticker symbols
    top_n    : number of extreme stocks to display per PC

    Returns
    -------
    pd.DataFrame
        Multi-indexed summary with columns [Stock, Loading] for each PC.
    """
    tickers = list(tickers)
    rows    = []
    for j in range(loadings.shape[1]):
        col   = loadings[:, j]
        order = np.argsort(col)
        top   = order[-top_n:][::-1]
        bot   = order[:top_n]
        for rank, idx in enumerate(top, 1):
            rows.append({
                "PC": f"PC{j+1}", "Rank": f"High {rank}",
                "Ticker": tickers[idx], "Loading": round(col[idx], 3)
            })
        for rank, idx in enumerate(bot, 1):
            rows.append({
                "PC": f"PC{j+1}", "Rank": f"Low {rank}",
                "Ticker": tickers[idx], "Loading": round(col[idx], 3)
            })
    return pd.DataFrame(rows).set_index(["PC", "Rank"])

# ═══════════════════════════════════════════════════════════════════════
# STAGE 3 — FACTOR VIEWS (interpretations printed inline)
# ═══════════════════════════════════════════════════════════════════════

def print_factor_views(pca: PCA) -> None:
    """
    Print variance explained and factor-view summary to stdout.

    Parameters
    ----------
    pca : fitted PCA object
    """
    ev = pca.explained_variance_ratio_ * 100
    print(SEP)
    print("FACTOR VIEWS & ECONOMIC INTERPRETATION")
    print(SEP)

    views = f"""
PC1 — Broad Market / Systematic Risk  ({ev[0]:.2f}% variance explained)
    Positive view.  PC1 loads positively across virtually all stocks,
    capturing broad equity-market co-movement.  Technology and
    semiconductor names (AMAT 0.174, SNPS 0.170, ASML 0.169, KLAC
    0.168) carry the highest loadings, indicating strong sensitivity to
    economic growth and investment cycles.  Defensive names such as PDD
    (0.042) and REGN (0.051) have subdued loadings.  Continued AI
    investment and steady macro activity support a positive view.

PC2 — Defensive Consumption & Healthcare vs Innovation Growth  ({ev[1]:.2f}%)
    Neutral view.  High PC2 stocks — PEP (0.316), AMGN (0.312), MDLZ
    (0.308), GILD (0.238) — are consumer-staple and pharmaceutical firms
    with predictable earnings.  Negative PC2 stocks — CRWD (−0.247),
    COIN (−0.229), NVDA (−0.193) — are innovation-driven high-growth
    names.  Both segments may perform well depending on the macro
    environment; we do not tilt toward either extreme.

PC3 — Semiconductor Investment Cycle  ({ev[2]:.2f}%)
    Positive view on low-PC3 stocks.  PC3 differentiates semiconductor
    equipment firms (MU −0.219, AMAT −0.210, KLAC −0.210, LRCX −0.208)
    from software and diversified-tech firms (WDAY +0.273, CRWD +0.245).
    Continued growth in AI infrastructure and cloud capital expenditure
    supports semiconductor demand; we prefer stocks with negative PC3
    loadings, whose returns are most sensitive to that cycle.

PC4 — Innovation Platform vs Mature Stability  ({ev[3]:.2f}%)
    Neutral view.  High PC4 names (REGN 0.291, EQIX 0.271, AMZN 0.251,
    NFLX 0.189) depend on intellectual property and platform scalability.
    Negative PC4 names (MAR −0.265, ADI −0.249, BKNG −0.228) have
    stable, established operating structures.  Performance is largely
    idiosyncratic; we hold no directional view.
"""
    print(views)

# ═══════════════════════════════════════════════════════════════════════
# STAGE 4 — PORTFOLIO CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════

def build_portfolio(loadings: np.ndarray,
                    tickers: list,
                    pc_weights: dict = PC_WEIGHTS,
                    n_stocks: int     = PORTFOLIO_SIZE
                    ) -> pd.DataFrame:
    """
    Score and rank stocks using a linear combination of PC loadings,
    then select the top `n_stocks` for an equal-weighted portfolio.

    Scoring rule:
        score_i = Σ_j  w_j · φ_{i,j}
    where w_j is the signed weight assigned to PC j (positive =
    favour high loadings; negative = favour low loadings).

    Parameters
    ----------
    loadings   : (N × k) loading array
    tickers    : list of N ticker symbols
    pc_weights : dict mapping 1-indexed PC number → signed weight
    n_stocks   : number of stocks in the final portfolio

    Returns
    -------
    pd.DataFrame
        Portfolio frame sorted by score with columns:
        [Ticker, Score, PC1, PC2, PC3, PC4, Weight].
    """
    scores = np.zeros(loadings.shape[0])
    for pc_num, w in pc_weights.items():
        scores += w * loadings[:, pc_num - 1]

    df = pd.DataFrame({
        "Ticker": list(tickers),
        "Score" : scores,
    })
    for j in range(loadings.shape[1]):
        df[f"PC{j+1}"] = loadings[:, j]

    df = df.nlargest(n_stocks, "Score").reset_index(drop=True)
    df["Weight"] = 1.0 / n_stocks
    df = df.round({"Score": 4, "PC1": 4, "PC2": 4, "PC3": 4,
                   "PC4": 4, "Weight": 4})
    return df


def plot_portfolio(portfolio: pd.DataFrame,
                   save_path: str = "portfolio_scores.png") -> None:
    """
    Horizontal bar chart of portfolio scores ranked highest to lowest.

    Parameters
    ----------
    portfolio : portfolio DataFrame from build_portfolio()
    save_path : output file path
    """
    df = portfolio.sort_values("Score", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(df["Ticker"], df["Score"], color=COLORS[2], alpha=0.80)
    ax.set_xlabel("Portfolio Score  (0.5 × PC1 − 0.5 × PC3)")
    ax.set_title("PCA Portfolio — Stock Scores\n"
                 "Training: Jan 2010 – Dec 2024",
                 fontsize=11, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.show()
    plt.close()


def print_portfolio_summary(portfolio: pd.DataFrame,
                            pca: PCA) -> None:
    """
    Print a formatted portfolio summary to stdout.

    Parameters
    ----------
    portfolio : portfolio DataFrame from build_portfolio()
    pca       : fitted PCA object
    """
    ev = pca.explained_variance_ratio_ * 100
    print(SEP)
    print("PORTFOLIO CONSTRUCTION SUMMARY")
    print(SEP)
    print(portfolio[["Ticker", "Score", "PC1", "PC3"]].to_string(index=False))

    summary = f"""
{SEP}
PORTFOLIO LOGIC
{SEP}
Scoring rule:  score = 0.5 × PC1_loading − 0.5 × PC3_loading

This formula simultaneously selects for (i) broad market beta via PC1
and (ii) sensitivity to the semiconductor capital-expenditure cycle via
negative PC3.  The two PCs on which we hold neutral views — PC2 and
PC4 — receive zero weight, so the selection is driven entirely by the
two dimensions where we have directional conviction.

Top names confirm the thesis.  AMAT leads with PC1 = 0.174 and
PC3 = −0.210, giving score 0.192.  KLAC (0.189) and LRCX (0.186)
follow with comparable loading profiles.  MU carries the most negative
PC3 of any selected stock (−0.219), consistent with its role as the
most cycle-sensitive name in the portfolio.  ASML (0.183) combines a
top-tier PC1 loading with negative PC3, reflecting its monopoly
position in EUV lithography.

Not every stock is a semiconductor pure-play.  ADSK, ADBE, and CSCO
score well through high market beta (PC1 ≈ 0.14–0.17) with roughly
neutral PC3, adding diversification without diluting the core thesis.

Sector composition:  ~43% semiconductors (13 / 30 names).  This
concentration is a consequence of the scoring formula — PC3 loadings
are most negative for semiconductor equipment makers — not a manual
sector tilt.  Biotech names (GILD, REGN, VRTX, AMGN) and a few
defensive names provide a partial cushion.  The primary risk is a
sharp reversal in AI capex or a semiconductor inventory correction,
which would hit a large portion of the portfolio simultaneously.

Portfolio:   {len(portfolio)} stocks · equal-weighted (1/30 each)
Universe:    NYSE / AMEX / NASDAQ common stocks, ≥ 60 months history
Training:    Jan 2010 – Dec 2024
Variance:    PC1 {ev[0]:.2f}%  |  PC2 {ev[1]:.2f}%  |  PC3 {ev[2]:.2f}%  |  PC4 {ev[3]:.2f}%
             Combined: {sum(ev):.2f}%
"""
    print(summary)

# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    print(SEP)
    print("PCA FACTOR STRATEGY — FINA 6339 PROJECT 2")
    print(SEP)

    # ── Stage 1: data ─────────────────────────────────────────────────
    print("\n[1/4] Loading and filtering CRSP data …")
    df = load_and_filter(DATA_PATH)
    R  = build_return_matrix(df)
    Z, scaler = standardise(R)
    print(f"      Return matrix shape: {R.shape}  "
          f"({R.shape[0]} months × {R.shape[1]} stocks)")

    # ── Stage 2: PCA ──────────────────────────────────────────────────
    print("\n[2/4] Running PCA …")
    pca, loadings = run_pca(Z)
    ev = pca.explained_variance_ratio_ * 100
    print(f"      Variance explained — "
          f"PC1: {ev[0]:.2f}%  PC2: {ev[1]:.2f}%  "
          f"PC3: {ev[2]:.2f}%  PC4: {ev[3]:.2f}%  "
          f"Total: {sum(ev):.2f}%")

    scree_plot(pca)

    tbl = loadings_table(loadings, R.columns)
    print("\n--- Extreme Loadings per Component ---")
    print(tbl.to_string())

    # ── Stage 3: views ────────────────────────────────────────────────
    print("\n[3/4] Factor interpretation & views …")
    print_factor_views(pca)

    # ── Stage 4: portfolio ────────────────────────────────────────────
    print("\n[4/4] Building portfolio …")
    portfolio = build_portfolio(loadings, R.columns)
    plot_portfolio(portfolio)
    print_portfolio_summary(portfolio, pca)

    # Save tickers for submission
    tickers_out = portfolio[["Ticker", "Weight"]]
    tickers_out.to_csv("portfolio_tickers.csv", index=False)
    print("\nPortfolio tickers saved → portfolio_tickers.csv")
    print(SEP)


if __name__ == "__main__":
    main()
