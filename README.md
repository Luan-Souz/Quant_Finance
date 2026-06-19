# Quantitative Finance Algorithms

A collection of quantitative finance algorithms implemented during the Quantitative Methods in Finance and Quantitative Portfolio Management graduate courses at Northeastern University, as part of the Master of Science in Finance program.

---

## Table of Contents

- [1. Descriptive Analysis of Market Index Returns](#1-descriptive-analysis-of-market-index-returns)
- [2. Factor Model Analysis of Industry Portfolios](#2-factor-model-analysis-of-industry-portfolios)
- [3. Volatility Estimation & GARCH Modeling](#3-volatility-estimation--garch-modeling)
- [4. Quantitative Portfolio Optimisation](#4-quantitative-portfolio-optimisation)
  * [4.1 Data, Statistics & Optimisation Inputs](#41-data-statistics--optimisation-inputs)
  * [4.2 Minimum-Variance Portfolio](#42-minimum-variance-portfolio)
  * [4.3 Maximum-Return Portfolio](#43-maximum-return-portfolio)
  * [4.4 Maximum-Sharpe Portfolio](#44-maximum-sharpe-portfolio)
  * [4.5 Risk-Parity Portfolio](#45-risk-parity-portfolio)
  * [4.6 Black-Litterman Portfolio](#46-black-litterman-portfolio)
- [5. Stochastic Processes & Option Pricing](#5-stochastic-processes--option-pricing)
  * [5.1 GBM Simulation & Discretization Comparison](#51-gbm-simulation--discretization-comparison)
  * [5.2 Binomial Option Pricing & Payoff Analysis](#52-binomial-option-pricing--payoff-analysis)
  * [5.3 Binomial Tree Model](#53-binomial-tree-model)
  * [5.4 Black-Scholes Model](#54-black-scholes-model)
  * [5.5 Monte Carlo — Risk-Neutral GBM](#55-monte-carlo--risk-neutral-gbm)
  * [5.6 Heston Stochastic Volatility Model](#56-heston-stochastic-volatility-model)

---

## 1. Descriptive Analysis of Market Index Returns

Profiled the full statistical behavior of two equity indices across 500 trading days — distributions, tail risk, and moving average momentum strategies. One index revealed excess kurtosis of 18.62, rendering any normality-based risk model unreliable. A 20-day MA crossover strategy captured only 58% of buy-and-hold returns while executing 40+ trades.

### Approach

- Computed full descriptive statistics (mean, variance, skewness, excess kurtosis, min, max) for each return series
- Tested for normality departures using histogram and KDE overlays against a fitted normal reference
- Constructed a 20-day moving average crossover momentum strategy and compared cumulative performance against a passive buy-and-hold benchmark


### Key Results

| Metric                      | Index 1 | Index 2 |
| --------------------------- | ------- | ------- |
| Excess Kurtosis             | 18.62   | 4.31    |
| Skewness                    | +0.79   | −0.52   |
| MA Strategy vs Buy-and-Hold | 58%     | 71%     |

---

## 2. Factor Model Analysis of Industry Portfolios

Decomposed Manufacturing and Telecom portfolio returns using four progressively richer factor models — CAPM through Fama-French 5-factor. Adding size and value factors alone lifted R² from 67% to 89% for Manufacturing. A recurring finding: both industries load negatively on momentum, behaving as contrarian portfolios where past losers tend to recover.

### Approach

- Estimated CAPM, Fama-French 3-factor, 4-factor (with momentum), and 5-factor models via OLS regression
- Factor data sourced from the Kenneth R. French Data Library
- Compared model fit (R², adjusted R², AIC) and statistical significance of factor loadings across models


### Factor Model Comparison — Manufacturing Portfolio

| Model       | R²   | Momentum Loading | SMB Loading  |
| ----------- | ---- | ---------------- | ------------ |
| CAPM        | 0.67 | —                | —            |
| FF 3-Factor | 0.89 | —                | +significant |
| FF 4-Factor | 0.90 | −significant     | +significant |
| FF 5-Factor | 0.91 | −significant     | +significant |

---

## 3. Volatility Estimation & GARCH Modeling

Estimated and modeled time-varying volatility across two stocks and two market indices. Compared Close-to-Close and Garman-Klass estimators, then fit GARCH and GJR-GARCH models with full diagnostic testing. One index showed a significant leverage effect — bad news amplifies future volatility far more than equivalent good news, confirmed by a 35-point AIC improvement over the symmetric model.

### Approach

- Computed 21-day rolling annualised volatility using both Close-to-Close and Garman-Klass estimators
- Fit GARCH(1,1) and GJR-GARCH(1,1) models to each series
- Performed diagnostic testing: Ljung-Box on standardised residuals, ARCH-LM test, and AIC/BIC comparison


### Estimator Comparison — Stock 1

| Estimator      | Mean Ann. Vol | Std Dev | Q95   |
| -------------- | ------------- | ------- | ----- |
| Close-to-Close | 23.0%         | 7.5%    | 41.0% |
| Garman-Klass   | 20.5%         | 4.6%    | 31.8% |


The Garman-Klass estimator produced consistently lower and more stable volatility estimates, reflecting its use of intraday high-low range information.

---

## 4. Quantitative Portfolio Optimisation

Applied five portfolio optimisation frameworks — Minimum-Variance, Maximum-Return, Maximum-Sharpe, Risk-Parity, and Black-Litterman — to a six-asset cross-market universe, implemented entirely in Python using SLSQP. The Black-Litterman model, combining an EWMA equilibrium prior with explicit investor views, outperformed all naive mean-variance methods on a risk-adjusted basis.

### Asset Universe

| Ticker  | Type           | Ann. Return | Ann. Volatility |
| ------- | -------------- | ----------- | --------------- |
| ETH-USD | Cryptocurrency | 13.45%      | 57.75%          |
| BTC-USD | Cryptocurrency | 27.03%      | 43.51%          |
| SPY     | US Equity ETF  | 14.80%      | 15.71%          |
| IEO     | Energy ETF     | 11.17%      | 26.74%          |
| ABEV    | Brazilian ADR  | 12.08%      | 27.10%          |
| PBR     | Brazilian ADR  | 22.26%      | 32.68%          |


- **Data source:** Yahoo Finance — daily adjusted closing prices
- **Sample:** January 2023 – March 2026 · 550+ observations
- **Return type:** Log returns (time-additive, approximate normality)
- **Risk-free rate:** 3-month US T-bill · 3.59% annualised · Constant
- **Constraints:** Long-only (`wᵢ ∈ [0,1]`) · Fully invested (`Σwᵢ = 1`)

---

### 4.1 Data, Statistics & Optimisation Inputs

Assembled and cleaned the six-asset cross-market universe, aligned over 550 trading days using log returns for their time-additivity and normality properties. Established the three optimisation inputs (μ̂, Σ̂, rf) and mapped the estimation risk hierarchy: Max-Return is fully exposed to noisy return estimates while Min-Variance and Risk-Parity bypass them entirely.

#### Sample Covariance Matrix (daily log returns)

|          | ETH      | BTC      | SPY      | IEO      | ABEV     | PBR      |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| **ETH**  | 0.001323 | 0.000820 | 0.000145 | 0.000129 | 0.000077 | 0.000122 |
| **BTC**  | 0.000820 | 0.000751 | 0.000097 | 0.000082 | 0.000067 | 0.000071 |
| **SPY**  | 0.000145 | 0.000097 | 0.000098 | 0.000085 | 0.000045 | 0.000060 |
| **IEO**  | 0.000129 | 0.000082 | 0.000085 | 0.000284 | 0.000052 | 0.000180 |
| **ABEV** | 0.000077 | 0.000067 | 0.000045 | 0.000052 | 0.000291 | 0.000125 |
| **PBR**  | 0.000122 | 0.000071 | 0.000060 | 0.000180 | 0.000125 | 0.000424 |


Key correlation findings: ETH–BTC correlation of 0.82 forms a tight crypto cluster. ABEV and PBR show near-zero correlation with both cryptos (0.12–0.16), making them natural diversifiers.

---

### 4.2 Minimum-Variance Portfolio

Solved the global minimum-variance problem using only the covariance matrix — expected returns are ignored entirely, making this the most robust method to return estimation error. Both crypto assets receive zero weight given their extreme volatility. SPY dominates at 77.95% as the lowest-volatility anchor.

#### Objective Function

$$w^* = \arg\min_{w} \sqrt{w^\top \Sigma w} \quad \text{s.t.} \quad \sum_i w_i = 1, \quad w_i \in [0,1]$$

#### Results

| Metric          | Value                   |
| --------------- | ----------------------- |
| Ann. Return     | 14.60%                  |
| Ann. Volatility | **14.86%**              |
| Sharpe Ratio    | 0.7405                  |
| Period Return   | 43.45%                  |
| Active Assets   | 4 (SPY, ABEV, PBR, IEO) |

---

### 4.3 Maximum-Return Portfolio

Maximised expected return with no risk penalty. Under long-only constraints this degenerates to 100% allocation to the single highest-return asset. Despite delivering the highest raw return, it produces the lowest Sharpe ratio of all five methods — confirming that return maximisation without risk control is counterproductive on a risk-adjusted basis.

#### Objective Function

$$w^* = \arg\max_{w} \mu^\top w \quad \text{s.t.} \quad \sum_i w_i = 1, \quad w_i \in [0,1]$$

#### Results

| Metric          | Value            |
| --------------- | ---------------- |
| Ann. Return     | 27.03%           |
| Ann. Volatility | 43.51%           |
| Sharpe Ratio    | 0.5386           |
| Period Return   | 94.64%           |
| Active Assets   | 1 (BTC-USD 100%) |

---

### 4.4 Maximum-Sharpe Portfolio

Finds the tangency portfolio — the point on the efficient frontier where the capital market line is tangent, maximising excess return per unit of risk. Uses both μ̂ and Σ̂ simultaneously. Concentrates in three assets (SPY, PBR, BTC) and achieves the best Sharpe ratio among the naive mean-variance methods.

#### Objective Function

$$w^* = \arg\max_{w} \frac{\mu_p - r_f}{\sigma_p} \quad \text{where} \quad \mu_p = \mu^\top w, \quad \sigma_p = \sqrt{w^\top \Sigma w \cdot 252} \times 100$$

#### Results

| Metric          | Value             |
| --------------- | ----------------- |
| Ann. Return     | 18.33%            |
| Ann. Volatility | 17.16%            |
| Sharpe Ratio    | **0.8593**        |
| Period Return   | 58.22%            |
| Active Assets   | 3 (SPY, PBR, BTC) |

---

### 4.5 Risk-Parity Portfolio

Forces each asset to contribute exactly 1/N ≈ 16.67% of total portfolio variance. Expected returns are not used — only Σ̂ is required — making this the most robust method to return estimation error alongside Min-Variance. The only method to allocate across all six assets simultaneously.

#### Objective Function

$$w^* = \arg\min_{w} \frac{1}{N} \sum_{i=1}^{N} \left( RC_i - \bar{RC} \right)^2 \quad \text{where} \quad RC_i = \frac{w_i \cdot (\Sigma w)_i}{w^\top \Sigma w}$$

#### Results

| Metric          | Value       |
| --------------- | ----------- |
| Ann. Return     | 15.94%      |
| Ann. Volatility | 18.73%      |
| Sharpe Ratio    | 0.6593      |
| Period Return   | 49.40%      |
| Active Assets   | **6 (all)** |


#### Risk Contributions

| Asset   | Weight | Risk Contribution |
| ------- | ------ | ----------------- |
| ETH-USD | 7.95%  | 16.67%            |
| BTC-USD | 10.92% | 16.73%            |
| SPY     | 28.02% | 16.67%            |
| IEO     | 17.73% | 16.66%            |
| ABEV    | 20.48% | 16.69%            |
| PBR     | 14.90% | 16.58%            |

---

### 4.6 Black-Litterman Portfolio

Applied a Bayesian framework combining an EWMA equilibrium prior with two absolute investor views, blending them through quantified uncertainty parameters to produce posterior expected returns. The resulting portfolio achieved the highest Sharpe ratio of all five methods, added ABEV to the allocation where the naive Max-Sharpe had ignored it, and demonstrated meaningfully more stable weight construction than any naive mean-variance approach.

#### Framework

The posterior expected returns are computed as:

$$\mu_{BL} = \left[ (\tau\Sigma)^{-1} + P^\top \Omega^{-1} P \right]^{-1} \left[ (\tau\Sigma)^{-1} \pi + P^\top \Omega^{-1} Q \right]$$

with posterior covariance $\Sigma_{BL} = \Sigma + H$ where $H = \left[ (\tau\Sigma)^{-1} + P^\top \Omega^{-1} P \right]^{-1}$.

#### Prior & Hyperparameters

| Parameter | Value                               | Description                             |
| --------- | ----------------------------------- | --------------------------------------- |
| π         | `returns.ewm(span=10).mean() × 252` | EWMA log returns, annualised            |
| τ         | 0.05                                | Prior uncertainty scalar                |
| Ω         | τ · PΣP'                            | View uncertainty, proportional to prior |


EWMA was chosen over the sample mean (more responsive to recent regime) and over CAPM equilibrium (unavailable for this cross-asset universe).

#### Investor Views

| View   | Type     | Statement                          | Q   |
| ------ | -------- | ---------------------------------- | --- |
| View 1 | Absolute | BTC-USD will return 30% annualised | 30% |
| View 2 | Absolute | SPY will return 15% annualised     | 15% |


#### Prior vs Posterior Expected Returns

| Asset   | Prior  | Posterior | Δ (pp) | Mechanism                |
| ------- | ------ | --------- | ------ | ------------------------ |
| ETH-USD | ~9.0%  | 16.31%    | +7.3   | Indirect — ρ(BTC) = 0.82 |
| BTC-USD | ~25.0% | 28.99%    | +4.0   | Direct — View 1          |
| SPY     | ~12.0% | 16.57%    | +4.6   | Direct — View 2          |
| IEO     | ~9.0%  | 11.06%    | +2.1   | Indirect via Σ           |
| ABEV    | ~14.0% | 15.36%    | +1.4   | Indirect via Σ           |
| PBR     | ~22.0% | 23.84%    | +1.8   | Indirect via Σ           |


#### Results vs Naive Max-Sharpe

| Metric          | Black-Litterman | Max-Sharpe | Δ        |
| --------------- | --------------- | ---------- | -------- |
| Ann. Return     | **19.42%**      | 18.33%     | +1.09 pp |
| Ann. Volatility | **16.77%**      | 17.16%     | −0.39 pp |
| Sharpe Ratio    | **0.9443**      | 0.8593     | +0.085   |
| Active Assets   | 4               | 3          | +1       |
| ABEV allocation | **6.45%**       | ~0.09%     | +6.4 pp  |

---

## Portfolio Optimisation — Full Comparison

| Portfolio           | Ann. Return | Ann. Vol   | Sharpe     | Period Return | Active Assets |
| ------------------- | ----------- | ---------- | ---------- | ------------- | ------------- |
| Min-Variance        | 14.60%      | 14.86%     | 0.7405     | 43.45%        | 4             |
| Max-Return          | 27.03%      | 43.51%     | 0.5386     | 94.64%        | 1             |
| Max-Sharpe          | 18.33%      | 17.16%     | 0.8593     | 58.22%        | 3             |
| Risk-Parity         | 15.94%      | 18.73%     | 0.6593     | 49.40%        | 6             |
| **Black-Litterman** | **19.42%**  | **16.77%** | **0.9443** | 55.24%        | 4             |

---

## 5. Stochastic Processes & Option Pricing

Implemented and compared GBM simulation schemes, binomial option pricing, and four European call pricing models — Binomial, Black-Scholes, Monte Carlo, and Heston — on a common parameter set. The Heston stochastic volatility model, with a leverage correlation of ρ = −0.70, produced a materially lower call price than Black-Scholes (6.02 vs 11.54), capturing the asymmetric risk-neutral distribution that constant-volatility models cannot reproduce.

### Asset & Parameters

| Parameter | Value  | Description                                   |
| --------- | ------ | --------------------------------------------- |
| Asset     | GOOGL  | GBM simulation (2022 – present)               |
| S₀        | 100    | Current stock price (stylised)                |
| K         | 105    | Strike price — slightly OTM                   |
| T         | 1      | Time to maturity (years)                      |
| r         | 0.04   | Continuously compounded risk-free rate        |
| σ         | 0.30   | Constant volatility (Binomial, BS, MC)        |
| rf (GBM)  | 3.59%  | Annualised T-bill rate (GOOGL simulation)     |

---

### 5.1 GBM Simulation & Discretization Comparison

Simulated 1,000 one-year price paths for GOOGL under two discretization schemes and two time-step frequencies. The exact (log-Euler) method is analytically correct for GBM — it produces positive prices at any step size and carries no discretization error. Moving from weekly to monthly steps leaves the mean terminal price virtually unchanged but widens the distribution, consistent with σ√T scaling.

#### Approach

- Estimated annualised μ and σ from GOOGL daily log returns (2022–present)
- Simulated four combinations: Euler vs Exact × Weekly (n=52) vs Monthly (n=12)
- Produced fan charts, terminal price histograms, and summary statistics per case
- Discussed at least three GBM limitations: constant volatility, no fat tails/jumps, and constant drift

#### Discretization Schemes

| Scheme        | Update Rule                                                              | Error         |
| ------------- | ------------------------------------------------------------------------ | ------------- |
| Euler         | S(t+dt) = S(t) · [1 + μ·dt + σ√dt·Z]                                    | First-order   |
| Exact (log-Euler) | S(t+dt) = S(t) · exp[(μ − ½σ²)·dt + σ√dt·Z]                        | None (exact)  |

#### Terminal Price Summary — All Four Cases

| Case            | Mean ($) | Median ($) | Std Dev ($) | P5 ($) | P95 ($) |
| --------------- | -------- | ---------- | ----------- | ------ | ------- |
| Exact — Weekly  | 352.52   | 332.88     | 116.44      | 191.03 | 573.74  |
| Euler — Weekly  | 348.33   | 335.56     | 114.02      | 188.46 | 552.71  |
| Exact — Monthly | 353.95   | 330.51     | 120.15      | 191.32 | 577.23  |
| Euler — Monthly | 353.82   | 338.19     | 112.67      | 187.70 | 548.52  |

Recommendation: exact discretization with weekly steps. The monthly Euler combination is the worst of both worlds — coarse grid and a biased approximation.

---

### 5.2 Binomial Option Pricing & Payoff Analysis

Priced European call and put options using a multi-step CRR binomial tree. With n = 4 steps the tree is coarse enough to illustrate put-call parity approximation error; the sensitivity analysis tracks convergence to the Black-Scholes value as n → 100, where the gap falls below one cent.

#### Approach

- Built CRR binomial tree with u = exp(σ√Δt), d = 1/u, p* = (exp(r·Δt) − d)/(u − d)
- Plotted payoff and profit diagrams for all four option positions (call/put × buyer/writer)
- Verified put-call parity: C − P = S₀ − K·e^(−rT)
- Sensitivity analysis across strike price K and number of steps n

#### Parameters & Base-Case Results

| Parameter | Value   | Metric            | Call    | Put     |
| --------- | ------  | ----------------- | ------- | ------- |
| n         | 4       | Price (C₀/P₀)     | 10.1532 | 10.5999 |
| u         | 1.1618  | p* (risk-neutral) | 0.4967  | —       |
| d         | 0.8607  | C − P             | −0.4468 | —       |
| Δt        | 0.2500  | S₀ − Ke^(−rT)     | −0.8829 | —       |

Put-call parity difference of 0.4361 at n=4 shrinks toward zero as the tree refines — the binomial model satisfies parity exactly in the continuous limit.

---

## Option Pricing Models — Four-Method Comparison

Priced the same European call option with four methods on a common parameter set (S₀=100, K=105, T=1, r=0.04, σ=0.30). The Binomial and Black-Scholes prices agree to within one cent at n=100, confirming convergence. Monte Carlo estimates tighten around the BS value as path count grows. The Heston model prices materially below BS due to the leverage effect (ρ = −0.70) — negative stock-vol correlation shifts the risk-neutral distribution leftward, reducing the expected call payoff.

| Method                       | Call Price  | vs Black-Scholes |
| ---------------------------- | ----------- | ---------------- |
| Binomial (n=100)             | 11.5448     | −0.0050          |
| **Black-Scholes**            | **11.5398** | —                |
| Monte Carlo — 1,000 paths    | 11.1577     | −0.3821          |
| Monte Carlo — 10,000 paths   | 11.4485     | −0.0913          |
| Monte Carlo — 100,000 paths  | 11.5713     | +0.0315          |
| Heston MC (ρ = −0.70)        | 6.0222      | −5.5176          |

---

### 5.3 Binomial Tree Model

Prices the European call by building a recombining CRR price tree and discounting expected payoffs backward under the risk-neutral measure. As n → ∞ the discrete tree converges to the continuous Black-Scholes solution — at n=100 the gap is less than one cent. The only model among the four that requires no closed-form formula and explicitly constructs the full distribution of terminal prices.

#### Pricing Formula

$$C_0 = e^{-rT} \sum_{j=0}^{n} \binom{n}{j} q^{j}(1-q)^{n-j} \max\!\left(S_0\, u^j\, d^{n-j} - K,\; 0\right), \quad q = p^*$$

$$u = e^{\sigma\sqrt{\Delta t}}, \quad d = \frac{1}{u}, \quad p^* = \frac{e^{r\Delta t} - d}{u - d}, \quad \Delta t = \frac{T}{n}$$

#### Results

| Metric            | Value                                   |
| ----------------- | --------------------------------------- |
| Call Price (n=100) | **11.5448**                            |
| vs Black-Scholes  | −0.0050                                 |
| u / d             | 1.1052 / 0.9048                         |
| p* (risk-neutral) | 0.5080                                  |
| Convergence       | Gap < $0.01 from n=100 onward           |

---

### 5.4 Black-Scholes Model

Provides the analytical closed-form solution for the European call price under constant volatility and log-normal stock dynamics. Serves as the benchmark for all other methods — the Binomial tree converges to it, Monte Carlo estimates cluster around it, and Heston departs from it precisely because it relaxes the constant-σ assumption. Greeks are also derived analytically, making BS the most interpretable model for sensitivity analysis.

#### Pricing Formula

$$C = S_0 \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)$$

$$d_1 = \frac{\ln(S_0/K) + (r + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

where $N(\cdot)$ is the standard normal CDF.

#### Results

| Metric      | Value      |   | Greek  | Symbol | Value   |
| ----------- | ---------- | - | ------ | ------ | ------- |
| Call Price  | **11.5398**|   | Delta  | Δ      | 0.4745  |
| d₁          | −0.0641    |   | Gamma  | Γ      | 0.0262  |
| d₂          | −0.3641    |   | Vega   | V      | 37.76   |
| N(d₁)       | 0.4745     |   | Theta  | Θ      | −9.28   |
| N(d₂)       | 0.3579     |   | Rho    | ρ      | 33.89   |

A 1 pp increase in σ raises the call price by ~$0.38 (Vega ÷ 100) — the BS sensitivity chart confirms the relationship is strictly increasing and convex in σ.

---

### 5.5 Monte Carlo — Risk-Neutral GBM

Estimates the call price by simulating a large number of risk-neutral terminal stock prices and discounting the average payoff. Uses exact (log-Euler) discretization at horizon T — no discretization error, always S_T > 0. The standard error scales as 1/√M, so quadrupling the path count halves the error; the 100,000-path estimate falls within $0.04 of the Black-Scholes value.

#### Pricing Formula

$$S_T = S_0 \cdot \exp\!\left[\left(r - \tfrac{1}{2}\sigma^2\right)T + \sigma\sqrt{T}\, Z\right], \quad Z \sim \mathcal{N}(0,1)$$

$$\hat{C}_{MC} = e^{-rT} \cdot \frac{1}{M} \sum_{m=1}^{M} \max\!\left(S_T^{(m)} - K,\; 0\right)$$

#### Results

| Paths (M)   | Call Price | Std Error | 95% CI              | vs Black-Scholes |
| ----------- | ---------- | --------- | ------------------- | ---------------- |
| 1,000       | 11.1577    | 0.3421    | [10.49 , 11.83]     | −0.3821          |
| 10,000      | 11.4485    | 0.1082    | [11.24 , 11.66]     | −0.0913          |
| **100,000** | **11.5713**| 0.0342    | [11.50 , 11.64]     | +0.0315          |

The MC estimator is unbiased — the Black-Scholes value (11.5398) lies inside the 95% confidence interval at all three path counts.

---

### 5.6 Heston Stochastic Volatility Model

Extends GBM by making variance itself a mean-reverting stochastic process correlated with the stock. This captures three empirically observed phenomena that Black-Scholes cannot: volatility clustering, mean-reversion toward a long-run level, and the leverage effect (ρ < 0 — stock drops coincide with volatility spikes). With ρ = −0.70, the Heston call price (6.02) is materially below Black-Scholes (11.54), reflecting the heavier left tail of the risk-neutral distribution.

#### Model Dynamics

$$dS = r\,S\,dt + \sqrt{v}\,S\,dW_1$$

$$dv = \kappa(\theta - v)\,dt + \xi\sqrt{v}\,dW_2, \qquad dW_1\,dW_2 = \rho\,dt$$

Discretized via full-truncation Euler (50,000 paths, 252 daily steps):

$$v_{t+\Delta t} = \max\!\left(v + \kappa(\theta - v)\Delta t + \xi\sqrt{\max(v,0)\,\Delta t}\;Z_2,\; 0\right)$$

$$S_{t+\Delta t} = S \cdot \exp\!\left[\left(r - \tfrac{1}{2}v\right)\Delta t + \sqrt{\max(v,0)\,\Delta t}\;Z_1\right]$$

#### Parameters

| Parameter | Value  | Description                                      |
| --------- | ------ | ------------------------------------------------ |
| v₀        | 0.09   | Initial variance — σ₀ = 30%, matches BS baseline |
| κ         | 2.0    | Mean-reversion speed — half-life ≈ 4.2 months    |
| θ         | 0.09   | Long-run variance — σ∞ = 30%                     |
| ξ         | 0.30   | Volatility of variance (vol-of-vol)              |
| ρ         | −0.70  | Stock-vol correlation — leverage effect          |

Feller condition 2κθ > ξ²: 0.36 > 0.09 ✓

#### Results

| Metric           | Value      |
| ---------------- | ---------- |
| Call Price       | **6.0222** |
| vs Black-Scholes | −5.5176    |
| Paths × Steps    | 50,000 × 252 |

#### Sensitivity to ρ

| ρ     | Heston Call | vs Black-Scholes |
| ----- | ----------- | ---------------- |
| −0.90 | ~4.80       | −6.74            |
| −0.70 | 6.02        | −5.52            |
| −0.50 | ~7.80       | −3.74            |
| −0.30 | ~9.50       | −2.04            |
| 0.00  | ~11.30      | −0.24            |

As ρ → 0 the leverage effect disappears and the Heston price converges toward the Black-Scholes value. A more negative ρ amplifies the left tail of the risk-neutral distribution — bad outcomes for the stock coincide with volatility spikes, reducing the expected call payoff.

---

## Repository Structure

```
Quant_Finance/
├── 01_Descriptive_Analysis/
│   └── descriptive_analysis.py          # HW1 Parts A1, A2, A3
├── 02_Factor_Models/
│   └── factor_models.py                 # HW1 Part B — CAPM through FF5
├── 03_Volatility_GARCH/
│   └── volatility_garch.py              # HW1 Part C — GK estimator + GARCH
├── 04_Portfolio_Optimisation/
│   ├── part_a_data_inputs.py            # HW2 Part A — data, stats, μ̂ Σ̂ rf
│   ├── part_b_optimisers.py             # HW2 Part B — all five methods + comparison
│   ├── part_c_analysis.py               # HW2 Part C — post-optimisation analysis
│   ├── optimiser_min_variance.py        # standalone Min-Variance
│   ├── optimiser_max_return.py          # standalone Max-Return
│   ├── optimiser_max_sharpe.py          # standalone Max-Sharpe
│   ├── optimiser_risk_parity.py         # standalone Risk-Parity
│   └── optimiser_black_litterman.py     # standalone Black-Litterman
└── 05_Option_Pricing/
    ├── part_a_gbm_simulation.py             # HW3 Part 1 — GBM simulation & discretization
    ├── part_b_binomial_pricing.py           # HW3 Part 2 — binomial trees & payoff diagrams
    ├── part_c_model_comparison.py           # HW3 Part 3 — four-model pricing comparison
    ├── pricing_binomial.py              # standalone Binomial (CRR) pricer
    ├── pricing_black_scholes.py         # standalone Black-Scholes pricer + Greeks
    ├── pricing_monte_carlo.py           # standalone Monte Carlo GBM pricer
    └── pricing_heston.py               # standalone Heston SV model pricer
```

**Run order for option pricing:**

```
python part_a_gbm_simulation.py     # GBM paths — no dependencies
python part_b_binomial_pricing.py   # binomial trees — no dependencies
python part_c_model_comparison.py   # four-model comparison — no dependencies
```

Each standalone pricer (`pricing_*.py`) is fully self-contained and requires no input files.

---

## Tech Stack

- **Python** — Primary implementation language for all algorithms
- **NumPy / Pandas** — Data manipulation and matrix operations
- **yfinance** — Market data download
- **SciPy** (`scipy.optimize.minimize`, SLSQP; `scipy.stats`) — Constrained optimisation and statistical testing
- **Matplotlib** — Visualisation
- **Kenneth R. French Data Library** — Factor data for regression models

---

## References

- Black, F., & Litterman, R. (1992). *Global portfolio optimization.* Financial Analysts Journal, 48(5), 28–43.
- Black, F., & Scholes, M. (1973). *The pricing of options and corporate liabilities.* Journal of Political Economy, 81(3), 637–654.
- Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). *Option pricing: A simplified approach.* Journal of Financial Economics, 7(3), 229–263.
- Fama, E. F., & French, K. R. (1993). *Common risk factors in the returns on stocks and bonds.* Journal of Financial Economics, 33(1), 3–56.
- Garman, M. B., & Klass, M. J. (1980). *On the estimation of security price volatilities from historical data.* Journal of Business, 53(1), 67–78.
- Heston, S. L. (1993). *A closed-form solution for options with stochastic volatility with applications to bond and currency options.* Review of Financial Studies, 6(2), 327–343.
- Maillard, S., Roncalli, T., & Teïletche, J. (2010). *The properties of equally weighted risk contribution portfolios.* Journal of Portfolio Management, 36(4), 60–74.

---

## Author

**Luan Ferreira de Souza**
