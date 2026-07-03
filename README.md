# Replication package

**"Approximate Minimax Estimation of a Bounded Normal Mean via Stochastic Mirror Ascent"**
José Luis Montiel Olea and Ekaterina Zubova.

---

## 1. Overview

This package reproduces all figures of the paper (Section 4 simulations, Appendix E, and the Section 5 LP--VAR application) and the values in Table 1. 

The code is written in Python: three Jupyter notebooks in `code/` call modules in `scripts/`. 

Heavy computations are cached in `data/` (`.pkl`), so by default the figure cells load the cache produced by this code and run in seconds. Setting the rerun flags recomputes from scratch. 

The Section 5 application additionally requires two external input datasets, `res_application.mat` and `Monetarydat.xlsx`, which are not redistributed here and must be downloaded (see Section 2).

```
code/      01_simulations.ipynb, 02_simulations_time_budget.ipynb, 03_application.ipynb
scripts/   sma.py, parameters.py, experiment.py, application.py, diagnostics.py, plotting.py
data/      cached results (.pkl); additional inputs res_application.mat & Monetarydat.xlsx must be downloaded (see Section 2)
requirements.txt, LICENSE.txt
```

## 2. Data availability

The Python code and the cached result files (`data/*.pkl`) are the authors' own and are included in this package. The simulation notebooks (`01`, `02`) require no external data. The Section 5 application (`03`) uses two external datasets that are not redistributed here and remain subject to their original providers' terms. Download them from the public sources below and place them in `data/`. 

| Additional file | Source | Citation |
|---|---|---|
| `res_application.mat` — VAR and LP impulse-response estimates and bootstrap standard errors | Regenerate by running the MIT-licensed code at <https://github.com/ckwolf92/lp_var_inference>; underlying macro data from Ramey (2016). | Montiel Olea, Plagborg-Møller, Qian & Wolf (forthcoming); Ramey (2016) |
| `Monetarydat.xlsx` — monthly dataset for the Gertler--Karadi re-estimation | sourced from Ramey's (2016) data archive. | Gertler & Karadi (2015); Ramey (2016) |

*These two files are not covered by this package's `LICENSE.txt`.

## 3. Computational Requirements

### 3.1 Software
Tested with Python 3.9.6. The exact versions are listed in `requirements.txt` (install with `pip install -r requirements.txt`).

### 3.2 Hardware 
The code was run on a personal laptop: MacBook Pro (M5). MacOS Tahoe 26.5.1.


## 4. Content description
- `scripts/sma.py` — stochastic mirror ascent (Algorithm 1): `stochastic_mirror_ascent` returns the
  averaged prior; `delta_pi_numba` is the (numerically stabilized) posterior-mean Bayes rule; `bayes_lower_bound` is the Bayes-risk lower bound.
- `scripts/parameters.py` — `epsilon_target(m)` and `J_I_from_epsilon(m, eps)` (the grid size `I` and iteration count `J` of Theorem 1).
- `scripts/experiment.py` — `run_minimal_experiment` and `run_budgeted_experiment` (Section 4.6).
- `scripts/diagnostics.py` — `certify_worst_case_risk` (Lipschitz-certified worst-case-risk bound, Lemma 1 / Appendix E) and the empirical-Lipschitz diagnostics.
- `scripts/application.py` — Section 5 LP--VAR helpers (loads `.mat`, re-estimates Gertler--Karadi in cumulative first differences, runs SMA per application, computes Hausman statistic, bias correction, and risk metrics).
- `scripts/plotting.py` — application figure helpers.

## 5. How to use
1. `pip install -r requirements.txt`.
2. `cd code` and launch Jupyter. The notebooks add the package root to `sys.path`, so run them **from `code/`**.
3. Run **`01_simulations.ipynb`** top to bottom (Caches load by default or set `rerun_full_algo=True` to recompute.)
4. Run **`02_simulations_time_budget.ipynb`** top to bottom: the 5-15-minute budgeted figures. (Caches load by default or set `rerun_budgeted_simulations=True` to recompute.)
5. Run **`03_application.ipynb`** top to bottom: the four LP--VAR applications and Table 1. (Set `rerun_sma=True` and `rerun_risks=True` to recompute.)



| Figure in the paper | Notebook | Cell (count from 0) |
|---|---|---|
| Table 1 ($m$, $\epsilon(m)$ per application) | 03_application | 4 |
| Fig. 1 — iterations J and grid I vs m (`J_and_I_of_m`) | 01_simulations | 5 |
| Fig. 2 — computation time and $\epsilon(m)$ (`time_SMD`) | 01_simulations | 7 |
| Fig. 3 — worst-case risk, certified (`certified_risk_200_000_points`) | 01_simulations | 11 |
| Fig. 4 — percentage improvement, certified (`certified_impr_200_000_points`) | 01_simulations | 12 |
| Fig. 5 — estimator comparison (`estimators_m_1_2_4`) | 01_simulations | 14 |
| Fig. 6 — required vs completed iterations (`J_required_vs_completed`) | 02_simulations_time_budget | 7 |
| Fig. 7 — risk, 5-min limit (`risk_5_min`) | 02_simulations_time_budget | 10 |
| Fig. 8 — improvement, 5 and 15-min (`risk_impr_5_15_min`) | 02_simulations_time_budget | 12 |
| Fig. 9 — LP vs VAR IRFs (`4_IRFs_diff`) | 03_application | 6 |
| Fig. 10 — Hausman statistic (`Delta_stat_diff`) | 03_application | 14 |
| Fig. 11 — bias correction (`bias_correction_1_pct_diff`) | 03_application | 15 |
| Fig. 12 — VAR/LP/linear/bias-corrected (`b_minimax_irfs_1_pct_diff`) | 03_application | 16 |
| Fig. 13 — risk improvement over linear (`risk_impr_irfs_1_pct_diff`) | 03_application | 17 |
| App. E Fig. 14 — Lipschitz constants (`lipschitz_constant`) | 01_simulations | 20 |
| App. E Fig. 15 — certified risk, all estimators (`certified_risk_all_estimators`) | 01_simulations | 10 |
| App. E Fig. 16 — Casella--Strawderman comparison (`risk_m_1`) | 01_simulations | 22 |
| App. E Fig. 17 — risk across seeds, m=2 (`risk_of_theta_2`) | 01_simulations | 17 |



## References
- Gertler, M. and P. Karadi (2015): "Monetary Policy Surprises, Credit Costs, and Economic Activity," *AEJ: Macroeconomics* 7(1), 44--76.
- Romer, C. D. and D. H. Romer (2010): "The Macroeconomic Effects of Tax Changes," *American Economic Review* 100(3), 763--801.
- Ramey, V. A. (2011): "Identifying Government Spending Shocks: It's All in the Timing," *QJE* 126(1), 1--50.
- Francis, N., M. T. Owyang, J. E. Roush, and R. DiCecio (2014): "A Flexible Finite-Horizon Alternative to Long-Run Restrictions…," *REStat* 96(4), 638--647.
- Ramey, V. A. (2016): "Macroeconomic Shocks and Their Propagation," *Handbook of Macroeconomics* 2, 71--162.
- Montiel Olea, J. L., M. Plagborg-Møller, E. Qian, and C. K. Wolf (forthcoming): "Double Robustness of Local Projections and Some Unpleasant VARithmetic," *Econometrica*.
- Montiel Olea, J. L., M. Plagborg-Møller, E. Qian, and C. K. Wolf (2025): "Local Projections or VARs? A Primer for Macroeconomists", *NBER Macroeconomics Annual* 40, 111--152.
