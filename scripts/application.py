"""
Helpers for Section 5: LP-VAR applications.
Loads the .mat estimates, runs SMA at each application-specific m (recomputing, or loading cached results from disk), evaluates the normalized BNM worst-case risk, and computes descriptive metrics for the four applications.
"""

import os
import pickle
import numpy as np
import pandas as pd
import mat73
from .experiment import run_minimal_experiment
from .parameters import epsilon_target
from .sma import delta_pi_numba


def load_application_data(mat_path):
    data_dict = mat73.loadmat(mat_path)
    appl = data_dict['appl']
    results = appl['results']

    # Gertler and Karadi
    results_1 = results[0]
    estimates = results_1["estims"]
    ses = results_1["ses"]
    VAR_GK = estimates[0, 0, :]
    LP_GK = estimates[1, 0, :]
    VAR_se_GK = ses[0, 0, :]
    LP_se_GK = ses[1, 0, :]

    # Romer and Romer
    results_2 = results[1]
    estimates = results_2["estims"]
    ses = results_2["ses"]
    VAR_RR = estimates[0, 1, :]
    LP_RR = estimates[1, 1, :]
    VAR_se_RR = ses[0, 1, :]
    LP_se_RR = ses[1, 1, :]

    # Ramey
    results_3 = results[2]
    estimates = results_3["estims"]
    ses = results_3["ses"]
    VAR_R = estimates[0, 0, :]
    LP_R = estimates[1, 0, :]
    VAR_se_R = ses[0, 0, :]
    LP_se_R = ses[1, 0, :]

    # FORD
    results_4 = results[3]
    estimates = results_4["estims"]
    ses = results_4["ses"]
    VAR_FORD = estimates[0, 2, :]
    LP_FORD = estimates[1, 2, :]
    VAR_se_FORD = ses[0, 2, :]
    LP_se_FORD = ses[1, 2, :]

    return {
        "GK": dict(VAR=VAR_GK, LP=LP_GK, VAR_se=VAR_se_GK, LP_se=LP_se_GK),
        "RR": dict(VAR=VAR_RR, LP=LP_RR, VAR_se=VAR_se_RR, LP_se=LP_se_RR),
        "R": dict(VAR=VAR_R, LP=LP_R, VAR_se=VAR_se_R, LP_se=LP_se_R),
        "FORD": dict(VAR=VAR_FORD, LP=LP_FORD, VAR_se=VAR_se_FORD, LP_se=LP_se_FORD)}


def build_applications_dict(raw_data, misspec_share):
    APPLICATIONS = {
        "GK": {
            "caption": "Gertler & Karadi (2015)",
            "title": "Monetary policy shock to industrial production",
            "T": 269,
            "VAR": raw_data["GK"]["VAR"],
            "LP": raw_data["GK"]["LP"],
            "VAR_se": raw_data["GK"]["VAR_se"],
            "LP_se": raw_data["GK"]["LP_se"],
            "xtick_step": 5},
        "RR": {
            "caption": "Romer & Romer (2010)",
            "title": "Tax shock to GDP",
            "T": 230,
            "VAR": raw_data["RR"]["VAR"],
            "LP": raw_data["RR"]["LP"],
            "VAR_se": raw_data["RR"]["VAR_se"],
            "LP_se": raw_data["RR"]["LP_se"],
            "xtick_step": 2},
        "R": {
            "caption": "Ramey (2011)",
            "title": "Military news shock to GDP",
            "T": 266,
            "VAR": raw_data["R"]["VAR"],
            "LP": raw_data["R"]["LP"],
            "VAR_se": raw_data["R"]["VAR_se"],
            "LP_se": raw_data["R"]["LP_se"],
            "xtick_step": 2},
        "FORD": {
            "caption": "Francis, Owyang, Roush, and DiCecio (2014)",
            "title": "TFP shock to labor productivity",
            "T": 242,
            "VAR": raw_data["FORD"]["VAR"],
            "LP": raw_data["FORD"]["LP"],
            "VAR_se": raw_data["FORD"]["VAR_se"],
            "LP_se": raw_data["FORD"]["LP_se"],
            "xtick_step": 2},
    }
    for key, app in APPLICATIONS.items():
        app["m"] = float(np.sqrt(app["T"] * misspec_share))
        app["VAR_var"] = np.asarray(app["VAR_se"], dtype=float) ** 2
        app["LP_var"] = np.asarray(app["LP_se"], dtype=float) ** 2
    return APPLICATIONS


def run_or_load_application_sma(applications, cache_path, force_rerun=False, n_samples_risk=1000, seed_base=123, show_progress=True):
    expected_keys = set(applications.keys())
    sma_results = {}
    if (not force_rerun) and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        cached_keys = set(cached.keys())
        if cached_keys == expected_keys:
            print(f"Loaded cached SMA results from {cache_path}")
            return cached
        sma_results.update(cached)
        missing = sorted(expected_keys - cached_keys)
        print(f"Loaded partial cache from {cache_path}; missing applications: {missing}")
    for j, (key, app) in enumerate(applications.items()):
        if key in sma_results and not force_rerun:
            print(f"Skipping {key}: already in cache.")
            continue
        m = float(app["m"])
        eps = epsilon_target(m)
        print("\n" + "=" * 80)
        print(f"{key}: {app['caption']}")
        print(f"T = {app['T']}, m = {m:.4f}, epsilon(m) = {eps:.4f}")
        print("=" * 80)
        res = run_minimal_experiment(
            m=m,
            epsilon=eps,
            n_mc_grad=1,
            n_samples_risk=n_samples_risk,
            seed=seed_base + j,
            show_progress=show_progress)
        sma_results[key] = res
        with open(cache_path, "wb") as f:
            pickle.dump(sma_results, f)
    print(f"\nSaved SMA application results to {cache_path}")
    return sma_results


def estimate_bnm_worst_case_risk(theta_grid, avg_prior, m, n_theta=301, n_mc=1000, seed=123):
    theta_eval = np.linspace(-m, m, n_theta)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n_mc)
    Y = theta_eval[:, None] + noise[None, :]
    d_vals = delta_pi_numba(Y.ravel(), theta_grid, avg_prior).reshape(Y.shape)
    risk_vals = np.mean((d_vals - theta_eval[:, None]) ** 2, axis=1)
    delta_eval = (2.0 * m) / (n_theta - 1)
    slack = (3.0 * m**2 + 4.0 * m) * delta_eval / 2.0
    wc_grid = float(np.max(risk_vals))
    return {
        "theta_eval": theta_eval,
        "risk_vals": risk_vals,
        "wc_risk": wc_grid,
        "wc_risk_certified": wc_grid + slack,
        "lipschitz_slack": slack,
        "theta_at_wc": float(theta_eval[np.argmax(risk_vals)])}


def compute_application_metrics(app, sma_res, normalized_wc_risk, tol=1e-12):
    m = float(app["m"])
    VAR = np.asarray(app["VAR"], dtype=float)
    LP = np.asarray(app["LP"], dtype=float)
    VAR_var = np.asarray(app["VAR_var"], dtype=float)
    LP_var = np.asarray(app["LP_var"], dtype=float)

    raw_delta_var = LP_var - VAR_var
    if np.any(raw_delta_var < -tol):
        bad = np.where(raw_delta_var < -tol)[0]
        print(f"Warning for {app['caption']}: LP variance is below VAR variance "
              f"at horizons {bad.tolist()}; using |LP_var - VAR_var| for the "
              f"discrepancy SE at these horizons.")

    # Note: this part uses the magnitude of the variance gap so the discrepancy SE is well-defined 
    # even when finite-sample noise makes LP_var < VAR_var (only few cases in early horizons; see warnings). 
    # The Hausman statistic itself keeps its sign so the bias correction still points the right way.
    sigma_delta_sq = np.abs(raw_delta_var)
    sigma_delta = np.sqrt(sigma_delta_sq)
    valid = sigma_delta > tol
    Delta = np.full_like(VAR, np.nan, dtype=float)
    Delta[valid] = (VAR[valid] - LP[valid]) / sigma_delta[valid]
    Delta[0] = 0.0
    Delta_for_rule = np.where(valid, Delta, 0)
    d_norm = delta_pi_numba(Delta_for_rule, sma_res["theta_grid"], sma_res["avg_prior"])
    d_norm = np.asarray(d_norm, dtype=float)
    d_norm[~valid] = 0

    bias_correction = sigma_delta * d_norm
    MM = VAR - bias_correction
    omega_star = m**2 / (1 + m**2)
    linear_star = omega_star * LP + (1 - omega_star) * VAR
    risk_VAR = VAR_var + (m**2) * sigma_delta_sq
    risk_LP = LP_var
    risk_linear = ((1 - omega_star)**2) * (m**2) * sigma_delta_sq + VAR_var + (omega_star**2) * sigma_delta_sq
    risk_MM = VAR_var + sigma_delta_sq * normalized_wc_risk

    risk_improvement_pct = 100 * (risk_linear - risk_MM) / risk_linear
    nonlinear_weight = np.full_like(VAR, np.nan, dtype=float)
    nonzero_delta = valid & (np.abs(Delta) > tol)
    nonlinear_weight[nonzero_delta] = d_norm[nonzero_delta] / Delta[nonzero_delta]

    return {
        "h": np.arange(len(VAR)),
        "m": m,
        "VAR": VAR,
        "LP": LP,
        "VAR_se": np.asarray(app["VAR_se"], dtype=float),
        "LP_se": np.asarray(app["LP_se"], dtype=float),
        "VAR_var": VAR_var,
        "LP_var": LP_var,
        "sigma_delta_sq": sigma_delta_sq,
        "sigma_delta": sigma_delta,
        "Delta": Delta,
        "abs_Delta": np.abs(Delta),
        "Hausman_sq": Delta**2,
        "d_norm": d_norm,
        "bias_correction": bias_correction,
        "MM": MM,
        "omega_star": omega_star,
        "linear_star": linear_star,
        "nonlinear_weight": nonlinear_weight,
        "risk_VAR": risk_VAR,
        "risk_LP": risk_LP,
        "risk_linear": risk_linear,
        "risk_MM": risk_MM,
        "risk_improvement_pct": risk_improvement_pct}


GK_CFG = dict(file="Monetarydat.xlsx", sheet="Monthly", dcol="DATES", dmin=1990, dmax=2012 + 5/12, p=2, hmax=48,
    shock="FF4_TC", resp_list=["LIP", "LCPI", "GS1", "EBP"], resp="LIP", diff_vars=["LIP", "LCPI"],
    series=lambda d: {c: d[c].values.astype(float) for c in ["FF4_TC", "LIP", "LCPI", "GS1", "EBP"]})


def _gk_estimate(Y, lhs, p, H, ri):
    T, n = Y.shape
    X = np.column_stack([np.ones(T - p)] + [Y[p - l:T - l] for l in range(1, p + 1)])
    B = np.linalg.solve(X.T @ X, X.T @ Y[p:])
    res = Y[p:] - X @ B
    A = [B[1 + l * n:1 + (l + 1) * n].T for l in range(p)]
    r0 = res[:, 0]; v0 = np.var(r0)
    nu = np.array([np.mean((res[:, i] - res[:, i].mean()) * (r0 - r0.mean())) / v0 for i in range(n)])
    Th = [np.eye(n)]
    for h in range(1, H + 1):
        Mx = np.zeros((n, n))
        for l in range(1, min(h, p) + 1):
            Mx += A[l - 1] @ Th[h - l]
        Th.append(Mx)
    var = np.cumsum(np.array([Th[h] @ nu for h in range(H + 1)])[:, ri])
    shock = Y[:, 0]
    lp = np.full(H + 1, np.nan)
    for h in range(H + 1):
        idx = np.arange(p + 1, T - h)
        y = lhs[idx + h] - lhs[idx - 1]
        cols = [np.ones(len(idx)), shock[idx]] + [Y[idx - L, k] for L in range(1, p + 1) for k in range(n)]
        Xl = np.column_stack(cols); ok = ~np.isnan(y) & ~np.isnan(Xl).any(1)
        lp[h] = np.linalg.lstsq(Xl[ok], y[ok], rcond=None)[0][1]
    return var, lp, B, res


def gk_cumulative_diff_1sd(data_path, n_boot=400, seed=123):
    cfg = GK_CFG
    df = pd.read_excel(data_path, sheet_name=cfg["sheet"])
    d = df[(df[cfg["dcol"]] >= cfg["dmin"] - 1e-6) & (df[cfg["dcol"]] <= cfg["dmax"] + 1e-6)].reset_index(drop=True)
    S = cfg["series"](d); names = [cfg["shock"]] + cfg["resp_list"]
    M = np.column_stack([S[k] for k in names]); M = M[~np.isnan(M).any(1)]
    R = {k: M[:, i] - M[:, i].mean() for i, k in enumerate(names)}
    sys = [R[cfg["shock"]]]; lvl = {}
    for nm in cfg["resp_list"]:
        lvl[nm] = R[nm]
        sys.append(np.r_[np.nan, np.diff(R[nm])] if nm in cfg["diff_vars"] else R[nm])
    Y = np.column_stack(sys); keep = ~np.isnan(Y).any(1); Y = Y[keep]; T, n = Y.shape
    ri = cfg["resp_list"].index(cfg["resp"]) + 1; p, H = cfg["p"], cfg["hmax"]
    lr = lvl[cfg["resp"]][keep]

    var, lp, B, res = _gk_estimate(Y, lr, p, H, ri)
    rng = np.random.default_rng(seed)
    VV = np.empty((n_boot, H + 1)); LL = np.empty((n_boot, H + 1))
    for b in range(n_boot):
        e = res[rng.integers(0, len(res), len(res))]; Ys = np.zeros((T, n)); Ys[:p] = Y[:p]
        for tt in range(p, T):
            Ys[tt] = B.T @ np.concatenate([[1.0]] + [Ys[tt - l] for l in range(1, p + 1)]) + e[tt - p]
        VV[b], LL[b], _, _ = _gk_estimate(Ys, np.cumsum(Ys[:, ri]), p, H, ri)
    sd = float(np.std(Y[:, 0]))
    out = dict(VAR=var, LP=lp, VAR_se=np.nanstd(VV, 0), LP_se=np.nanstd(LL, 0))
    return {k: np.asarray(v, float) * sd for k, v in out.items()}
