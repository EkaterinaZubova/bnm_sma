"""Diagnostics of the risk function.

- certify_worst_case_risk: Lipschitz-certified upper bound on sup_theta R(d, theta) (Lemma 1), used by the main worst-case-risk figures.
- evaluate_risk_profile / empirical_lipschitz_from_grid / local_lipschitz_from_grid / lipschitz_constant: compare the theoretical Lipschitz constant $L(m)=3m^2+4m$ to its
empirical analogue on the SMA risk function (See Appendix E).
"""

import numpy as np
from .sma import delta_pi_numba


def evaluate_risk_profile(theta_eval, theta_grid, avg_prior, n_risk_mc=1000, seed=123):
    theta_eval = np.asarray(theta_eval)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n_risk_mc)
    Y = theta_eval[:, None] + noise[None, :]
    d_vals = delta_pi_numba(Y.ravel(), theta_grid, avg_prior).reshape(Y.shape)
    risk_vals = np.mean((d_vals - theta_eval[:, None])**2, axis=1)
    return risk_vals


def empirical_lipschitz_from_grid(theta_eval, risk_vals):
    theta_eval = np.asarray(theta_eval)
    risk_vals = np.asarray(risk_vals)
    dtheta = np.abs(theta_eval[:, None] - theta_eval[None, :])
    drisk = np.abs(risk_vals[:, None] - risk_vals[None, :])
    slope_mat = np.zeros_like(drisk)
    mask = dtheta > 0
    slope_mat[mask] = drisk[mask] / dtheta[mask]
    i_star, j_star = np.unravel_index(np.argmax(slope_mat), slope_mat.shape)
    return {
        "L_emp": float(slope_mat[i_star, j_star]),
        "i_star": int(i_star),
        "j_star": int(j_star),
        "theta_i": float(theta_eval[i_star]),
        "theta_j": float(theta_eval[j_star]),
        "risk_i": float(risk_vals[i_star]),
        "risk_j": float(risk_vals[j_star])}


def local_lipschitz_from_grid(theta_eval, risk_vals):
    theta_eval = np.asarray(theta_eval)
    risk_vals = np.asarray(risk_vals)
    slopes = np.abs(np.diff(risk_vals)) / np.diff(theta_eval)
    k = int(np.argmax(slopes))
    return {
        "L_local": float(slopes[k]),
        "k_star": k,
        "theta_left": float(theta_eval[k]),
        "theta_right": float(theta_eval[k+1]),
        "risk_left": float(risk_vals[k]),
        "risk_right": float(risk_vals[k+1])}


def lipschitz_constant(m):
    return 3 * m**2 + 4 * m


def certify_worst_case_risk(risk_vals, theta_eval, m, L=None):
    risk_vals = np.asarray(risk_vals, dtype=float)
    theta_eval = np.asarray(theta_eval, dtype=float)
    n = len(theta_eval)
    if n < 2:
        raise ValueError("theta_eval needs at least two points to define a spacing.")
    if L is None:
        L = lipschitz_constant(m)        # default: SMA estimator
    delta_eval = (theta_eval.max() - theta_eval.min()) / (n - 1)
    slack = L * delta_eval / 2
    adjacent_avg = (risk_vals[:-1] + risk_vals[1:]) / 2
    per_subinterval = adjacent_avg + slack
    certified_upper = float(min(np.max(per_subinterval), 4 * m**2))
    return {
        "grid_max": float(np.max(risk_vals)),
        "slack": slack,
        "certified_upper": certified_upper,
        "delta_eval": delta_eval,
    }