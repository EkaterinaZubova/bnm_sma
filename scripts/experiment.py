"""
Main algorithm simulations.

Two versions:
    run_minimal_experiment: full simulation for the given m.
    run_budgeted_experiment: simulations with time limit (see Section 4.6).
"""

import time
import numpy as np
from .parameters import J_I_from_epsilon
from .sma import bayes_lower_bound, delta_pi_numba, stochastic_mirror_ascent


def run_minimal_experiment(
    m: float = 1.4,
    epsilon: float = 0.05,
    n_mc_grad: int = 1,
    n_samples_risk: int = 1000,
    seed: int = 123,
    show_progress: bool = True) -> dict:
    J, I = J_I_from_epsilon(m, epsilon)
    theta_grid = np.linspace(-m, m, I)
    G = 4 * m**2
    eta = (epsilon/2) / (G**2)
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    avg_prior = stochastic_mirror_ascent(theta_grid=theta_grid, eta=eta, J=J, n_mc_grad=n_mc_grad, rng=rng, show_progress=show_progress)
    t1 = time.perf_counter()
    smd_runtime = t1 - t0  # 'smd_runtime' key = SMA (mirror-ascent) stage timing
    t2 = time.perf_counter()
    rng_eval = np.random.default_rng(seed + 999) 
    bayes_risk = bayes_lower_bound(theta_grid, avg_prior, n_samples=n_samples_risk, rng=rng_eval)
    t3 = time.perf_counter()
    risk_runtime = t3 - t2

    return {
        'm': m,
        'epsilon': epsilon,
        'J': J,
        'I': I,
        'theta_grid': theta_grid,
        'avg_prior': avg_prior,
        'bayes_risk': bayes_risk,
        'upper_bound': bayes_risk + epsilon,
        'n_mc_grad': n_mc_grad,
        'seed': seed,
        'smd_runtime': smd_runtime,
        'risk_runtime': risk_runtime,
        'total_runtime': smd_runtime + risk_runtime}


def run_budgeted_experiment(
    m: float,
    epsilon: float,
    time_budget: float = 300,
    n_mc_grad: int = 1,
    n_samples_risk: int = 5000,
    seed: int = 123) -> dict:
    J_target, I = J_I_from_epsilon(m, epsilon)
    theta_grid = np.linspace(-m, m, I)
    G = 4 * m**2
    eta = (epsilon / 2) / (G**2)
    rng = np.random.default_rng(seed)

    pi = np.full(I, 1 / I)
    pi_sum = np.zeros(I)
    t0 = time.perf_counter()
    j_completed = 0
    for j in range(1, J_target + 1):
        pi_sum += pi
        x_mc = rng.standard_normal((I, n_mc_grad)) + theta_grid[:, None]
        d_vals = delta_pi_numba(x_mc.ravel(), theta_grid, pi).reshape(I, n_mc_grad)
        grad = np.sum((d_vals - theta_grid[:, None]) ** 2, axis=1) / n_mc_grad
        pi *= np.exp(eta * grad)
        pi /= pi.sum()
        j_completed = j
        if j % 1000 == 0 and time.perf_counter() - t0 > time_budget:
            break

    smd_runtime = time.perf_counter() - t0
    avg_prior = pi_sum / j_completed
    completed = j_completed >= J_target
    rng_eval = np.random.default_rng(seed + 999)
    bayes_risk = bayes_lower_bound(theta_grid, avg_prior, n_samples=n_samples_risk, rng=rng_eval)
    # Evaluate the worst-case risk on a fine grid with the Lipschitz certificate
    n_theta_eval = max(301, 4 * I)
    theta_eval = np.linspace(-m, m, n_theta_eval)
    noise = rng_eval.standard_normal(n_samples_risk)
    Y = noise[None, :] + theta_eval[:, None]
    d_vals = delta_pi_numba(Y.ravel(), theta_grid, avg_prior).reshape(Y.shape)
    risk_per_theta = np.mean((d_vals - theta_eval[:, None]) ** 2, axis=1)
    worst_case_risk = float(np.max(risk_per_theta))
    delta_eval = (2 * m) / (n_theta_eval - 1)
    lipschitz_slack = (3 * m**2 + 4 * m) * delta_eval / 2
    worst_case_risk_certified = worst_case_risk + lipschitz_slack

    return {
        'm': m,
        'epsilon': epsilon,
        'I': I,
        'J_target': J_target,
        'J_completed': j_completed,
        'completed': completed,
        'pct_completed': j_completed / J_target * 100,
        'theta_grid': theta_grid,
        'avg_prior': avg_prior,
        'bayes_risk': bayes_risk,
        'worst_case_risk': worst_case_risk,
        'worst_case_risk_certified': worst_case_risk_certified,
        'lipschitz_slack': lipschitz_slack,
        'upper_bound_2eps': bayes_risk + epsilon,
        'smd_runtime': smd_runtime}
