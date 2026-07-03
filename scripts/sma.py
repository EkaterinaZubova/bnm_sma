"""Stochastic mirror ascent for the bounded normal mean problem.

Contains main algorithm functions:
- delta_pi_numba_core_vec: numba-jitted kernel computing the posterior-mean rule.
- delta_pi_numba: Python wrapper around delta_pi_numba_core_vec.
- stochastic_mirror_ascent: main SMA algorithm; returns the averaged prior.
- bayes_lower_bound: Monte Carlo estimate of the Bayes risk of d_{avg_prior} used as a lower bound on the worst-case risk in the paper.
"""

from typing import Optional
import numpy as np
from numba import jit
from tqdm import trange

@jit(nopython=True)
def delta_pi_numba_core_vec(x_flat, theta_grid, pi, theta_min, theta_max):
    n_x = len(x_flat)
    n_theta = len(theta_grid)
    est = np.empty(n_x)
    step = (theta_grid[n_theta - 1] - theta_grid[0]) / (n_theta - 1) if n_theta > 1 else 1.0

    for i in range(n_x):
        x = x_flat[i]
        k = int(round((x - theta_grid[0]) / step))
        if k < 0:
            k = 0
        elif k >= n_theta:
            k = n_theta - 1
        dk = x - theta_grid[k]
        emax = -0.5 * dk * dk  # largest exponent (at the nearest support point)
        num = 0.0
        den = 0.0
        for j in range(n_theta):
            diff = x - theta_grid[j]
            w = pi[j] * np.exp(-0.5 * diff * diff - emax)
            num += w * theta_grid[j]
            den += w
        mu = num / den if den > 0.0 else theta_grid[k]
        if mu < theta_min:
            mu = theta_min
        elif mu > theta_max:
            mu = theta_max
        est[i] = mu
    return est


def delta_pi_numba(x, theta_grid, pi):
    x = np.atleast_1d(x)
    return delta_pi_numba_core_vec(x, theta_grid, pi, theta_grid[0], theta_grid[-1])


def stochastic_mirror_ascent(
    theta_grid: np.ndarray,
    eta: float,
    J: int,
    n_mc_grad: int = 1,  # number of MC draws for the stochastic supergradient; the paper uses 1!
    rng: Optional[np.random.Generator] = None,
    show_progress: bool = False):
    if rng is None:
        rng = np.random.default_rng()
    I = len(theta_grid)
    pi = np.full(I, 1 / I)
    pi_sum = np.zeros_like(pi)
    iterator = trange(1, J + 1, desc="SMA") if show_progress else range(1, J + 1)

    for j in iterator:
        # accumulate current pi_j for averaging at the end
        pi_sum += pi
        # stochastic supergradient evaluated with 1 MC draw
        x_mc = rng.standard_normal((I, n_mc_grad)) + theta_grid[:, None]
        d_vals = delta_pi_numba(x_mc.ravel(), theta_grid, pi).reshape(I, n_mc_grad)
        grad = np.sum((d_vals - theta_grid[:, None]) ** 2, axis=1) / n_mc_grad
        # multiplicative weights update
        pi *= np.exp(eta * grad)
        pi /= pi.sum()

    avg_prior = pi_sum / J
    return avg_prior


def bayes_lower_bound(theta_grid, avg_prior, n_samples=100, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n_theta = len(theta_grid)
    eps = rng.standard_normal((n_theta, n_samples))
    X = theta_grid[:, None] + eps
    delta = delta_pi_numba(X.ravel(), theta_grid, avg_prior).reshape(n_theta, n_samples)
    risk = ((delta - theta_grid[:, None]) ** 2).mean(axis=1)
    return float(np.dot(avg_prior, risk))
