"""
Partially collapsing bubble simulation (Bertelsen, 2019, section 5.2).

Implements:
  Dividend (random walk with drift):
    D_t = mu + D_{t-1} + eps^D_t,    eps^D_t ~ N(0, sigma_D^2)

  Fundamental price:
    F_t = mu * rho / (1 - rho)^2 + rho/(1 - rho) * D_t

  Bubble process (threshold + partial collapses):
    If B_{t-1} <= alpha:
      B_t = (1 + r) * B_{t-1} * eps^B_t
    If B_{t-1} > alpha:
      B_t = [ delta + pi^{-1} (1 + r) * theta_t * (B_{t-1} - (1+r)^{-1} delta) ] * eps^B_t
    theta_t ~ Bernoulli(pi)
    eps^B_t = exp(y_t * sigma_B - sigma_B^2/2), y_t ~ N(0,1)  => E[eps^B_t] = 1

  Observed price:
    P_t = F_t + eta * B_t

Note: In Bertelsen (2019) rho is a discount factor with rho^{-1} = 1 + r.
By default we set r = 1/rho - 1 to satisfy that relationship.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def simulate_partially_collapsing_bubble(
    T: int = 200,
    # Parameters (Bertelsen / Phillips et al. defaults)
    mu: float = 0.0024,
    rho: float = 0.985,
    alpha: float = 1.0,
    pi: float = 0.85,
    delta: float = 0.5,
    sigma_D: float = 0.0316,
    sigma_B: float = 0.05,
    eta: float = 20.0,
    # Optional override: bubble growth rate r. If None, use r = 1/rho - 1.
    r: Optional[float] = None,
    # Initial values
    D0: float = 0.0,
    B0: float = 0.0,
    # RNG
    seed: Optional[int] = None,
    # If True, also return theta_t and eps_B_t
    return_components: bool = False,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series] | Tuple[
    pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series
]:
    """
    Returns
    -------
    D, F, B, P : pd.Series
        Dividend, fundamental, bubble, and observed price.

    If return_components is True:
        also returns theta and eps_B as pd.Series.
    """

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if r is None:
        r = (1.0 / rho) - 1.0

    if not (0.0 < rho < 1.0):
        raise ValueError("rho must be in (0,1) as a discount factor.")
    if not (0.0 < pi <= 1.0):
        raise ValueError("pi must be in (0,1].")
    if not (0.0 < delta < (1.0 + r) * alpha):
        raise ValueError("delta must satisfy 0 < delta < (1+r)*alpha.")

    D = np.zeros(T)
    F = np.zeros(T)
    B = np.zeros(T)
    P = np.zeros(T)

    theta = np.zeros(T)
    eps_B = np.ones(T)

    D[0] = D0
    B[0] = B0

    # Fundamental depends on current D_t
    F[0] = mu * rho / ((1.0 - rho) ** 2) + (rho / (1.0 - rho)) * D[0]
    P[0] = F[0] + eta * B[0]

    inv_1pr = 1.0 / (1.0 + r)

    for t in range(1, T):
        # Dividend
        D[t] = mu + D[t - 1] + rng.normal(0.0, sigma_D)

        # Fundamental price
        F[t] = mu * rho / ((1.0 - rho) ** 2) + (rho / (1.0 - rho)) * D[t]

        # Bubble shock (positive, mean 1)
        y = rng.normal(0.0, 1.0)
        eps_B[t] = float(np.exp(y * sigma_B - (sigma_B**2) / 2.0))

        if B[t - 1] <= alpha:
            B[t] = (1.0 + r) * B[t - 1] * eps_B[t]
        else:
            theta[t] = 1.0 if rng.random() < pi else 0.0
            core = delta + (1.0 / pi) * (1.0 + r) * theta[t] * (B[t - 1] - inv_1pr * delta)
            B[t] = core * eps_B[t]

        P[t] = F[t] + eta * B[t]

    D_s = pd.Series(D, name="D")
    F_s = pd.Series(F, name="F")
    B_s = pd.Series(B, name="B")
    P_s = pd.Series(P, name="P")

    if return_components:
        theta_s = pd.Series(theta, name="theta")
        epsB_s = pd.Series(eps_B, name="eps_B")
        return D_s, F_s, B_s, P_s, theta_s, epsB_s

    return D_s, F_s, B_s, P_s


__all__ = ["simulate_partially_collapsing_bubble"]

