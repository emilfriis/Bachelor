# piecewise process (single bubble window):
# P_t = d*T^{-eta} + P_{t-1} + eps_t,              t = 1,...,tau_e-1
# P_t = d*T^{-eta} + rho*P_{t-1} + eps_t,          t = tau_e,...,tau_f
# P_t = d*T^{-eta} + P_{tau_e} + eps_t,            t = tau_f+1
# P_t = d*T^{-eta} + P_{t-1} + eps_t,              t = tau_f+2,...,T
# eps_t ~ N(0, sigma^2)

import numpy as np
import pandas as pd


def simulate_piecewise_bubble_process(
    T=200,
    d=1.0,
    eta=1.0,
    rho=1.03,          # explosive coefficient inside bubble window
    sigma=0.5,
    P0=0.0,
    # choose bubble window either directly (tau_e/tau_f) or by fractions (r_e/r_f)
    tau_e=None,
    tau_f=None,
    r_e=0.3,
    r_f=0.75,
    seed=None,
    return_meta=False,
):
    """
    Simulate the piecewise process shown in your equation.

    If tau_e/tau_f are not provided, they are set from fractions:
      tau_e = floor(r_e*T), tau_f = floor(r_f*T)
    """
    if seed is not None:
        np.random.seed(seed)

    mu = d * (T ** (-eta))
    if tau_e is None:
        tau_e = int(np.floor(r_e * T))
    if tau_f is None:
        tau_f = int(np.floor(r_f * T))
    tau_e = max(1, min(int(tau_e), T - 2))
    tau_f = max(tau_e, min(int(tau_f), T - 2))

    P = np.zeros(T)
    P[0] = P0

    for t in range(1, T):
        eps_t = np.random.normal(0.0, sigma)
        if t < tau_e:
            # random walk regime
            P[t] = mu + P[t - 1] + eps_t
        elif tau_e <= t <= tau_f:
            # explosive regime
            P[t] = mu + rho * P[t - 1] + eps_t
        elif t == tau_f + 1:
            # one-step collapse to P_{tau_e}
            P[t] = mu + P[tau_e] + eps_t
        else:
            # back to random walk regime
            P[t] = mu + P[t - 1] + eps_t

    P_series = pd.Series(P, name="Price")

    if return_meta:
        meta = {
            "mu": mu,
            "tau_e": tau_e,
            "tau_f": tau_f,
            "rho": rho,
        }
        return P_series, meta
    return P_series


def simulate_piecewise_bubble_process_two(
    T=200,
    d=1.0,
    eta=1.0,
    rho=1.03,          # explosive coefficient inside both bubble windows
    sigma=0.5,
    P0=0.0,
    # choose windows either directly (taus) or by fractions
    tau_e1=None,
    tau_f1=None,
    tau_e2=None,
    tau_f2=None,
    r_e1=0.1,
    r_f1=0.4,
    r_e2=0.6,
    r_f2=0.9,
    seed=None,
    return_meta=False,
):
    """
    Simulate the two-bubble piecewise process:

      t = 1,...,tau_e1-1:         random walk
      t = tau_e1,...,tau_f1:      explosive (rho*P_{t-1})
      t = tau_f1+1:               collapse to P_{tau_e1}
      t = tau_f1+2,...,tau_e2-1:  random walk
      t = tau_e2,...,tau_f2:      explosive (rho*P_{t-1})
      t = tau_f2+1:               collapse to P_{tau_e2}
      t = tau_f2+2,...,T:         random walk
    """
    if seed is not None:
        np.random.seed(seed)

    mu = d * (T ** (-eta))

    if tau_e1 is None:
        tau_e1 = int(np.floor(r_e1 * T))
    if tau_f1 is None:
        tau_f1 = int(np.floor(r_f1 * T))
    if tau_e2 is None:
        tau_e2 = int(np.floor(r_e2 * T))
    if tau_f2 is None:
        tau_f2 = int(np.floor(r_f2 * T))

    tau_e1 = max(1, min(int(tau_e1), T - 4))
    tau_f1 = max(tau_e1, min(int(tau_f1), T - 3))
    tau_e2 = max(tau_f1 + 2, min(int(tau_e2), T - 2))
    tau_f2 = max(tau_e2, min(int(tau_f2), T - 2))

    P = np.zeros(T)
    P[0] = P0

    for t in range(1, T):
        eps_t = np.random.normal(0.0, sigma)

        if t < tau_e1:
            P[t] = mu + P[t - 1] + eps_t
        elif tau_e1 <= t <= tau_f1:
            P[t] = mu + rho * P[t - 1] + eps_t
        elif t == tau_f1 + 1:
            P[t] = mu + P[tau_e1] + eps_t
        elif tau_f1 + 2 <= t <= tau_e2 - 1:
            P[t] = mu + P[t - 1] + eps_t
        elif tau_e2 <= t <= tau_f2:
            P[t] = mu + rho * P[t - 1] + eps_t
        elif t == tau_f2 + 1:
            P[t] = mu + P[tau_e2] + eps_t
        else:
            P[t] = mu + P[t - 1] + eps_t

    P_series = pd.Series(P, name="Price")

    if return_meta:
        meta = {
            "mu": mu,
            "tau_e1": tau_e1,
            "tau_f1": tau_f1,
            "tau_e2": tau_e2,
            "tau_f2": tau_f2,
            "rho": rho,
        }
        return P_series, meta
    return P_series


__all__ = ["simulate_piecewise_bubble_process", "simulate_piecewise_bubble_process_two"]
