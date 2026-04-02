"""
Bertelsen (2019), section 5.3 — Random walk with explosive periods.

Single bubble (Phillips et al. 2015 / Pedersen & Schütte 2017 style):
    P_t =
      P_{t-1} + eps_t,                     t = 1,...,tau_e-1
      delta_T * P_{t-1} + eps_t,           t = tau_e,...,tau_f
      P_{tau_e} + eps_t,                   t = tau_f+1
      P_{t-1} + eps_t,                     t = tau_f+2,...,T

where delta_T = 1 + c * T^{-alpha}, and eps_t iid ~ N(0, sigma^2).
Bubble timing is set by fractions:
    tau_e = floor(r_e * T)
    tau_f = floor(r_f * T)

Two-bubble variant repeats the explosive regime twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExplosivePeriod:
    """Explosive window specified in fractions of sample length."""

    r_e: float  # start fraction in (0,1)
    r_f: float  # end fraction in (0,1), must satisfy r_f > r_e


def _tau_from_fraction(r: float, T: int) -> int:
    return int(np.floor(r * T))


def simulate_rw_explosive_periods(
    T: int = 200,
    P0: float = 100.0,
    sigma: float = 6.79,
    c: float = 1.0,
    alpha: float = 0.8,
    window1: ExplosivePeriod = ExplosivePeriod(r_e=0.3, r_f=0.75),
    window2: Optional[ExplosivePeriod] = None,
    seed: Optional[int] = None,
    return_delta: bool = False,
) -> Tuple[pd.Series, pd.Series] | pd.Series:
    """
    Simulate the section 5.3 process.

    Parameters match Bertelsen (2019) defaults for the single-bubble case:
      P0=100, sigma=6.79, c=1, alpha=0.8, r_e=0.3, r_f=0.75

    If window2 is provided, simulates two explosive periods (same delta_T).

    Returns
    -------
    P : pd.Series
        Price path of length T with index 0..T-1.

    If return_delta is True:
        returns (P, delta_T_series).
    """

    if T <= 3:
        raise ValueError("T must be > 3.")
    if not (0.0 < alpha):
        raise ValueError("alpha must be > 0.")
    if not (0.0 < window1.r_e < window1.r_f < 1.0):
        raise ValueError("window1 must satisfy 0 < r_e < r_f < 1.")
    if window2 is not None and not (0.0 < window2.r_e < window2.r_f < 1.0):
        raise ValueError("window2 must satisfy 0 < r_e < r_f < 1.")

    rng = np.random.default_rng(seed)

    # delta_T = 1 + c * T^{-alpha}
    delta_T = 1.0 + float(c) * (float(T) ** (-float(alpha)))

    # Convert to integer times (0-based indexing for the array)
    tau_e1 = _tau_from_fraction(window1.r_e, T)
    tau_f1 = _tau_from_fraction(window1.r_f, T)

    if window2 is None:
        tau_e2 = tau_f2 = None
    else:
        tau_e2 = _tau_from_fraction(window2.r_e, T)
        tau_f2 = _tau_from_fraction(window2.r_f, T)

    eps = rng.normal(0.0, float(sigma), size=T)
    P = np.zeros(T)
    P[0] = float(P0)

    # Helper to check if t is inside an explosive window (inclusive)
    def in_window(t: int, te: int, tf: int) -> bool:
        return te <= t <= tf

    for t in range(1, T):
        # special collapse step(s): t == tau_f + 1
        if t == tau_f1 + 1:
            P[t] = P[tau_e1] + eps[t]
            continue
        if window2 is not None and t == tau_f2 + 1:
            P[t] = P[tau_e2] + eps[t]
            continue

        # explosive regimes
        if in_window(t, tau_e1, tau_f1):
            P[t] = delta_T * P[t - 1] + eps[t]
            continue
        if window2 is not None and in_window(t, tau_e2, tau_f2):
            P[t] = delta_T * P[t - 1] + eps[t]
            continue

        # random walk regime
        P[t] = P[t - 1] + eps[t]

    P_s = pd.Series(P, name="P")

    if return_delta:
        delta_s = pd.Series(np.full(T, delta_T), name="delta_T")
        return P_s, delta_s

    return P_s


__all__ = ["ExplosivePeriod", "simulate_rw_explosive_periods"]

