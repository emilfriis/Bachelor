"""
Collection of simulation utilities for simple time series models.

For now this focuses on different variants of an AR(1) process.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def simulate_ar1_price(
    T: int = 200,
    rho: float = 0.8,
    sigma: float = 1.0,
    mu: float = 0.0,
    P0: float = 0.0,
    seed: Optional[int] = None,
) -> pd.Series:
    """
    Simulate a plain AR(1) price process:

        P_t = mu + rho * P_{t-1} + eps_t,  eps_t ~ N(0, sigma^2)

    Parameters
    ----------
    T : int
        Number of time periods.
    rho : float
        AR(1) coefficient.
    sigma : float
        Standard deviation of the shock eps_t.
    mu : float
        Constant (drift) term.
    P0 : float
        Initial price P_0.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.Series
        Simulated price path P_t of length T, indexed from 0..T-1.
    """
    if seed is not None:
        np.random.seed(seed)

    eps = np.random.normal(0.0, sigma, size=T)
    P = np.empty(T)
    P[0] = P0

    for t in range(1, T):
        P[t] = mu + rho * P[t - 1] + eps[t]

    return pd.Series(P, name="Price")


def simulate_price_with_bubbles(
    T: int = 200,
    # Fundamental AR(1) parameters
    rho=1.0,                            # AR(1) coefficient:
                                        #   - scalar → constant over time
                                        #   - array of length T → rho[t] time-varying
    mu_f: float = 0.0,                  # Constant drift in the fundamental
    trend_f: float = 0.0,               # Linear time trend coefficient in the fundamental
    sigma_f: float = 0.5,               # Std. dev. of fundamental shocks
    P0: float = 0.0,                    # Initial fundamental price
    # Time-varying rho options
    random_rho: bool = False,           # If True, draw time-varying rho[t] ~ U(rho_min, rho_max)
    rho_min: float = 0.5,               # Lower bound for random rho[t]
    rho_max: float = 1.0,               # Upper bound (≤1 ⇒ no explosive AR)
    rho_block_size: int = 10,           # New rho draw every `rho_block_size` periods
    # Bubble parameters (up to two bubbles)
    include_bubble: bool = True,        # If False, no bubble term is added (pure AR(1) fundamental)
    R: float = 0.04,                    # Growth rate of the bubble (e.g. 4%)
    sigma_b: float = 0.5,               # Std. dev. of bubble shocks
    bubble_decay: float = 0.7,          # Fraction of bubble that survives each period after the peak (0<decay<1)
    # First bubble interval
    t_start1: Optional[int] = 75,       # Start of first bubble (inclusive)
    t_end1: Optional[int] = 100,        # End of first bubble (inclusive)
    B0_1: float = 0.0,                  # Initial size of first bubble
    # Second bubble interval
    t_start2: Optional[int] = None,     # Start of second bubble (inclusive), or None for no second bubble
    t_end2: Optional[int] = None,       # End of second bubble (inclusive)
    B0_2: float = 0.0,                  # Initial size of second bubble
    # Random seed
    seed: Optional[int] = None,         # Seed for reproducibility (None → no fixed seed)
    return_rho: bool = False,           # If True, also return the time-varying rho_t as a Series
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Simulate a price process with an AR(1) fundamental and up to two temporary bubbles.

    Fundamental:
        P_t^f = mu_f + trend_f * t + rho_t * P_{t-1}^f + eps_t^f,
        eps_t^f ~ N(0, sigma_f^2)

    Bubbles (for each active bubble interval j = 1, 2):
        B_t = 0                           before t_start_j
        B_t = B0_j                        at t == t_start_j
        B_t = (1+R) B_{t-1} + eps_t^b     for t_start_j < t <= t_end_j
        B_t = decay * B_{t-1} + eps_t^b   after t_end_j, with 0 < decay < 1,
              until the bubble has effectively died out or a new bubble begins.

    Observed price:
        P_t = P_t^f + B_t

    The AR(1) coefficient rho_t can be:
      - constant (scalar rho, random_rho=False),
      - provided as an array of length T,
      - or random, piecewise-constant, uniform on [rho_min, rho_max],
        with a new draw every `rho_block_size` periods (random_rho=True).
    """

    if seed is not None:
        np.random.seed(seed)

    # ----- Time-varying rho handling -----
    if random_rho:
        block_size = max(int(rho_block_size), 1)
        n_blocks = int(np.ceil(T / block_size))
        rho_blocks = np.random.uniform(rho_min, rho_max, size=n_blocks)
        rho_t = np.repeat(rho_blocks, block_size)[:T]
    else:
        if np.isscalar(rho):
            rho_t = np.full(T, float(rho))
        else:
            rho_arr = np.asarray(rho, dtype=float)
            if rho_arr.size != T:
                raise ValueError(
                    f"rho must be scalar or array of length T={T}, got length {rho_arr.size}"
                )
            rho_t = rho_arr

    # Empty arrays to store time series
    Pf = np.zeros(T)                   # Fundamental price series
    B = np.zeros(T)                    # Bubble component series
    P = np.zeros(T)                    # Total observed price (fundamental + bubble)

    # Initial values at time t = 0
    Pf[0] = P0
    P[0] = P0

    # Helper flags for whether bubbles are defined
    has_bubble1 = (
        include_bubble
        and t_start1 is not None
        and t_end1 is not None
        and t_start1 < T
        and t_start1 <= t_end1
    )
    has_bubble2 = (
        include_bubble
        and t_start2 is not None
        and t_end2 is not None
        and t_start2 < T
        and t_start2 <= t_end2
    )

    # Loop over time
    for t in range(1, T):
        # ----- Fundamental process -----
        Pf[t] = mu_f + trend_f * t + rho_t[t] * Pf[t - 1] + np.random.normal(0.0, sigma_f)

        # ----- Bubble process -----
        if not include_bubble:
            B[t] = 0.0
        else:
            in_bubble1 = has_bubble1 and t_start1 <= t <= t_end1
            in_bubble2 = has_bubble2 and t_start2 <= t <= t_end2

            # Priority: first bubble if overlapping, then second
            if in_bubble1:
                if t == t_start1:
                    B[t] = B0_1
                else:
                    B[t] = (1.0 + R) * B[t - 1] + np.random.normal(0.0, sigma_b)
                    B[t] = max(B[t], 0.0)
            elif in_bubble2:
                if t == t_start2:
                    B[t] = B0_2
                else:
                    B[t] = (1.0 + R) * B[t - 1] + np.random.normal(0.0, sigma_b)
                    B[t] = max(B[t], 0.0)
            else:
                # Outside any bubble interval:
                # if there was a bubble in the previous period, let it
                # decay geometrically instead of crashing to zero.
                if B[t - 1] > 0.0:
                    decay = float(bubble_decay)
                    # Clamp decay into (0,1) to avoid explosions or sign flips
                    if decay <= 0.0:
                        decay = 0.0
                    elif decay >= 1.0:
                        decay = 1.0
                    B[t] = decay * B[t - 1] + np.random.normal(0.0, sigma_b)
                    B[t] = max(B[t], 0.0)
                else:
                    B[t] = 0.0

        # ----- Total price -----
        P[t] = Pf[t] + B[t]

    Pf_series = pd.Series(Pf, name="Fundamental")
    B_series = pd.Series(B, name="Bubble")
    P_series = pd.Series(P, name="Price")
    rho_series = pd.Series(rho_t, name="rho_t")

    if return_rho:
        return Pf_series, B_series, P_series, rho_series

    return Pf_series, B_series, P_series


__all__ = [
    "simulate_ar1_price",
    "simulate_price_with_bubbles",
]