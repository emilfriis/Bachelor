# equations for simulation

# D_{t+1} = mu + D_t + eps_{t+1}    # dividend process
# P_t^f   = D_t/R + mu*(1+R)/R^2    # fundamental price
# B_{t+1} = (1+R)B_t                # bubble process
# P_t     = P_t^f + B_t             # stock price

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import pandas as pd


def simulate_price_with_bubbles(
    T: int  = 200,
    rho: float = 1.0,                            # AR(1) coefficient: scalar, or array of length T for rho[t]
    mu_f: float = 0.0,                  # Constant drift in the fundamental
    trend_f: float = 0.0,               # Linear time trend coefficient in the fundamental
    sigma_f: float = 0.5,               # Std. dev. of fundamental shocks
    P0: float = 0.0,                    # Initial fundamental price
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
) -> Tuple[pd.Series, pd.Series, pd.Series]:

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

__all__ = ["simulate_price_with_bubbles"]
