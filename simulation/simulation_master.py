""" 
This module allows for simulations of different variants of an AR(1) process. E.g. a normal random walk with drift, a random walk with drift and a bubble, a random walk with drift and two bubbles, etc. 
"""

# 1. import libraries

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import pandas as pd

# 2. create master function

def simulate_price_with_bubbles(

    # Sample size
    T: int = 432, # 432 corresponds to monthly data from 1990-2025
    
    # Fundamental component
    rho_f: float = 1.0,
    mu_f: float = 0.0,
    sigma_f: float = 0.5,
    Pf0: float = 0.0,

    # Bubble component
    include_bubble: bool = True,
    R: float = 0.04,
    sigma_b: float = 5.0,
    omega: float = 0.7,
    B0: float = 0.0,

    # a. First bubble interval
    tau_s1: Optional[int] = 75,
    tau_e1: Optional[int] = 100,
    B0_1: float = 0.0,

    # b. Second bubble interval
    tau_s2: Optional[int] = None,
    tau_e2: Optional[int] = None,
    B0_2: float = 0.0,

    # c. Third bubble interval
    tau_s3: Optional[int] = None,
    tau_e3: Optional[int] = None,
    B0_3: float = 0.0,

    # Random seed
    seed: Optional[int] = None,
    
) -> Tuple[pd.Series, pd.Series, pd.Series]:

    """
    Simulate a price process with an AR(1) fundamental and up to two temporary bubbles.

    Fundamental:
        P_t^f = mu_f + rho_f * P_{t-1}^f + eps_t^f,
        eps_t^f ~ N(0, sigma_f^2)

    Bubbles (for each active bubble interval j = 1, 2, 3):
        B_t = 0                           before tau_s_j
        B_t = B0_j                        at t == tau_s_j
        B_t = (1+R) B_{t-1} + eps_t^b     for tau_s_j < t <= tau_e_j
        B_t = omega * B_{t-1} + eps_t^b   after tau_e_j, with 0 < omega < 1,
              until the bubble has effectively died out or a new bubble begins.

    Observed price:
        P_t = P_t^f + B_t

    """

    if seed is not None:
        np.random.seed(seed)

    # Empty arrays to store time series
    Pf = np.zeros(T)
    B = np.zeros(T)
    P = np.zeros(T)

    # Initial values at time t = 0
    Pf[0] = Pf0
    B[0] = B0
    P[0] = Pf[0] + B[0]

    # Helper flags for whether bubbles are defined
    has_bubble1 = (
        include_bubble
        and tau_s1 is not None
        and tau_e1 is not None
        and tau_s1 < T
        and tau_s1 <= tau_e1
    )
    has_bubble2 = (
        include_bubble
        and tau_s2 is not None
        and tau_e2 is not None
        and tau_s2 < T
        and tau_s2 <= tau_e2
    )
    has_bubble3 = (
        include_bubble
        and tau_s3 is not None
        and tau_e3 is not None
        and tau_s3 < T
        and tau_s3 <= tau_e3
    )

    # Loop over time
    for t in range(1, T):

        # Fundamental process
        Pf[t] = mu_f + rho_f * Pf[t - 1] + np.random.normal(0.0, sigma_f)

        # Bubble process
        if not include_bubble:
            B[t] = 0.0
        else:
            in_bubble1 = has_bubble1 and tau_s1 <= t <= tau_e1
            in_bubble2 = has_bubble2 and tau_s2 <= t <= tau_e2
            in_bubble3 = has_bubble3 and tau_s3 <= t <= tau_e3

            # Priority: first bubble if overlapping, then second
            if in_bubble1:
                if t == tau_s1:
                    B[t] = B0_1
                else:
                    B[t] = (1.0 + R) * B[t - 1] + np.random.normal(0.0, sigma_b)
                    B[t] = max(B[t], 0.0)
            elif in_bubble2:
                if t == tau_s2:
                    B[t] = B0_2
                else:
                    B[t] = (1.0 + R) * B[t - 1] + np.random.normal(0.0, sigma_b)
                    B[t] = max(B[t], 0.0)
            elif in_bubble3:
                if t == tau_s3:
                    B[t] = B0_3
                else:
                    B[t] = (1.0 + R) * B[t - 1] + np.random.normal(0.0, sigma_b)
                    B[t] = max(B[t], 0.0)
            else:
                # Outside any bubble interval:
                # if there was a bubble in the previous period, let it
                # decay geometrically instead of crashing to zero.
                if B[t - 1] > 0.0:
                    decay = float(omega)
                    # Clamp decay into (0,1) to avoid explosions or sign flips
                    if decay <= 0.0:
                        decay = 0.0
                    elif decay >= 1.0:
                        decay = 1.0
                    B[t] = decay * B[t - 1] + np.random.normal(0.0, sigma_b)
                    B[t] = max(B[t], 0.0)
                else:
                    B[t] = 0.0

        # Total price
        P[t] = Pf[t] + B[t]

    Pf_series = pd.Series(Pf, name="Fundamental")
    B_series = pd.Series(B, name="Bubble")
    P_series = pd.Series(P, name="Price")

    return Pf_series, B_series, P_series


__all__ = [
    "simulate_price_with_bubbles",
]