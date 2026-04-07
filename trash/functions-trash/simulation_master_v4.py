# Function for simulating bubble process with P = Pf + B
# D_t = mu + rho*D_{t-1} + eps_{D,t}
# P_t^f = mu/R + P_{t-1}^f + eps_{D,t} / R
# B_t = (1+R)B_{t-1} + eps_{B,t}
# P_t = P_t^f + B_t

from typing import Optional

import numpy as np
import pandas as pd

def simulate_price_with_bubbles(
    T: int = 200,
    R: float = 0.04,
    d: float = 1.0,
    eta: float = 1.0,
    mu: float = 0.0,
    rho: float = 1.0,
    sigma_D: float = 0.5,
    sigma_B: float = 0.5,
    decay: float = 0.7,
    include_bubble: bool = False,
    t_start1: Optional[int] = None,
    t_end1: Optional[int] = None,
    B0_1: float = 0.0,
    t_start2: Optional[int] = None,
    t_end2: Optional[int] = None,
    B0_2: float = 0.0,
    Pf0: float = 0.0,
    B0: float = 0.0,
    seed: Optional[int] = None,
): 

    if seed is not None:
        np.random.seed(seed)

    # Empty arrays to store time series
    D = np.zeros(T)                    # Dividend series
    Pf = np.zeros(T)                   # Fundamental price series
    B = np.zeros(T)                    # Bubble component series
    P = np.zeros(T)                    # Total observed price (fundamental + bubble)

    # Initial values at time t = 0
    Pf[0] = Pf0
    B[0] = B0
    P[0] = Pf[0] + B[0]

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
        Pf[t] = mu/R + Pf[t - 1] + np.random.normal(0.0, sigma_D)/R

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
                    B[t] = (1.0 + R) * B[t - 1] + np.random.normal(0.0, sigma_B)
                    B[t] = max(B[t], 0.0)
            elif in_bubble2:
                if t == t_start2:
                    B[t] = B0_2
                else:
                    B[t] = (1.0 + R) * B[t - 1] + np.random.normal(0.0, sigma_B)
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

                    B[t] = decay * B[t - 1] + np.random.normal(0.0, sigma_B)
                    B[t] = max(B[t], 0.0)
                else:
                    B[t] = 0.0

        # ----- Total price -----
        P[t] = Pf[t] + B[t]

    Pf_series = pd.Series(Pf, name="Fundamental")
    B_series = pd.Series(B, name="Bubble")
    P_series = pd.Series(P, name="Price")
    return Pf_series, B_series, P_series