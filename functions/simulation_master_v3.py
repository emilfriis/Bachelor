# equation for simulation

# P_t = dT^{-eta} + rho P_{t-1} + eps_t, eps ~ N(0, sigma^2)

import numpy as np
import pandas as pd


def simulate_price_with_bubbles(

    # Sample size
    T = 200,

    # Parameters for discount rate and drift scaling
    R = 0.04,
    d = 1.0,
    eta = 1.0,
    
    # Dividend process
    rho = 1.0,
    mu_D = 0.0,
    sigma_D = 0.5,

    # Bubble process
    mu_B = 0.0,
    sigma_B = 0.5, 
    decay = 0.7,

    # Multiple bubbles
    include_bubble = False,
    t_start1 = 75,
    t_end1 = 100,
    B0_1 = 0.0,
    t_start2 = None,
    t_end2 = None,
    B0_2 = 0.0,
    
    # Initial values
    D0 = 0.0,
    B0 = 0.0,

    # Seed
    seed = None,
):

    # Derived dividend drift
    mu = d * (T ** (-eta))

    if seed is not None:
        np.random.seed(seed)

    # Empty arrays to store time series
    D = np.zeros(T)                    # Dividend series
    Pf = np.zeros(T)                   # Fundamental price series
    B = np.zeros(T)                    # Bubble component series
    P = np.zeros(T)                    # Total observed price (fundamental + bubble)

    # Initial values at time t = 0
    D[0] = D0
    Pf[0] = (rho / (1.0 + R - rho)) * D[0] + (mu + mu_D) * (1.0 + R) / (R * (1.0 + R - rho))
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
        # ----- Dividend and fundamental -----
        eps_D = np.random.normal(mu_D, sigma_D)

        D[t] = mu + rho * D[t - 1] + eps_D

        Pf[t] = (rho / (1.0 + R - rho)) * D[t] + (mu + mu_D) * (1.0 + R) / (R * (1.0 + R - rho))

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
                    # Clamp decay into [0,1] to avoid explosions or sign flips
                    d_eff = float(decay)
                    if d_eff <= 0.0:
                        d_eff = 0.0
                    elif d_eff >= 1.0:
                        d_eff = 1.0
                    B[t] = d_eff * B[t - 1] + np.random.normal(0.0, sigma_B)
                    B[t] = max(B[t], 0.0)
                else:
                    B[t] = 0.0

        # ----- Total price -----
        P[t] = Pf[t] + B[t]

    D_series = pd.Series(D, name="Dividend")
    Pf_series = pd.Series(Pf, name="Fundamental")
    B_series = pd.Series(B, name="Bubble")
    P_series = pd.Series(P, name="Price")

    return D_series, Pf_series, B_series, P_series

__all__ = ["simulate_price_with_bubbles"]
