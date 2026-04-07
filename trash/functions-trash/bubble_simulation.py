import numpy as np                     # Import NumPy for numerical operations and random draws
import pandas as pd                    # Import pandas for Series objects

def simulate_bubble(                   # Define a function that simulates a price with a temporary bubble
    T=200,                             # Total number of time periods
    mu_f=0.0,                          # Drift in the fundamental (if >0 → random walk with drift)
    sigma_f=0.5,                       # Standard deviation of fundamental shocks
    R=0.04,                            # Growth rate of the bubble (e.g. 4%)
    sigma_b=0.5,                       # Standard deviation of bubble shocks
    t_start=75,                        # Period where bubble begins
    t_end=100,                         # Period where bubble collapses
    P0=0,                             # Initial fundamental price
    B0=0,                              # Initial bubble size
    seed=42                            # Seed for reproducibility
):
    np.random.seed(seed)               # Fix random seed so results are reproducible

    # Empty arrays to store time series
    Pf = np.zeros(T)                   # Fundamental price series
    B  = np.zeros(T)                   # Bubble component series
    P  = np.zeros(T)                   # Total observed price (fundamental + bubble)

    # Initial values at time t = 0
    Pf[0] = P0                         # Set initial fundamental price
    P[0]  = P0                         # Total price equals fundamental at t=0 (no bubble yet)

    # Loop over time
    for t in range(1, T):              # Iterate from period 1 to T-1

        # ----- Fundamental process -----
        Pf[t] = Pf[t-1] + mu_f + np.random.normal(0, sigma_f)
        # Random walk:
        # Previous value + drift + Gaussian shock

        # ----- Bubble process -----
        if t < t_start:                # Before bubble starts
            B[t] = 0                   # No bubble component

        elif t == t_start:             # At the moment bubble begins
            B[t] = B0                  # Initialize bubble at B0

        elif t_start < t <= t_end:     # During bubble period
            B[t] = (1 + R) * B[t-1] + np.random.normal(0, sigma_b)
            # Explosive growth at rate (1+R) plus stochastic shock
            
            B[t] = max(B[t], 0)        # Ensure bubble does not become negative

        else:                          # After bubble collapses
            B[t] = 0                   # Bubble fully bursts

        # ----- Total price -----
        P[t] = Pf[t] + B[t]            # Observed price = fundamental + bubble

    Pf = pd.Series(Pf, name="Fundamental")  # Convert fundamental to pandas Series
    B  = pd.Series(B,  name="Bubble")  # Convert bubble to pandas Series
    P  = pd.Series(P,  name="Price")  # Convert total price to pandas Series


    return Pf, B, P                    # Return all three simulated series