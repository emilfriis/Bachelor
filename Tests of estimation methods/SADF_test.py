import numpy as np
import statsmodels.api as sm

def SADF_test(y, r0=0.05, lags=0, trend='c'):
    """
    Computes the SADF statistic:
        SADF = sup_{r ∈ [r0,1]} ADF(r)

    Parameters
    ----------
    y : array-like
        Time series of length T
    r0 : float
        Minimum window fraction (r0 ∈ (0,1))
    lags : int
        Number of lagged differences in ADF regression
    trend : str
        'n' = no constant
        'c' = constant
        'ct' = constant + trend

    Returns
    -------
    sadf_stat : float
        Supremum ADF statistic
    adf_path : list
        Sequence of ADF(r) statistics
    r_path : list
        Corresponding r values
    """

    y = np.asarray(y)
    T = len(y)

    # Minimum window size τ0 = floor(r0 T)
    tau0 = int(np.floor(r0 * T))

    adf_path = []
    r_path = []

    # Loop over τ = τ0, τ0+1, ..., T
    for tau in range(tau0, T + 1):

        r = tau / T                      # r = τ / T
        y_sub = y[:tau]                  # Subsample of size τ

        # Construct ADF regression
        dy = np.diff(y_sub)
        y_lag = y_sub[:-1]

        X = y_lag.reshape(-1, 1)

        # Add deterministic components if needed
        if trend == 'c':
            X = sm.add_constant(X)

        elif trend == 'ct':
            trend_vec = np.arange(1, len(y_lag)+1)
            X = np.column_stack((np.ones(len(y_lag)), trend_vec, y_lag))

        # Add lagged differences if lags > 0
        if lags > 0:
            for i in range(1, lags + 1):
                dy_lag = np.roll(dy, i)
                dy_lag[:i] = 0
                X = np.column_stack((X, dy_lag))

        model = sm.OLS(dy, X)
        results = model.fit()

        # Extract t-stat on y_{t-1}
        if trend == 'n':
            t_stat = results.tvalues[0]
        elif trend == 'c':
            t_stat = results.tvalues[1]
        elif trend == 'ct':
            t_stat = results.tvalues[2]

        adf_path.append(t_stat)
        r_path.append(r)

    sadf_stat = max(adf_path)

    return sadf_stat, adf_path, r_path