import numpy as np
import statsmodels.api as sm


def GSADF_test(
        y,
        r0=0.05,
        lags=0,
        trend='c'
        ):
    """
    Computes the GSADF statistic.

    GSADF = sup_{r2 ∈ [r0,1], r1 ∈ [0, r2-r0]} ADF(r1, r2)

    The test is based on recursively estimated right-tailed ADF regressions
    over all possible subsamples satisfying the minimum window size.

    Parameters
    ----------
    y : array-like
        Time series of length T (e.g., log price series)

    r0 : float
        Minimum window fraction (0 < r0 < 1).
        Minimum subsample size is tau0 = floor(r0 * T).

    lags : int
        Number of lagged differences in ADF regression.

    trend : str
        Deterministic specification:
        'n'  = no constant
        'c'  = constant
        'ct' = constant + linear trend

    Returns
    -------
    gsadf_stat : float
        Supremum ADF statistic across all admissible subsamples.

    adf_path : list
        List of all ADF statistics.

    r_path : list
        Corresponding (r1, r2) pairs for each ADF regression.
    """

    # Convert to numpy array and ensure float
    y = np.asarray(y, dtype=float)
    T = len(y)

    # Minimum window size tau0
    tau0 = int(np.floor(r0 * T))

    adf_path = []
    r_path = []

    # Loop over ending points tau2
    for tau2 in range(tau0, T + 1):

        # Loop over starting points tau1
        for tau1 in range(0, tau2 - tau0 + 1):

            # Ensure window satisfies minimum length
            if tau2 - tau1 < tau0:
                continue

            # Extract subsample
            y_sub = y[tau1:tau2]

            # Construct ADF regression variables
            dy = np.diff(y_sub)
            y_lag = y_sub[:-1]

            # Skip if too few observations
            if len(dy) <= lags + 1:
                continue

            # Handle lagged differences properly (ADF alignment)
            if lags > 0:

                dy_reg = dy[lags:]
                y_lag_reg = y_lag[lags:]

                dy_lags = []
                for i in range(1, lags + 1):
                    dy_lags.append(dy[lags - i:-i])

                X = np.column_stack([y_lag_reg] + dy_lags[::-1])

            else:

                dy_reg = dy
                X = y_lag.reshape(-1, 1)

            # Add deterministic components
            if trend == 'c':
                X = sm.add_constant(X)

            elif trend == 'ct':
                trend_vec = np.arange(1, len(dy_reg) + 1)
                X = np.column_stack((np.ones(len(dy_reg)), trend_vec, X))

            # Estimate ADF regression
            model = sm.OLS(dy_reg, X)
            results = model.fit()

            # Extract t-statistic on y_{t-1}
            if trend == 'n':
                t_stat = results.tvalues[0]
            elif trend == 'c':
                t_stat = results.tvalues[1]
            elif trend == 'ct':
                t_stat = results.tvalues[2]

            adf_path.append(t_stat)
            r_path.append((tau1 / T, tau2 / T))

    # Supremum across all windows
    gsadf_stat = max(adf_path)

    return gsadf_stat, adf_path, r_path