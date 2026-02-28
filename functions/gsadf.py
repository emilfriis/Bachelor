import numpy as np
import statsmodels.api as sm

def GSADF_test(
        y, r0=0.05, lags=0, trend='c'
        ):
    """
    Computes the GSADF statistic:
        GSADF = sup_{r2 ∈ [r0,1], r1 ∈ [0, r2-r0]} ADF(r1, r2)

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
    gsadf_stat : float
        Supremum ADF statistic over all subwindows
    adf_path : list
        All ADF statistics (one per (r1, r2))
    r_path : list
        Corresponding (r1, r2) pairs
    """

    y = np.asarray(y)
    T = len(y)
    tau0 = int(np.floor(r0 * T))

    adf_path = []
    r_path = []

    # Match exuber: first window ends at 1-based index minw+lag+1, so tau2 from tau0+lags+1
    for tau2 in range(tau0 + lags + 1, T + 1):
        for tau1 in range(0, tau2 - tau0 + 1):
            y_sub = y[tau1:tau2]  # Subsample [τ1, τ2)

            dy = np.diff(y_sub)
            y_lag = y_sub[:-1]
            n_eff = len(dy)
            if n_eff < 2:  # need enough obs for OLS (const + y_lag)
                continue

            X = y_lag.reshape(-1, 1)

            if trend == 'c':
                X = sm.add_constant(X)
            elif trend == 'ct':
                trend_vec = np.arange(1, len(y_lag) + 1)
                X = np.column_stack((np.ones(len(y_lag)), trend_vec, y_lag))

            if lags > 0:
                for i in range(1, lags + 1):
                    dy_lag = np.zeros_like(dy)
                    dy_lag[i:] = dy[:-i]
                    X = np.column_stack((X, dy_lag))

            model = sm.OLS(dy, X)
            results = model.fit()

            if trend == 'n':
                t_stat = results.tvalues[0]
            elif trend == 'c':
                t_stat = results.tvalues[1]
            elif trend == 'ct':
                t_stat = results.tvalues[2]

            adf_path.append(t_stat)
            r_path.append((tau1 / T, tau2 / T))

    gsadf_stat = max(adf_path)
    return gsadf_stat, adf_path, r_path