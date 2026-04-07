import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def ADF_test(y, lags=0, trend='c'):
    """
    Computes the ADF statistic over the whole sample and Dickey-Fuller critical values:
        Δy_t = α + β y_{t-1} + ... (with optional trend and lagged Δy).

    Parameters
    ----------
    y : array-like
        Time series of length T
    lags : int
        Number of lagged differences in ADF regression
    trend : str
        'n' = no constant
        'c' = constant
        'ct' = constant + trend

    Returns
    -------
    adf_stat : float
        ADF t-statistic on the lagged level (y_{t-1}) coefficient
    critical_values : dict
        Dickey-Fuller type critical values: '1%', '5%', '10%' (MacKinnon 1994)
    """

    y = np.asarray(y)
    dy = np.diff(y)
    y_lag = y[:-1]

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
        adf_stat = results.tvalues[0]
    elif trend == 'c':
        adf_stat = results.tvalues[1]
    elif trend == 'ct':
        adf_stat = results.tvalues[2]

    # Critical values from Dickey-Fuller distribution (MacKinnon 1994, same as adfuller)
    _adf_result = adfuller(y, maxlag=lags, regression=trend, autolag=None)
    adf_cv = _adf_result[4]  # dict: '1%', '5%', '10%'

    return adf_stat, adf_cv