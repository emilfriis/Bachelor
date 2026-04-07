# A simpler version using the build in adfuller function from statsmodels

import numpy as np
from statsmodels.tsa.stattools import adfuller


def ADF_test_v2(y, lags: int = 0, trend: str = "c"):
    """
    Thin wrapper around statsmodels' adfuller to compute the
    ADF test statistic and Dickey–Fuller critical values using
    the same implementation as in your BSADF code.

    Parameters
    ----------
    y : array-like
        Time series of length T.
    lags : int, default 0
        Maximum lag length passed to adfuller (maxlag argument).
    trend : {'n', 'c', 'ct'}, default 'c'
        'n'  = no constant
        'c'  = constant
        'ct' = constant + linear trend

    Returns
    -------
    adf_stat : float
        ADF test statistic from adfuller (element 0).
    critical_values : dict
        Dickey–Fuller critical values from adfuller (element 4),
        keys: '1%', '5%', '10%'.
    """

    y = np.asarray(y)

    # Use the same underlying routine (adfuller) for both statistic and CVs.
    # We fix autolag=None so that the lag choice is fully controlled by `lags`,
    # which matches how you already used adfuller for critical values.
    result = adfuller(y, maxlag=lags, regression=trend, autolag=None)
    adf_stat = result[0]
    adf_cv = result[4]

    return adf_stat, adf_cv