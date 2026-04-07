import numpy as np
from math import floor
from numba import njit


def _prepare_data(y):
    y = np.asarray(y, dtype=np.float64)
    dy = np.diff(y)
    return y, dy


@njit
def _ols_tstat(dy, y_lag, start, end, lags, trend):
    """
    Compute ADF t-statistic in window [start,end)
    """
    T = end - start

    nobs = T - 1 - lags
    k = 1 + lags
    if trend == 1:  # c
        k += 1
    elif trend == 2:  # ct
        k += 2

    X = np.zeros((nobs, k))
    Y = np.zeros(nobs)

    for i in range(nobs):
        t = start + lags + 1 + i
        Y[i] = dy[t - 1]

        col = 0

        if trend == 1:
            X[i, col] = 1.0
            col += 1
        elif trend == 2:
            X[i, col] = 1.0
            col += 1
            X[i, col] = i + 1
            col += 1

        X[i, col] = y_lag[t - 1]
        col += 1

        for j in range(lags):
            X[i, col] = dy[t - 2 - j]
            col += 1

    XtX = X.T @ X
    XtY = X.T @ Y
    beta = np.linalg.solve(XtX, XtY)

    resid = Y - X @ beta
    sigma2 = (resid @ resid) / (nobs - k)

    cov = sigma2 * np.linalg.inv(XtX)

    if trend == 0:
        idx = 0
    elif trend == 1:
        idx = 1
    else:
        idx = 2

    se = np.sqrt(cov[idx, idx])
    return beta[idx] / se


@njit
def _radf_core(y, dy, r0, lags, trend):
    T = len(y)
    minw = int(np.floor(r0 * T))

    y_lag = y[:-1]

    sadf = -1e10
    gsadf = -1e10

    adf_full = _ols_tstat(dy, y_lag, 0, T, lags, trend)

    for r2 in range(minw, T + 1):

        stat = _ols_tstat(dy, y_lag, 0, r2, lags, trend)
        if stat > sadf:
            sadf = stat

        for r1 in range(0, r2 - minw + 1):

            stat = _ols_tstat(dy, y_lag, r1, r2, lags, trend)
            if stat > gsadf:
                gsadf = stat

    return adf_full, sadf, gsadf, minw


@njit
def _radf_bsadf_core(y, dy, r0, lags, trend):

    T = len(y)
    minw = int(np.floor(r0 * T))

    y_lag = y[:-1]

    bsadf = np.empty(T)
    bsadf[:] = np.nan

    for r2 in range(minw, T + 1):

        max_stat = -1e10

        for r1 in range(0, r2 - minw + 1):

            stat = _ols_tstat(dy, y_lag, r1, r2, lags, trend)

            if stat > max_stat:
                max_stat = stat

        bsadf[r2 - 1] = max_stat

    return bsadf, minw


def radf(y, r0, lags=0, trend="c"):
    """
    Compute ADF, SADF and GSADF
    """

    trend_map = {"n": 0, "c": 1, "ct": 2}
    trend_id = trend_map[trend]

    y, dy = _prepare_data(y)

    adf, sadf, gsadf, minw = _radf_core(y, dy, r0, lags, trend_id)

    return {
        "adf": adf,
        "sadf": sadf,
        "gsadf": gsadf,
        "minw": minw,
    }


def radf_bsadf(y, r0, lags=0, trend="c"):

    trend_map = {"n": 0, "c": 1, "ct": 2}
    trend_id = trend_map[trend]

    y, dy = _prepare_data(y)

    bsadf, minw = _radf_bsadf_core(y, dy, r0, lags, trend_id)

    return {
        "bsadf": bsadf,
        "minw": minw,
    }


def radf_cv(y, r0, lags=0, trend="c", nrep=1999, seed=None):

    y = np.asarray(y)
    T = len(y)

    rng = np.random.default_rng(seed)

    adf = np.zeros(nrep)
    sadf = np.zeros(nrep)
    gsadf = np.zeros(nrep)

    for i in range(nrep):

        eps = rng.standard_normal(T)
        sim = np.cumsum(eps)

        res = radf(sim, r0, lags, trend)

        adf[i] = res["adf"]
        sadf[i] = res["sadf"]
        gsadf[i] = res["gsadf"]

    return {
        "90": {
            "adf": np.quantile(adf, 0.9),
            "sadf": np.quantile(sadf, 0.9),
            "gsadf": np.quantile(gsadf, 0.9),
        },
        "95": {
            "adf": np.quantile(adf, 0.95),
            "sadf": np.quantile(sadf, 0.95),
            "gsadf": np.quantile(gsadf, 0.95),
        },
        "99": {
            "adf": np.quantile(adf, 0.99),
            "sadf": np.quantile(sadf, 0.99),
            "gsadf": np.quantile(gsadf, 0.99),
        },
    }


def smooth_series(x, window=5): # brugerdefineret funktion til at glatte kritiske værdier, så det er lettere at se trends

    y = x.copy()

    for i in range(len(x)):

        start = max(0, i-window)
        end = min(len(x), i+window+1)

        y[i] = np.nanmean(x[start:end])

    return y


def radf_bsadf_cv(T, r0, lags=0, trend="c", nrep=1999, seed=None):
    """
    Simulate BSADF critical value paths
    """

    rng = np.random.default_rng(seed)

    trend_map = {"n": 0, "c": 1, "ct": 2}
    trend_id = trend_map[trend]

    minw = int(np.floor(r0 * T))

    bsadf_sim = np.zeros((nrep, T))

    for i in range(nrep):

        eps = rng.standard_normal(T)
        sim = np.cumsum(eps)

        y, dy = _prepare_data(sim)

        bsadf, _ = _radf_bsadf_core(y, dy, r0, lags, trend_id)

        bsadf_sim[i] = bsadf

    cv90 = np.nanquantile(bsadf_sim, 0.90, axis=0)
    cv95 = np.nanquantile(bsadf_sim, 0.95, axis=0)
    cv99 = np.nanquantile(bsadf_sim, 0.99, axis=0)

    # smoothing (samme idé som i exuber)
    cv90 = smooth_series(cv90)
    cv95 = smooth_series(cv95)
    cv99 = smooth_series(cv99)


    return {
        "90": cv90,
        "95": cv95,
        "99": cv99,
        "minw": minw
    }