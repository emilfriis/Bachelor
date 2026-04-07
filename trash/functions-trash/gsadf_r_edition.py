import numpy as np
import pandas as pd


def radf_python(y, minw=None, lags=0, trend="c"):
    """
    Python implementation close to R exuber::radf()
    (fast version using NumPy OLS instead of statsmodels)

    Parameters
    ----------
    y : array-like
        time series

    minw : int
        minimum window size (same as exuber)

    lags : int
        ADF lag length

    trend : str
        "n", "c", "ct"

    Returns
    -------
    dict with
        adf
        sadf
        gsadf
        bsadf path
    """

    y = np.asarray(y).astype(float)
    T = len(y)

    if minw is None:
        minw = int(np.floor(T * (0.01 + 1.8 / np.sqrt(T))))

    adf_mat = np.full((T, T), np.nan)

    for tau2 in range(minw, T + 1):

        for tau1 in range(0, tau2 - minw + 1):

            y_sub = y[tau1:tau2]
            dy = np.diff(y_sub)

            y_lag = y_sub[:-1]

            if lags > 0:

                dy_reg = dy[lags:]
                y_lag_reg = y_lag[lags:]

                lagged_dy = np.column_stack(
                    [dy[lags - i:-i] for i in range(1, lags + 1)]
                )

                X = np.column_stack((y_lag_reg, lagged_dy))

            else:

                dy_reg = dy
                X = y_lag.reshape(-1, 1)

            # deterministic components
            if trend == "c":
                X = np.column_stack((np.ones(len(X)), X))

            elif trend == "ct":
                trend_vec = np.arange(1, len(X) + 1)
                X = np.column_stack((np.ones(len(X)), trend_vec, X))

            # -------- FAST OLS (replaces statsmodels) --------

            XtX = X.T @ X
            XtY = X.T @ dy_reg

            beta = np.linalg.solve(XtX, XtY)

            resid = dy_reg - X @ beta

            sigma2 = resid @ resid / (len(dy_reg) - X.shape[1])

            var_beta = sigma2 * np.linalg.inv(XtX)

            se = np.sqrt(np.diag(var_beta))

            # extract t-statistic on y_{t-1}
            if trend == "n":
                t_stat = beta[0] / se[0]

            elif trend == "c":
                t_stat = beta[1] / se[1]

            elif trend == "ct":
                t_stat = beta[2] / se[2]

            adf_mat[tau1, tau2 - 1] = t_stat

    # SADF
    sadf = np.nanmax(adf_mat[0, minw - 1:])

    # BSADF sequence
    bsadf = np.nanmax(adf_mat[:, minw - 1:], axis=0)

    # GSADF
    gsadf = np.nanmax(adf_mat)

    # Full sample ADF
    adf_full = adf_mat[0, T - 1]

    return {
        "adf": adf_full,
        "sadf": sadf,
        "gsadf": gsadf,
        "bsadf_path": bsadf,
        "adf_matrix": adf_mat,
    }


def radf_mc_cv(T,
                    minw,
                    nrep=2000,
                    lags=0,
                    trend="c",
                    seed=123):
    """
    Monte Carlo critical values for ADF, SADF, GSADF
    similar to exuber::radf_mc_cv()

    Parameters
    ----------
    T : int
        sample size

    minw : int
        minimum window size

    nrep : int
        number of simulations

    lags : int
        ADF lag length

    trend : str
        deterministic specification

    seed : int
        random seed
    """

    np.random.seed(seed)

    adf_stats = []
    sadf_stats = []
    gsadf_stats = []

    for _ in range(nrep):

        # simulate random walk under unit root null
        eps = np.random.normal(size=T)
        y = np.cumsum(eps)

        res = radf_python(
            y,
            minw=minw,
            lags=lags,
            trend=trend
        )

        adf_stats.append(res["adf"])
        sadf_stats.append(res["sadf"])
        gsadf_stats.append(res["gsadf"])

    cv = pd.DataFrame({
        "adf": np.quantile(adf_stats, [0.90, 0.95, 0.99]),
        "sadf": np.quantile(sadf_stats, [0.90, 0.95, 0.99]),
        "gsadf": np.quantile(gsadf_stats, [0.90, 0.95, 0.99])
    })

    cv.index = [90, 95, 99]

    return cv