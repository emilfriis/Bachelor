import numpy as np
import statsmodels.api as sm


def SADF_test(
        y,
        r0=0.05,
        lags=0,
        trend='c', # 'n' for no trend, 'c' for constant, 'ct' for constant + trend
        simulate_cv=False, # false for just SADF statistic, true for also simulating critical values
        M=2000, # number of simulations for critical values
        seed=None # random seed for reproducibility of critical values
        ):

    y = np.asarray(y, dtype=float)
    T = len(y)
    tau0 = int(np.floor(r0 * T))

    adf_path = []
    r_path = []

    # --------- Compute SADF ---------
    for tau in range(tau0, T + 1):

        r = tau / T
        y_sub = y[:tau]

        dy = np.diff(y_sub)
        y_lag = y_sub[:-1]

        X = y_lag.reshape(-1, 1)

        if trend == 'c':
            X = sm.add_constant(X)

        elif trend == 'ct':
            trend_vec = np.arange(1, len(y_lag) + 1)
            X = np.column_stack((np.ones(len(y_lag)), trend_vec, y_lag))

        if lags > 0:
            for i in range(1, lags + 1):
                dy_lag = np.roll(dy, i)
                dy_lag[:i] = 0
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
        r_path.append(r)

    sadf_stat = max(adf_path)

    output = {
        "SADF": sadf_stat,
        "adf_path": adf_path,
        "r_path": r_path
    }

    # --------- Monte Carlo Critical Values ---------
    if simulate_cv:

        if seed is not None:
            np.random.seed(seed)

        sadf_sim = []

        for _ in range(M):

            eps = np.random.normal(0, 1, T)
            y_sim = np.cumsum(eps)

            # Recursive call WITHOUT simulation
            sim_output = SADF_test(
                y_sim,
                r0=r0,
                lags=lags,
                trend=trend,
                simulate_cv=False
            )

            sadf_sim.append(sim_output["SADF"])

        sadf_sim = np.array(sadf_sim, dtype=float)

        cv = {
            "90%": np.quantile(sadf_sim, 0.90),
            "95%": np.quantile(sadf_sim, 0.95),
            "99%": np.quantile(sadf_sim, 0.99)
        }

        output["critical_values"] = cv
        output["reject_95"] = sadf_stat > cv["95%"] # True if we reject the null hypothesis at the 5% significance level

    return output