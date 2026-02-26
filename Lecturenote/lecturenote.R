# install.packages("exuber")   # run once if not installed
library(exuber)

# 1. Read the simulated price series (from Python)
dat <- read.csv("P_t.csv")     # columns: t, Price
y   <- ts(dat$Price)           # time series object

# 2. Run SADF / right-tailed unit root test
#    (recursive window, lag=0; adjust if you want lags)
res <- radf(
  y,
  lag   = 0,
  window = "recursive"         # SADF-style expanding window
)

# 3. Test statistics
print(res)                     # shows SADF, BSADF, GSADF etc.
summary(res)

# 4. Simulate critical values by Monte Carlo
cv <- crit_vals(
  res,
  nrep = 1999,                 # number of replications
  seed = 123                   # for reproducibility
)

print(cv)
summary(cv)

# 5. (Optional) extract and plot BSADF sequence and dates
# ds <- datestamp(res, cv)
# autoplot(res, cv, shading = ds)