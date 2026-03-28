library(exuber)

# 1. Load series
dat <- read.csv('emil/apple_close.csv')
y   <- ts(dat$AAPL)     # change after $ to the column name
T   <- length(y)

# 2. Test statistics (ADF, SADF, GSADF)
res  <- radf(y, lag = 0)

# 3. Critical values (Monte Carlo)
cv   <- radf_mc_cv(n = T, nrep = 1999, seed = 123)

options(digits = 10)
print(res)
print(cv)