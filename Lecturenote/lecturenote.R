# exuber package
library(exuber)

# 1. Load series
dat <- read.csv("/Users/emilrand/Desktop/Uni/Bachelor/6. semester/BA_Emil/BA_repo/data/P_t.csv")
y   <- ts(dat$Price)

# 2. Test statistics (ADF, SADF, GSADF)
res <- radf(y, lag = 0)

# 3. Critical values (Monte Carlo)
T   <- length(y)
minw <- floor(0.05*T) # could also be: floor((0.01 + 1.8 / sqrt(T)) * T)
cv   <- radf_mc_cv(n = T, minw = minw, nrep = 1999, seed = 123)

# Show more decimals (default is 7)
options(digits = 10)
print(res)
print(cv)