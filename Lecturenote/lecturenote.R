# exuber package
library(exuber)

# 1. Load series
dat <- read.csv("/Users/emilrand/Desktop/Uni/Bachelor/6. semester/BA_Emil/BA_repo/Lecturenote/P_t.csv")
y   <- ts(dat$Price)

# 2. Test statistics (ADF, SADF, GSADF)
res <- radf(y, lag = 0)

# 3. Critical values (Monte Carlo)
n    <- length(y)
minw <- floor((0.01 + 1.8 / sqrt(n)) * n)
cv   <- radf_mc_cv(n = n, minw = minw, nrep = 1999, seed = 123)

print(res)
print(cv)