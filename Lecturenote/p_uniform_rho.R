library(exuber)

# 1. Load series
dat <- read.csv("/Users/emilrand/Desktop/Uni/Bachelor/6. semester/BA_Emil/BA_repo/data/P_t.csv")
y   <- ts(dat$Price)          # <- column is 'Price'
T   <- length(y)

# 2. Test statistics (ADF, SADF, GSADF)
minw <- floor(0.05 * T)
res  <- radf(y, minw = minw, lag = 0)

# 3. Critical values (Monte Carlo)
cv   <- radf_mc_cv(n = T, minw = minw, nrep = 1999, seed = 123)

options(digits = 10)
print(res)
print(cv)

# BSADF critical values (90%, 95%, 99%) and test statistic
cv$bsadf_cv # cv
res$bsadf   # test statistic

# download BSADF data

# BSADF test statistic over time
bsadf_stat <- data.frame(bsadf = as.numeric(res$bsadf))
write.csv(bsadf_stat, "data/sp500_bsadf_stat.csv", row.names = FALSE)

# BSADF critical values (90, 95, 99)
bsadf_cv <- as.data.frame(cv$bsadf_cv)        # keep all significance-level columns
write.csv(bsadf_cv, "data/sp500_bsadf_cv.csv", row.names = FALSE)