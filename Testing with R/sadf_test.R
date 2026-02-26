library(quantmod)
library(exuber)

getSymbols("AAPL", src="yahoo", from="2000-01-01")
x <- na.omit(log(Ad(AAPL)))
x <- as.numeric(x)

obj <- radf(x, lag = 0)
cv  <- radf_mc_cv(length(x), nrep = 500, seed = 123, lag = 1)

print(obj)                 # prints test object
print(cv)                  # prints critical values table
print(max(obj$radf))        # prints SADF stat (max RADF)

print(radf_dating(obj, cv = cv))  # prints dating table (may be empty)