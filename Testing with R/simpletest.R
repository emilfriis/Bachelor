library(quantmod)
library(exuber)
library(xts)

getSymbols("AAPL", src="yahoo", from="2000-01-01", to=Sys.Date())

# 1) Brug KUN Adjusted close og gør den til 1-kolonne xts
x <- Ad(AAPL)
x <- na.omit(x)

# sikkerhed: sørg for 1 kolonne
x <- x[, 1, drop = FALSE]

# 2) log-priser (stadig xts)
xlog <- log(x)

# 3) SADF / RADF
obj <- radf(xlog, lag = 0)

# 4) Kritiske værdier + dating
cv  <- radf_cv(nrow(xlog), lag = 0)
ind <- radf_dating(obj, cv = cv)

ind