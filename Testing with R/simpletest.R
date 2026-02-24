# Packages for
install.packages(c("quantmod", "exuber", "xts"))
library(quantmod) # download data
library(exuber)   # bubble tests
library(xts)

# Choose sample period
start_date <- as.Date("2000-01-01")
end_date   <- Sys.Date()

# Download AAPL from Yahoo
getSymbols("AAPL", src = "yahoo", from = start_date, to = end_date)

# Extract Adjusted close (xts)
px <- Ad(AAPL)

# Convert to log prices (recommended in bubble tests)
x <- log(px)

# Drop any missing values
x <- na.omit(x)

# Quick sanity check
head(x)
tail(x)

# A simple starting point: no ADF lags (lag = 0)
# You can change lag later (see notes below)
obj <- radf(x, lag = 0)

# Inspect what's inside (useful for extracting stats programmatically)
names(obj)
str(obj, max.level = 1)