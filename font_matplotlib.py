# imports
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

# LaTeX font settings
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

start_date = "2020-01-01"
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA"]
labels  = ["Apple", "Google", "Microsoft", "Amazon", "Meta", "Nvidia", "Tesla"]
colors = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

# Download data
stocks = [yf.Ticker(t).history(start=start_date) for t in tickers]

# Example vertical line date
vline_date = pd.Timestamp("2023-01-01")

# Create 7 subplots in one figure (smaller size)
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12,4))
axes = axes.flatten()

for ax, stock, label, ticker, color in zip(axes, stocks, labels, tickers, colors):
    ax.plot(stock.index, stock["Close"], color=color, lw=0.4)
    ax.axvline(x=vline_date, color='black', lw=0.5, ls='--')
    ax.tick_params(axis='both', labelsize=10)
    ax.set_title(label, fontsize=10)

# Remove empty subplot
fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()