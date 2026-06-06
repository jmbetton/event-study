import yfinance as yf
import pandas as pd

# Event Parameters

EVENT_DATE = "2020-09-10"
TICKER = "NKLA"
MARKET = "^GSPC" # S&P 500

# Estimation Window: 280 to 31 trading days before event
# Event Window: 5 trading days before to 5 days after
# First download then slice

START_DATE = "2019-11-01"
END_DATE = "2020-09-25"

# Download Price Data
print("Downloading Nikola (NKLA) price data...")
lk_raw = yf.download(TICKER, start = START_DATE, end = END_DATE, auto_adjust=True)
print("Downloading S&P 500 price data...")
mkt_raw = yf.download(MARKET, start = START_DATE, end = END_DATE, auto_adjust=True)

# Daily Returns
# Use closing price and comput percentage change day over day
lk_returns = lk_raw["Close"].pct_change().dropna()
mkt_returns = mkt_raw["Close"].pct_change().dropna()

# Combine Returns
returns = pd.DataFrame({
    "R_stock": lk_returns,
    "R_market": mkt_returns
}).dropna()

print(f"\nData shape: {returns.shape}")
print(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
print(f"\nFirst 5 rows:")
print(returns.head())
print(f"\nLast 5 Rows:")
print(returns.tail())

# Save to CSV
returns.to_csv("returns_data.csv")
print("\nData saved to returns_data.csv")

