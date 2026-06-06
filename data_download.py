import yfinance as yf
import pandas as pd
import os

# ── Event Parameters ───────────────────────────────────────────
# Event: Gotham City Research short-seller report alleging Carvana
# overstated earnings by $1B+ through undisclosed related-party
# transactions. Published January 28, 2026.
# Source: Gotham City Research, "Carvana: Bridgecrest and the
# Undisclosed Transactions and Debts" (Jan 28, 2026)

EVENT_DATE = "2026-01-28"
TICKER = "CVNA"
MARKET = "SPY"

# Estimation window: ~252 trading days ending 30 days before event
# Event window: 5 days before to 5 days after
# Download wide range and slice in later scripts

START_DATE = "2024-01-01"
END_DATE = "2026-02-01"

def fetch_daily_returns(symbol, start, end):
    print(f"Fetching data for {symbol}...")
    
    raw = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for {symbol}. Check ticker.")
    
    returns = raw["Close"].squeeze().pct_change().dropna()
    returns.index.name = "Date"
    returns.name = symbol

    return returns

cvna_returns = fetch_daily_returns(TICKER, START_DATE, END_DATE)
spy_returns = fetch_daily_returns(MARKET, START_DATE, END_DATE)

returns = pd.DataFrame({
    "R_stock": cvna_returns,
    "R_market": spy_returns
}).dropna()

# Validate
print(f"\nData shape: {returns.shape}")
print(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
print(f"\nSample around event date:")
print(returns.loc["2026-01-26":"2026-01-30"])

# Save to CSV
returns.to_csv("returns_data.csv")
print("\nData saved to returns_data.csv")


