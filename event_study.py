import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib as plt

returns = pd.read_csv("returns_data.csv", index_col="Date", parse_dates=True)

EVENT_DATE = pd.Timestamp("2026-01-28")

# Find integer position of the event date in the index
event_idx = returns.index.get_loc(EVENT_DATE)

# Estimation Window
est_start = event_idx - 252 - 30
est_end = event_idx - 30

# Event Window
evt_start = event_idx - 5
evt_end = event_idx + 5

# Slice Windows
estimation = returns.iloc[est_start:est_end]
event = returns.iloc[evt_start:evt_end + 1] 

print(f"Estimation window: {estimation.index[0].date()} to {estimation.index[-1].date()}")
print(f"Number of trading days: {len(estimation)}")
print(f"\nEvent window: {event.index[0].date()} to {event.index[-1].date()}")
print(f"Number of trading days: {len(event)}")

# Market Model Regression
# R_Stock = alpha + beta * R_market + epsilon
# Run OLS on estimation window

X = sm.add_constant(estimation["R_market"])
y = estimation["R_stock"]

model = sm.OLS(y, X).fit()

alpha = model.params["const"]
beta = model.params["R_market"]
r_squared = model.rsquared

print(f"\n── Market Model Results ──────────────────────────")
print(f"Alpha:     {alpha:.6f}")
print(f"Beta:      {beta:.4f}")
print(f"R-squared: {r_squared:.4f}")
print(f"\nFull regression summary:")
print(model.summary())
