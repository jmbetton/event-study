import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

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

# Abnormal Returns
# For each day in the event window:
# Expected Return = alpha + beta * R_market
# Abnormal Return = actual return - expected return

event["expected"] = alpha + beta * event["R_market"]
event["AR"] = event["R_stock"] - event["expected"]

# Cumulative Abnormal Return - Sum of ARs across event window
event["CAR"] = event["AR"].cumsum()

print("\n-- Abnormal Returns ---")
print(event[["R_stock", "R_market", "expected", "AR", "CAR"]].round(4))

# ── T-statistic ────────────────────────────────────────────────
# Standard error derived from residuals in estimation window
# This measures how much the CAR deviates from zero relative
# to normal variation in the estimation window

residuals = model.resid
sigma = np.std(residuals, ddof=1)
n_event = len(event)
car_total = event["CAR"].iloc[-1]
t_stat = car_total / (sigma * np.sqrt(n_event))

print(f"\n── Statistical Results ───────────────────────────")
print(f"Total CAR:   {car_total:.4f}  ({car_total*100:.2f}%)")
print(f"Sigma:       {sigma:.6f}")
print(f"T-statistic: {t_stat:.4f}")
print(f"Significant at 5% level: {abs(t_stat) > 1.96}")

# ── Alternative Event Windows ──────────────────────────────────
# Standard practice is to report multiple windows
# The most economically meaningful is the disclosure day itself

windows = {
    "Day 0 only":       event.loc[event.index == EVENT_DATE, "AR"],
    "Day 0 to Day +1":  event.loc[event.index >= EVENT_DATE, "AR"].iloc[:2],
    "Day 0 to Day +2":  event.loc[event.index >= EVENT_DATE, "AR"].iloc[:3],
}

print("\n── Event Window Sensitivity ──────────────────────")
for name, window_returns in windows.items():
    car = window_returns.sum()
    n = len(window_returns)
    t = car / (sigma * np.sqrt(n))
    print(f"{name:<20} CAR: {car*100:>7.2f}%   t-stat: {t:>7.4f}   "
          f"Significant: {abs(t) > 1.96}")
    


# Plot Cumulative Abnormal Returns
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(event.index, event["CAR"] * 100,
        color="crimson", linewidth=2, marker="o", label="CAR (%)")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.axvline(EVENT_DATE, color="navy", linewidth=1.5,
           linestyle="--", label="Disclosure Date (Jan 28, 2026)")

ax.set_title("Cumulative Abnormal Returns - Carvana (CVNA)\n"
             "Gotham City Research Disclosure, Jan 28, 2026",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Cumulative Abnormal Return (%)")
ax.set_xlabel("Date")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("car_chart.png", dpi=150)
print("\nChart saved to car_chart.png")
plt.show()