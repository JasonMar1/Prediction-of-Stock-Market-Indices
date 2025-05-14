import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from backtesting_full_rebalancing import run_strategy as run_full
from backtesting_fixed_percentage import run_strategy as run_fixed
from backtesting_hybrid import run_strategy as run_hybrid

# Run each strategy and get the portfolio history
df_full = run_full()
df_fixed = run_fixed()
df_hybrid = run_hybrid()

# Merge dataframes on Date
df_combined = df_full.merge(df_fixed, on="Date").merge(df_hybrid, on="Date")

# Convert Date to datetime format
df_combined.set_index("Date", inplace=True)

plt.figure(figsize=(12, 8))
plt.plot(df_combined.index, df_combined["full_rebalancing"], label="Full Rebalancing", linestyle="dashed", marker="o")
plt.plot(df_combined.index, df_combined["fixed_percentage"], label="Fixed Percentage", linestyle="solid", marker="s")
plt.plot(df_combined.index, df_combined["hybrid"], label="Hybrid", linestyle="dashdot", marker="d")

# Set x-axis to show only months
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Show every month
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))  # Format as YYYY-MM
plt.xticks(rotation=45)

plt.title("Portfolio Value - Comparison Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid()
plt.show()