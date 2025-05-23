import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

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



def expected_return(returns, periods_per_year):
    return returns.mean() * periods_per_year

def volatility(returns, periods_per_year):
    return returns.std() * np.sqrt(periods_per_year)

def sharpe_ratio(returns, risk_free_rate, periods_per_year):
    er = expected_return(returns, periods_per_year)
    vol = volatility(returns, periods_per_year)
    return (er - risk_free_rate) / vol if vol != 0 else np.nan

def sortino_ratio(returns, risk_free_rate, periods_per_year):
    # downside deviation (only negative returns)
    neg = returns[returns < 0]
    downside_dev = np.sqrt((neg**2).mean()) * np.sqrt(periods_per_year)
    er = expected_return(returns, periods_per_year)
    return (er - risk_free_rate) / downside_dev if downside_dev != 0 else np.nan

def max_drawdown(series: pd.Series):
    # series: portfolio value time-series
    cum_max = series.cummax()
    drawdown = (series - cum_max) / cum_max
    return drawdown.min()

periods_per_year = 252
risk_free_rate = 0.01  # 1% ετησίως

results = {}
for strat in ["full_rebalancing", "fixed_percentage", "hybrid"]:
    prices = df_combined[strat]
    rets = prices.pct_change().dropna()
    er  = expected_return(rets, periods_per_year)
    vol = volatility(rets, periods_per_year)
    sr  = sharpe_ratio(rets, risk_free_rate, periods_per_year)
    so  = sortino_ratio(rets, risk_free_rate, periods_per_year)
    mdd = max_drawdown(prices)
    results[strat] = [er, vol, sr, so, mdd]

metrics_df = pd.DataFrame(
    results,
    index=["Expected Return", "Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown"]
).T


pd.set_option('display.max_columns', None)
print("\nPerformance Metrics per Strategy (annualized):")
print(metrics_df.map(lambda x: f"{x:.2%}"))


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