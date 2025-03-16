import pandas as pd


portfolio_history = []

investment_fraction = 0.5


df_predictions = pd.read_csv("../predictions.csv", parse_dates=["Date"], index_col="Date")

# print(df_predictions)


def calculate_portfolio_value(current_date, cash, position): # without selling any positions

    total_value = cash  # available cash
    for index, shares in position.items():
        close_price_row = df_predictions.loc[(df_predictions.index == current_date) & (df_predictions["Index"] == index)]
        # print(close_price_row)
        if not close_price_row.empty:
            close_price = close_price_row["Adjusted_Close"].values[0]  # convert the pd series to np array
            total_value += shares * close_price
    return total_value


def rebalance_portfolio(current_date, previous_month, previous_year, strategy_type, cash, position):

    # Select past month's predictions
    last_month_predictions = df_predictions[(df_predictions.index.month == previous_month) & (df_predictions.index.year == previous_year)]

    pd.set_option("display.float_format", "{:.15f}".format)  # Show up to 15 decimals (to make sure the values are calculated correctly)

    # print(f'last month predictions: {last_month_predictions}')

    # Sum predictions for each index based on the "Index" column
    index_scores = last_month_predictions.groupby("Index")["Predicted_Log_Return"].sum()

    # print(f'Index Scores: {index_scores}')

    # Select indices with positive total predicted return
    selected_indices = index_scores[index_scores > 0].index.tolist()

    # Sell the non selected_indices
    for index in position.keys():
        if index not in selected_indices:
            close_price_row_sell = df_predictions.loc[(df_predictions.index == current_date) & (df_predictions["Index"] == index)]

            if not close_price_row_sell.empty:
                try:
                    close_price_sell = close_price_row_sell["Adjusted_Close"].values[0]

                    if close_price_sell > 0 and position[index] > 0:
                            cash += close_price_sell * position[index]
                            position[index] = 0

                except IndexError:
                    print(f"Warning: No closing price found for {index} on {current_date}")

    if selected_indices:
        if strategy_type == "fixed_percentage":
            allocation_per_index = (cash * investment_fraction) / len(selected_indices) if cash > 0 else 0

        # Update positions based on the closing price
        for index in selected_indices:
            close_price_row_buy = df_predictions.loc[(df_predictions.index == current_date) & (df_predictions["Index"] == index)]

            if not close_price_row_buy.empty:
                try:
                    close_price_buy = close_price_row_buy["Adjusted_Close"].values[0]

                    if close_price_buy > 0 and allocation_per_index > 0:
                        position[index] += allocation_per_index / close_price_buy  # Buy shares
                        cash -= allocation_per_index

                except IndexError:
                    print(f"Warning: No closing price found for {index} on {current_date}")


    # print(f"{current_date}: Rebalanced Portfolio | Selected Indices: {selected_indices} | Cash: {cash} | Positions: {position}")
    return cash, position

previous_month = None
previous_year = None
strategy = "fixed_percentage"

print(f"\n{'='*20} Running Strategy: {strategy.upper()} {'='*20}")
cash = 100000
position = {"DJA": 0, "GSPC": 0, "IXIC": 0, "NYA": 0}

for current_date in df_predictions.index:
    current_month = current_date.month
    current_year = current_date.year
    if previous_month is not None and (current_month != previous_month or current_year != previous_year):
        cash, position = rebalance_portfolio(current_date, previous_month, previous_year, strategy, cash, position)

        monthly_portfolio_value = calculate_portfolio_value(current_date, cash, position)
        portfolio_history.append((current_date, monthly_portfolio_value))

    previous_month = current_month
    previous_year = current_year

final_portfolio_value = calculate_portfolio_value(df_predictions.index[-1], cash, position)

print(f"\nFinal Portfolio Value: ${final_portfolio_value:.2f}")

def run_strategy():
    # Return the portfolio history as a DataFrame
    return pd.DataFrame(portfolio_history, columns=["Date", strategy])
