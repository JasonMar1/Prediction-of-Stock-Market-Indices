import math

import pandas as pd


portfolio_history = []


df_predictions = pd.read_csv("../Daily Predictions/predictions_wide_lstm.csv", parse_dates=["Date"], index_col="Date")

# print(df_predictions)


def calculate_portfolio_value(current_date, cash, position): # without selling any positions

    total_value = cash  # available cash
    for index, shares in position.items():
        close_price_row = df_predictions.loc[current_date]
        # print(close_price_row)
        if not close_price_row.empty:
            close_price = close_price_row[f"{index}_Adjusted_Close"]  # convert the pd series to np array
            total_value += shares * close_price
    return total_value


def rebalance_portfolio(current_date, strategy_type, cash, position):
    transaction_fee = 0.001  # 10 basis points = 0.1%

    next_day = df_predictions.index[df_predictions.index > current_date].min()

    if pd.isna(next_day):
        return cash, position  # No future predictions available

    next_day_predictions = df_predictions.loc[next_day, ["Predicted_DJA", "Predicted_GSPC", "Predicted_IXIC", "Predicted_NYA"]]

    # Select indices with positive predicted return for buying
    selected_indices = next_day_predictions[next_day_predictions > 0].index.str.replace("Predicted_", "").tolist()

    # Sell everything
    for index in position.keys():
        if strategy_type == "full_rebalancing":
            close_price_sell = df_predictions.loc[current_date, f"{index}_Adjusted_Close"]
            if close_price_sell > 0 and position[index] > 0:
                sale_amount = close_price_sell * position[index]
                fee = sale_amount * transaction_fee
                cash += sale_amount - fee
                position[index] = 0

    if selected_indices:
        if strategy_type == "full_rebalancing":
            allocation_per_index = cash / len(selected_indices) if cash > 0 else 0

        # Update positions based on the closing price
        for index in selected_indices:
            close_price_buy = df_predictions.loc[current_date, f"{index}_Adjusted_Close"]

            if close_price_buy > 0 and allocation_per_index > 0:
                shares_to_buy = math.floor(allocation_per_index / close_price_buy)
                adjusted_allocation = allocation_per_index / (1 + transaction_fee)

                shares_to_buy = math.floor(adjusted_allocation / close_price_buy)
                total_cost = shares_to_buy * close_price_buy
                fee = total_cost * transaction_fee
                total_spent = total_cost + fee
                if total_spent <= cash:
                    position[index] = shares_to_buy  # Buy shares
                    cash -= total_spent

    # print(f"{current_date}: Rebalanced Portfolio | Selected Indices: {selected_indices} | Cash: {cash} | Positions: {position}")
    return cash, position

strategy = "full_rebalancing"

print(f"\n{'='*20} Running Strategy: {strategy.upper()} {'='*20}")
cash = 100000
position = {"DJA": 0, "GSPC": 0, "IXIC": 0, "NYA": 0}

for current_date in df_predictions.index.unique():
        cash, position = rebalance_portfolio(current_date, strategy, cash, position)

        portfolio_value = calculate_portfolio_value(current_date, cash, position)
        portfolio_history.append((current_date, portfolio_value))

final_portfolio_value = calculate_portfolio_value(df_predictions.index[-1], cash, position)

print(f"\nFinal Portfolio Value: ${final_portfolio_value:.2f}")

def run_strategy():
    # Return the portfolio history as a DataFrame
    return pd.DataFrame(portfolio_history, columns=["Date", strategy])