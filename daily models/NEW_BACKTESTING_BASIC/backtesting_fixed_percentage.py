import math

import pandas as pd


portfolio_history = []

cash_fraction = 0.5


csv_files = [
    "../predictions_basic_lstm_DJA.csv",
    "../predictions_basic_lstm_GSPC.csv",
    "../predictions_basic_lstm_IXIC.csv",
    "../predictions_basic_lstm_NYA.csv"
]

df_list = [pd.read_csv(file, parse_dates=["Date"]) for file in csv_files]
df_predictions = pd.concat(df_list).sort_values(by="Date")
df_predictions.set_index("Date", inplace=True)

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


def rebalance_portfolio(current_date, strategy_type, cash, position):
    next_day = df_predictions.index[df_predictions.index > current_date].min()

    next_day_predictions = df_predictions[df_predictions.index == next_day]

    index_scores = next_day_predictions.set_index("Index")["Predicted_Log_Return"]  # predicted daily returns for each index

    selected_indices = index_scores[index_scores > 0].index.tolist()

    # Sell the non selected_indices
    for index in position.keys():
        if index not in selected_indices:
            close_price_row_sell = df_predictions.loc[(df_predictions.index == current_date) & (df_predictions["Index"] == index)]

            if not close_price_row_sell.empty:
                close_price_sell = close_price_row_sell["Adjusted_Close"].values[0]

                if close_price_sell > 0 and position[index] > 0:
                        cash += close_price_sell * position[index]
                        position[index] = 0

    if selected_indices:
        if strategy_type == "fixed_percentage":
            allocation_per_index = (cash * cash_fraction) / len(selected_indices) if cash > 0 else 0


        for index in selected_indices:
            close_price_row_buy = df_predictions.loc[(df_predictions.index == current_date) & (df_predictions["Index"] == index)]

            if not close_price_row_buy.empty:
                close_price_buy = close_price_row_buy["Adjusted_Close"].values[0]

                if close_price_buy > 0 and allocation_per_index > 0:
                    shares_to_buy = math.floor(allocation_per_index / close_price_buy)
                    position[index] +=  shares_to_buy  # Buy shares
                    cash -= shares_to_buy * close_price_buy


    # print(f"{current_date}: Rebalanced Portfolio | Selected Indices: {selected_indices} | Cash: {cash} | Positions: {position}")
    return cash, position


strategy = "fixed_percentage"

print(f"\n{'-'*20} Running Strategy: {strategy.upper()} {'-'*20}")
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