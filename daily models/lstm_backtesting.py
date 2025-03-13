import pandas as pd


cash = 100000
position = {"DJA": 0, "GSPC": 0, "IXIC": 0, "NYA": 0}
portfolio_value = []


df_predictions = pd.read_csv("predictions.csv", parse_dates=["Date"], index_col="Date")

# print(df_predictions)

def rebalance_portfolio(current_date):
    global cash, position

    # Select past month's predictions
    last_month_predictions = df_predictions[df_predictions.index.month == previous_month]

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

                    if close_price_sell > 0:  # Avoid division errors
                        if position[index] > 0:
                            cash += close_price_sell * position[index]
                            position[index] = 0  # Sell all the shares for the index

                except IndexError:
                    print(f"Warning: No closing price found for {index} on {current_date}")

    if selected_indices:
        print(f'Positions: {position}')
        # Equal allocation
        allocation_per_index = cash / len(selected_indices) if cash > 0 else 0  # Avoid division by zero

        # Update positions based on the closing price
        for index in selected_indices:
            close_price_row_buy = df_predictions.loc[(df_predictions.index == current_date) & (df_predictions["Index"] == index)]

            if not close_price_row_buy.empty:
                try:
                    close_price_buy = close_price_row_buy["Adjusted_Close"].values[0]

                    if close_price_buy > 0 and allocation_per_index > 0:  # Avoid division errors
                        position[index] = allocation_per_index / close_price_buy  # Buy shares

                except IndexError:
                    print(f"Warning: No closing price found for {index} on {current_date}")

        if allocation_per_index > 0:
            cash = 0  # All cash is allocated

    print(f"{current_date}: Rebalanced Portfolio | Selected Indices: {selected_indices} | Cash: {cash} | Positions: {position}")


previous_month = None
for current_date in df_predictions.index:
    current_month = current_date.month
    if previous_month is not None and current_month != previous_month:
        rebalance_portfolio(current_date)
    previous_month = current_month