import pandas as pd
from data_loader import load_daily_data_log_returns




cash = 100000
position = {"DJA": 0, "GSPC": 0, "IXIC": 0, "NYA": 0}
portfolio_value = []


TRAIN_START_DATE = "2006-01-01"
TRAIN_END_DATE = "2019-12-31"

VALID_START_DATE = "2020-01-01"
VALID_END_DATE = "2023-01-23"

TEST_START_DATE = "2023-01-24"
TEST_END_DATE = "2025-01-24"

# X_train, y_train, X_valid, y_valid, X_test, y_test, df_test, features = load_daily_data_log_returns(False, TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE)

df_predictions = pd.read_csv("predictions.csv", parse_dates=["Date"], index_col="Date")

print(df_predictions)

def rebalance_portfolio(current_date):
    global cash, position
    last_month_predictions = {"DJA": 0, "GSPC": 0, "IXIC": 0, "NYA": 0}

    # Select past month's predictions
    last_month_predictions = df_predictions[df_predictions.index.month == previous_month]

    # Sum predictions for each index based on the "Index" column
    index_scores = last_month_predictions.groupby("Index")["Predicted_Log_Return"].sum()

    # print(f'Index Scores: {index_scores}')

    # Select indices with positive total predicted return
    selected_indices = index_scores[index_scores > 0].index.tolist()

    if selected_indices:
        # Equal allocation
        allocation_per_index = cash / len(selected_indices)

        # Update positions based on the closing price
        for index in selected_indices:
            close_price_row = df_predictions.loc[(df_predictions.index == current_date) & (df_predictions["Index"] == index)]

            if not close_price_row.empty:
                try:
                    close_price = close_price_row["Adjusted_Close"].values[0]  # Extract price

                    if close_price > 0:  # Avoid division errors
                        position[index] = allocation_per_index / close_price  # Buy shares

                except IndexError:
                    print(f"Warning: No closing price found for {index} on {current_date}")
        cash = 0  # All cash is allocated

    print(f"{current_date}: Rebalanced Portfolio | Selected Indices: {selected_indices} | Cash: {cash} | Positions: {position}")


previous_month = None
for current_date in df_predictions.index:
    current_month = current_date.month
    if previous_month is not None and current_month != previous_month:
        rebalance_portfolio(current_date)
    previous_month = current_month