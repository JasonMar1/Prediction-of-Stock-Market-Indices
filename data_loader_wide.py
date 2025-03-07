import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Get the absolute path of the project's root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))



""" WIDE LSTM IMPLEMENTATION """

selected_indices = {
    "A": "DJA",  # Dow Jones Industrial Average
    "C": "GSPC",  # S&P 500
    "D": "IXIC",  # NASDAQ Composite
    "E": "NYA",  # NYSE Composite
}

indices_wide = {v: os.path.join(BASE_DIR, "index_data", f"{v}.INDX.csv") for v in selected_indices.values()}

def compute_RSI(df, period=14):
    delta = df["Adjusted_close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def load_index_data(index_name, TRAIN_START_DATE, TEST_END_DATE):
    file_path = os.path.join(BASE_DIR, "index_data", indices_wide[index_name])

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    df = df[df.index >= "1950-01-03"]
    df = df.loc[TRAIN_START_DATE:TEST_END_DATE]

    """Case 2: Compute Log Returns for 1 shift"""
    df["Log_Returns_1"] = np.log(df["Adjusted_close"]) - np.log(df["Adjusted_close"].shift(1))
    df["Log_Returns_Tomorrow"] = df["Log_Returns_1"].shift(-1)

    """ Extra Features"""
    df["Log_Returns_5"] = np.log(df["Adjusted_close"]) - np.log(df["Adjusted_close"].shift(5))
    df["Log_Returns_10"] = np.log(df["Adjusted_close"]) - np.log(df["Adjusted_close"].shift(10))
    df["Log_Returns_20"] = np.log(df["Adjusted_close"]) - np.log(df["Adjusted_close"].shift(20))

    df["Volatility"] = df["Log_Returns_1"].rolling(window=10).std()
    df["RSI_14"] = compute_RSI(df, period=14)

    df.dropna(inplace=True)

    return df


def wide_lstm_load_daily_data(standardized, TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE):

    # Load data for all selected indices
    dfs = {index_name: load_index_data(index_name, TRAIN_START_DATE, TEST_END_DATE) for index_name in selected_indices.values()}

    # (intersection of all available dates)
    common_dates = set.intersection(*(set(df.index) for df in dfs.values()))
    for key in dfs:
        dfs[key] = dfs[key].loc[sorted(common_dates)]  # Keep only common dates

    # Rename columns to include index name as prefix
    for key in dfs:
        dfs[key].columns = [f"{key}_{col}" for col in dfs[key].columns]

    df_combined = pd.concat(dfs.values(), axis=1) # Combine the 4 dataframes into one

    excluded_columns = ["Log_Returns_Tomorrow", "Open", "High", "Low", "Adjusted_close", "Volume"]
    feature_columns = [col for col in df_combined.columns if not any(column in col for column in excluded_columns)]
    # feature_columns = [col for col in df_combined.columns if "Log_Returns_Tomorrow" not in col]
    target_columns = [col for col in df_combined.columns if "Log_Returns_Tomorrow" in col]

    # Split the data by date
    df_train = df_combined.loc[TRAIN_START_DATE:TRAIN_END_DATE]
    df_valid = df_combined.loc[VALID_START_DATE:VALID_END_DATE]
    df_test = df_combined.loc[TEST_START_DATE:TEST_END_DATE]

    X_train = df_train[feature_columns]
    y_train = df_train[target_columns]

    X_valid = df_valid[feature_columns]
    y_valid = df_valid[target_columns]

    X_test = df_test[feature_columns]
    y_test = df_test[target_columns]

    if standardized:
        # Standardize Features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)


    return X_train, y_train, X_valid, y_valid, X_test, y_test, df_test, feature_columns

