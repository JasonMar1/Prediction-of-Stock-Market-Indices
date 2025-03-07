import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Get the absolute path of the project's root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

selected_indices = {
    "A": "DJA",  # Dow Jones Industrial Average
    "C": "GSPC",  # S&P 500
    "D": "IXIC",  # NASDAQ Composite
    "E": "NYA",  # NYSE Composite
}

indices_long = {v: os.path.join(BASE_DIR, "index_data", f"{v}.INDX.csv") for v in selected_indices.values()}

def load_index_data(index_name, TRAIN_START_DATE, TEST_END_DATE):
    file_path = indices_long[index_name]

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

    df.dropna(inplace=True)

    return df


def long_lstm_load_daily_data(selected_index, standardized, TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE):
    # Load data for the selected index
    df = load_index_data(selected_index, TRAIN_START_DATE, TEST_END_DATE)

    feature_columns = ["Open", "High", "Low", "Adjusted_close", "Volume", "Log_Returns_1", "Log_Returns_5", "Log_Returns_10", "Log_Returns_20"]

    # Split the data by date
    df_train = df.loc[TRAIN_START_DATE:TRAIN_END_DATE]
    df_valid = df.loc[VALID_START_DATE:VALID_END_DATE]
    df_test = df.loc[TEST_START_DATE:TEST_END_DATE]

    X_train = df_train[feature_columns]
    y_train = df_train["Log_Returns_Tomorrow"]

    X_valid = df_valid[feature_columns]
    y_valid = df_valid["Log_Returns_Tomorrow"]

    X_test = df_test[feature_columns]
    y_test = df_test["Log_Returns_Tomorrow"]

    if standardized:
        # Standardize Features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, df_train, df_valid, df_test, feature_columns


def long_lstm_load_multiple_indices(standardized, TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE):
    all_X_train, all_y_train = [], []
    all_X_valid, all_y_valid = [], []
    all_X_test, all_y_test = [], []

    dfs_test = []

    for index_name in selected_indices.values():
        print(f"Loading data for {index_name}...")

        X_train, y_train, X_valid, y_valid, X_test, y_test, df_train, df_valid, df_test, features = long_lstm_load_daily_data(index_name, standardized, TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE)

        # Convert everything to DataFrames to enable date alignment
        X_train, y_train = pd.DataFrame(X_train, index=df_train.loc[TRAIN_START_DATE:TRAIN_END_DATE].index), pd.DataFrame(y_train, index=df_train.loc[TRAIN_START_DATE:TRAIN_END_DATE].index)
        X_valid, y_valid = pd.DataFrame(X_valid, index=df_valid.loc[VALID_START_DATE:VALID_END_DATE].index), pd.DataFrame(y_valid, index=df_valid.loc[VALID_START_DATE:VALID_END_DATE].index)
        X_test, y_test = pd.DataFrame(X_test, index=df_test.loc[TEST_START_DATE:TEST_END_DATE].index), pd.DataFrame(y_test, index=df_test.loc[TEST_START_DATE:TEST_END_DATE].index)

        # Store the individual datasets
        all_X_train.append(X_train)
        all_y_train.append(y_train)
        all_X_valid.append(X_valid)
        all_y_valid.append(y_valid)
        all_X_test.append(X_test)
        all_y_test.append(y_test)

        dfs_test.append(df_test)

    #  Merge all df_test DataFrames into a single DataFrame sorted by date
    df_test_combined = pd.concat(dfs_test, axis=0).sort_index()

    return all_X_train, all_y_train, all_X_valid, all_y_valid, all_X_test, all_y_test, df_test_combined, features


def combine_and_sort_data(all_X_train, all_y_train, all_X_valid, all_y_valid, all_X_test, all_y_test):
    X_train_combined = pd.concat(all_X_train, axis=0).sort_index()
    y_train_combined = pd.concat(all_y_train, axis=0).sort_index()

    X_valid_combined = pd.concat(all_X_valid, axis=0).sort_index()
    y_valid_combined = pd.concat(all_y_valid, axis=0).sort_index()

    X_test_combined = pd.concat(all_X_test, axis=0).sort_index()
    y_test_combined = pd.concat(all_y_test, axis=0).sort_index()

    return X_train_combined, y_train_combined, X_valid_combined, y_valid_combined, X_test_combined, y_test_combined





