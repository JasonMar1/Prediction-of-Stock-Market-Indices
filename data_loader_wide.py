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


def load_index_data(index_name, TRAIN_START_DATE, TEST_END_DATE):
    """Loads and preprocesses stock index data from CSV"""
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

    df.dropna(inplace=True)

    return df


def wide_lstm_load_daily_data(standardized, TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE):

    features = ["Open", "High", "Low", "Adjusted_close", "Volume"]
    log_return_features = ["Log_Returns_1"]
    # log_return_features = ["Log_Returns_1", "Log_Returns_5", "Log_Returns_10", "Log_Returns_20"]
    all_features = features + log_return_features

    # Load data for all selected indices
    dfs = {index_name: load_index_data(index_name, TRAIN_START_DATE, TEST_END_DATE) for index_name in selected_indices.values()}

    # Align datasets by date (intersection of all available dates)
    common_dates = set.intersection(*(set(df.index) for df in dfs.values()))
    for key in dfs:
        dfs[key] = dfs[key].loc[sorted(common_dates)]  # Keep only common dates

    # Rename columns to include index name as prefix
    for key in dfs:
        dfs[key].columns = [f"{key}_{col}" for col in dfs[key].columns]

    df_combined = pd.concat(dfs.values(), axis=1) # combine the 4 dataframes into one

    feature_columns = [col for col in df_combined.columns if "Log_Returns_Tomorrow" not in col]
    X = df_combined[feature_columns]
    y = np.column_stack([dfs[index_name][f"{index_name}_Log_Returns_Tomorrow"].values for index_name in selected_indices.values()])

    # Convert to DataFrame for easy indexing
    df = pd.DataFrame(X, index=df_combined.index, columns=feature_columns)

    # Split into train, validation, test
    df_train = df.loc[TRAIN_START_DATE:TRAIN_END_DATE]
    df_valid = df.loc[VALID_START_DATE:VALID_END_DATE]
    df_test = df.loc[TEST_START_DATE:TEST_END_DATE]

    y_train = y[:len(df_train)]
    y_valid = y[len(df_train):len(df_train) + len(df_valid)]
    y_test = y[len(df_train) + len(df_valid):]


    # Standardization
    if standardized:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(df_train)
        X_valid = scaler.transform(df_valid)
        X_test = scaler.transform(df_test)
    else:
        X_train, X_valid, X_test = df_train.values, df_valid.values, df_test.values


    return X_train, y_train, X_valid, y_valid, X_test, y_test, df_test, feature_columns

