import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Get the absolute path of the project's root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mapping of available indices to corresponding letters
index_mapping = {
    "A": "DJA",  # Dow Jones Industrial Average
    "B": "DWCF",  # Dow Jones U.S. Total Stock Market Index
    "C": "GSPC",  # S&P 500
    "D": "IXIC",  # NASDAQ Composite
    "E": "NYA",  # NYSE Composite
    "F": "W5000"  # Wilshire 5000
}

# Reverse mapping to retrieve filenames
indices = {v: f"{v}.INDX.csv" for v in index_mapping.values()}


def pick_index():
    print("Pick an index:")
    for letter, index in index_mapping.items():
        print(f"  {letter}) {index}")

    choice = input("Enter the corresponding letter: ").strip().upper()

    if choice not in index_mapping:
        raise ValueError(f"Invalid choice: {choice}. Please select a valid letter.")

    return index_mapping[choice]


def load_daily_data(standardized, TRAIN_START_DATE, TEST_END_DATE):
    index_name = pick_index()
    file_path = os.path.join(BASE_DIR, "index_data", indices[index_name])

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    # Ignore all data before 1950-01-03
    df = df[df.index >= "1950-01-03"]

    df = df[TRAIN_START_DATE:TEST_END_DATE]

    # Compute Daily Log Returns
    df["Log_Returns"] = np.log(df["Close"]) - np.log(df["Close"].shift(1))
    df["Log_Returns_Tomorrow"] = df["Log_Returns"].shift(-1)

    df.dropna(inplace=True)

    features = ["Open", "High", "Low", "Close", "Volume"]

    # Extract Features & Target
    X = df[features]
    y = df["Log_Returns_Tomorrow"]

    if standardized:
        # Standardize Features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Convert to DataFrame
    df_processed = pd.DataFrame(X, index=df.index, columns=features)
    df_processed["y"] = y

    return df_processed


def load_monthly_data(standardized, TRAIN_START_DATE, TEST_END_DATE):
    index_name = pick_index()
    file_path = os.path.join(BASE_DIR, "index_data", indices[index_name])

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    # Ignore all data before 1950-01-03
    df = df[df.index >= "1950-01-03"]

    df = df[TRAIN_START_DATE:TEST_END_DATE]

    # Aggregate daily data into monthly data.
    monthly_df = df.resample("ME").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Adjusted_close": "last",
        "Volume": "sum"
    })

    # Monthly Log Returns
    monthly_df["Log_Returns"] = np.log(monthly_df["Close"]) - np.log(monthly_df["Close"].shift(1))
    monthly_df["Log_Returns_Next_Month"] = monthly_df["Log_Returns"].shift(-1)

    monthly_df.dropna(inplace=True)

    # features = ["Open", "High", "Low", "Close", "Adjusted_close", "Volume"]
    features = ["Close"]
    log_return_columns = ["Log_Returns"]

    # Extract Features & Target
    X = monthly_df[features]
    y = monthly_df["Log_Returns_Next_Month"]

    if standardized:
        if features:
            # Standardize Features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

    # Convert to DataFrame
    df_processed = pd.DataFrame(X, index=monthly_df.index, columns=features)

    if log_return_columns:
        features += log_return_columns
        df_processed[log_return_columns] = monthly_df[log_return_columns]

    df_processed["y"] = y

    return df_processed, features


def load_weekly_data(standardized, TRAIN_START_DATE, TEST_END_DATE):
    index_name = pick_index()
    file_path = os.path.join(BASE_DIR, "index_data", indices[index_name])

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    # Ignore all data before 1950-01-03
    df = df[df.index >= "1950-01-03"]

    df = df[TRAIN_START_DATE:TEST_END_DATE]

    # Aggregate daily data into weekly data.
    weekly_df = df.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Adjusted_close": "last",
        "Volume": "sum"
    })

    # Weekly Log Returns
    weekly_df["Log_Returns"] = np.log(weekly_df["Close"]) - np.log(weekly_df["Close"].shift(1))
    weekly_df["Log_Returns_Next_Week"] = weekly_df["Log_Returns"].shift(-1)

    weekly_df.dropna(inplace=True)

    features = ["Open", "High", "Low", "Close", "Adjusted_close", "Volume"]
    log_return_columns = ["Log_Returns"]

    # Extract Features & Target
    X = weekly_df[features]
    y = weekly_df["Log_Returns_Next_Week"]

    if standardized:
        if features:
            # Standardize Features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

    # Convert to DataFrame
    df_processed = pd.DataFrame(X, index=weekly_df.index, columns=features)

    if log_return_columns:
        features += log_return_columns
        df_processed[log_return_columns] = weekly_df[log_return_columns]

    df_processed["y"] = y

    return df_processed, features


def load_daily_data_log_returns(standardized, TRAIN_START_DATE, TEST_END_DATE):
    index_name = pick_index()
    file_path = os.path.join(BASE_DIR, "index_data", indices[index_name])

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    # Ignore all data before 1950-01-03
    df = df[df.index >= "1950-01-03"]

    df = df[TRAIN_START_DATE:TEST_END_DATE]

    # Case 1: Compute Log Returns for different shifts
    max_shift = 15
    for i in range(1, max_shift + 1):
        df[f"Log_Returns_{i}"] = np.log(df["Close"]) - np.log(df["Close"].shift(i))

    # # Case 2: Compute Log Returns for 1 shift
    # df["Log_Returns_1"] = np.log(df["Close"]) - np.log(df["Close"].shift(1))

    df["Log_Returns_Tomorrow"] = df["Log_Returns_1"].shift(-1)

    df.dropna(inplace=True)

    # Case 1
    log_return_columns = [f"Log_Returns_{i}" for i in range(1, max_shift + 1)]

    # # Case 2
    # log_return_columns = ["Log_Returns_1"]

    # features = ['Close']
    features = []

    # Extract Features & Target
    X = df[features]
    y = df["Log_Returns_Tomorrow"]

    if features:
        if standardized:
            # Standardize Features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

    # Convert to DataFrame
    df_processed = pd.DataFrame(X, index=df.index, columns=features)

    if log_return_columns:
        features += log_return_columns
        df_processed[log_return_columns] = df[log_return_columns]

    df_processed["y"] = y

    return df_processed, features
