import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Get the absolute path of the project's root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mapping of available indices to corresponding letters
INDEX_MAPPING = {
    "A": "DJA",  # Dow Jones Industrial Average
    "B": "DWCF",  # Dow Jones Wilshire 5000 Total Market Index
    "C": "GSPC",  # S&P 500
    "D": "IXIC",  # NASDAQ Composite
    "E": "NYA",  # NYSE Composite Index
    "F": "W5000"  # Wilshire 5000 Total Market Index
}

# Reverse mapping to retrieve filenames
AVAILABLE_INDICES = {v: f"{v}.INDX.csv" for v in INDEX_MAPPING.values()}


def pick_index():
    print("Pick an index:")
    for letter, index in INDEX_MAPPING.items():
        print(f"  {letter}) {index}")

    choice = input("Enter the corresponding letter: ").strip().upper()

    if choice not in INDEX_MAPPING:
        raise ValueError(f"Invalid choice: {choice}. Please select a valid letter.")

    return INDEX_MAPPING[choice]


def load_data(standardized):
    """Load dataset for the user-selected index and preprocess it."""
    index_name = pick_index()

    file_path = os.path.join(BASE_DIR, "index_data", AVAILABLE_INDICES[index_name])

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    # Ignore all data before 1950-01-03
    df = df[df.index >= "1950-01-03"]

    # Compute Log Returns
    df["Log_Returns"] = np.log(df["Close"]) - np.log(df["Close"].shift(1))
    df["Log_Returns_Tomorrow"] = df["Log_Returns"].shift(-1)  # Target variable

    # Drop missing values (first row & last row will have NaN)
    df.dropna(inplace=True)

    features = ["Open", "High", "Low", "Close", "Volume"]

    # Extract Features & Target
    X = df[features]
    y_reg = df["Log_Returns_Tomorrow"]

    if standardized:
        # Standardize Features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Convert to DataFrame
    df_processed = pd.DataFrame(X, index=df.index, columns=features)
    df_processed["y_reg"] = y_reg  # Add target variable

    return df_processed


def load_monthly_data(standardized):
    """Load dataset for the user-selected index, aggregate to monthly, and preprocess it."""
    index_name = pick_index()

    file_path = os.path.join(BASE_DIR, "index_data", AVAILABLE_INDICES[index_name])

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    # Ignore all data before 1950-01-03
    df = df[df.index >= "1950-01-03"]

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
    monthly_df["Log_Returns_Tomorrow"] = monthly_df["Log_Returns"].shift(-1)
    monthly_df.dropna(inplace=True)

    features = ["Open", "High", "Low", "Close", "Adjusted_close", "Volume"]

    # Extract Features & Target
    X = monthly_df[features]
    y_reg = monthly_df["Log_Returns_Tomorrow"]

    if standardized:
        # Standardize Features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Convert to DataFrame
    df_processed = pd.DataFrame(X, index=monthly_df.index, columns=features)
    df_processed["y_reg"] = y_reg

    return df_processed


def load_weekly_data(standardized):
    index_name = pick_index()
    file_path = os.path.join(BASE_DIR, "index_data", AVAILABLE_INDICES[index_name])

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    # Ignore all data before 1950-01-03
    df = df[df.index >= "1950-01-03"]

    # Weekly Log Returns
    weekly_df = df.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Adjusted_close": "last",
        "Volume": "sum"
    })

    weekly_df["Log_Returns"] = np.log(weekly_df["Close"]) - np.log(weekly_df["Close"].shift(1))
    weekly_df["Log_Returns_Tomorrow"] = weekly_df["Log_Returns"].shift(-1)
    weekly_df.dropna(inplace=True)

    features = ["Open", "High", "Low", "Close", "Adjusted_close", "Volume"]
    X = weekly_df[features]
    y_reg = weekly_df["Log_Returns_Tomorrow"]

    if standardized:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    df_processed = pd.DataFrame(X, index=weekly_df.index, columns=features)
    df_processed["y_reg"] = y_reg

    return df_processed


def load_data_log_returns(standardized, TRAIN_START_DATE, TEST_END_DATE):
    """Load dataset for the user-selected index and preprocess it."""
    index_name = pick_index()

    file_path = os.path.join(BASE_DIR, "index_data", AVAILABLE_INDICES[index_name])

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    # Ignore all data before 1950-01-03
    df = df[df.index >= "1950-01-03"]


    df = df[TRAIN_START_DATE:TEST_END_DATE]

    # Compute Log Returns for different shifts
    max_shift = 5
    for i in range(1, max_shift + 1):
        df[f"Log_Returns_{i}"] = np.log(df["Close"]) - np.log(df["Close"].shift(i))

    df[f"Log_Returns_{1}"] = np.log(df["Close"]) - np.log(df["Close"].shift(1))

    df["Log_Returns_Tomorrow"] = df["Log_Returns_1"].shift(-1)  # Target variable

    # Drop missing values (first row & last row will have NaN)
    df.dropna(inplace=True)

    log_return_columns = [f"Log_Returns_{i}" for i in range(1, max_shift + 1)]
    # log_return_columns = ["Log_Returns_1"]

    # features = ['Close']
    features = []
    all_features = features + log_return_columns

    # Extract Features & Target
    X = df[features]
    y_reg = df["Log_Returns_Tomorrow"]

    if standardized:
        if features:
            # Standardize Features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        else:
            print('empty')

    # Convert to DataFrame
    df_processed = pd.DataFrame(X, index=df.index, columns=features)

    df_processed[log_return_columns] = df[log_return_columns]


    df_processed["y_reg"] = y_reg  # Add target variable

    print(df_processed)
    return df_processed, all_features
