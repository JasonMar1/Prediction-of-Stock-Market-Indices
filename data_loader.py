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

    print(f'Selected index: {index_mapping[choice]}')
    return index_mapping[choice]

def df_creation(TRAIN_START_DATE, TEST_END_DATE):
    index_name = pick_index()
    file_path = os.path.join(BASE_DIR, "index_data", indices[index_name])

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    # Ignore all data before 1950-01-03
    df = df[df.index >= "1950-01-03"]

    df = df.loc[TRAIN_START_DATE:TEST_END_DATE]

    return df


def load_daily_data(standardized, TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE):
    df = df_creation(TRAIN_START_DATE, TEST_END_DATE)

    # Compute Daily Log Returns
    df["Log_Returns"] = np.log(df["Adjusted_close"]) - np.log(df["Adjusted_close"].shift(1))
    df["Log_Returns_Tomorrow"] = df["Log_Returns"].shift(-1)

    df.dropna(inplace=True)

    features = ["Open", "High", "Low", "Close", "Volume"]
    log_return_columns = ["Log_Returns"]

    # Split the data by date
    df_train = df.loc[TRAIN_START_DATE:TRAIN_END_DATE]
    df_test = df.loc[TEST_START_DATE:TEST_END_DATE]

    X_train = df_train[features]
    y_train = df_train["Log_Returns_Tomorrow"]

    X_test = df_test[features]
    y_test = df_test["Log_Returns_Tomorrow"]

    if standardized:
        # Standardize Features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    X_train = np.hstack([X_train, df_train[log_return_columns]])
    X_test = np.hstack([X_test, df_test[log_return_columns]])

    return X_train, y_train, X_test, y_test, df_test


def load_monthly_data(standardized, TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE):
    df = df_creation(TRAIN_START_DATE, TEST_END_DATE)

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
    monthly_df["Log_Returns"] = np.log(monthly_df["Adjusted_close"]) - np.log(monthly_df["Adjusted_close"].shift(1))
    monthly_df["Log_Returns_Next_Month"] = monthly_df["Log_Returns"].shift(-1)

    monthly_df.dropna(inplace=True)

    # features = ["Open", "High", "Low", "Close", "Adjusted_close", "Volume"]
    features = ["Close"]
    log_return_columns = ["Log_Returns"]

    # Split the data by date
    df_train = monthly_df.loc[TRAIN_START_DATE:TRAIN_END_DATE]
    df_test = monthly_df.loc[TEST_START_DATE:TEST_END_DATE]

    X_train = df_train[features]
    y_train = df_train["Log_Returns_Next_Month"]

    X_test = df_test[features]
    y_test = df_test["Log_Returns_Next_Month"]

    if standardized:
        if features:
            # Standardize Features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

    X_train = np.hstack([X_train, df_train[log_return_columns]])
    X_test = np.hstack([X_test, df_test[log_return_columns]])

    return X_train, y_train, X_test, y_test, df_test


def load_weekly_data(standardized, TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE):
    df = df_creation(TRAIN_START_DATE, TEST_END_DATE)

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
    weekly_df["Log_Returns"] = np.log(weekly_df["Adjusted_close"]) - np.log(weekly_df["Adjusted_close"].shift(1))
    weekly_df["Log_Returns_Next_Week"] = weekly_df["Log_Returns"].shift(-1)

    weekly_df.dropna(inplace=True)

    features = ["Open", "High", "Low", "Close", "Adjusted_close", "Volume"]
    log_return_columns = ["Log_Returns"]

    # Split the data by date
    df_train = weekly_df.loc[TRAIN_START_DATE:TRAIN_END_DATE]
    df_test = weekly_df.loc[TEST_START_DATE:TEST_END_DATE]

    X_train = df_train[features]
    y_train = df_train["Log_Returns_Next_Week"]

    X_test = df_test[features]
    y_test = df_test["Log_Returns_Next_Week"]

    if standardized:
        if features:
            # Standardize Features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

    X_train = np.hstack([X_train, df_train[log_return_columns]])
    X_test = np.hstack([X_test, df_test[log_return_columns]])

    return X_train, y_train, X_test, y_test, df_test


def compute_RSI(df, period=14):
    delta = df["Adjusted_close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def load_daily_data_log_returns(standardized, TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE):
    log_return_columns = []

    df = df_creation(TRAIN_START_DATE, TEST_END_DATE)

    """Case 1: Compute Log Returns for different shifts"""
    # max_shift = 15
    # for i in range(1, max_shift + 1):
    #     df[f"Log_Returns_{i}"] = np.log(df["Adjusted_close"]) - np.log(df["Adjusted_close"].shift(i))

    """Case 2: Compute Log Returns for 1 shift"""
    df["Log_Returns_1"] = np.log(df["Adjusted_close"]) - np.log(df["Adjusted_close"].shift(1))

    df["Log_Returns_Tomorrow"] = df["Log_Returns_1"].shift(-1)

    """ Extra Features"""
    df["Log_Returns_5"] = np.log(df["Adjusted_close"]) - np.log(df["Adjusted_close"].shift(5))
    df["Log_Returns_10"] = np.log(df["Adjusted_close"]) - np.log(df["Adjusted_close"].shift(10))
    df["Log_Returns_20"] = np.log(df["Adjusted_close"]) - np.log(df["Adjusted_close"].shift(20))
    log_return_columns += ["Log_Returns_5", "Log_Returns_10", "Log_Returns_20"]

    df["Volatility"] = df["Log_Returns_1"].rolling(window=10).std()  # Θεωρείται data-leakage αν δεν κοπεί πληροφορία στις πρώτες μέρες του validation, test set?
    df["RSI_14"] = compute_RSI(df, period=14)

    df.dropna(inplace=True)

    """Case 1"""
    # log_return_columns = [f"Log_Returns_{i}" for i in range(1, max_shift + 1)]

    """Case 2"""
    log_return_columns += ["Log_Returns_1"]

    """Extra Features"""
    extra_features = ["RSI_14"] + ["Volatility"]

    """Features"""
    # features = ['Close']
    # features = []
    features = ["Open", "High", "Low", "Adjusted_close", "Volume"]

    # Split the data by date
    df_train = df.loc[TRAIN_START_DATE:TRAIN_END_DATE]
    df_valid = df.loc[VALID_START_DATE:VALID_END_DATE]
    df_test = df.loc[TEST_START_DATE:TEST_END_DATE]


    X_train = df_train[features]
    y_train = df_train["Log_Returns_Tomorrow"]

    X_valid = df_valid[features]
    y_valid = df_valid["Log_Returns_Tomorrow"]

    X_test = df_test[features]
    y_test = df_test["Log_Returns_Tomorrow"]


    if features:
        if standardized:
            # Standardize Features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_valid = scaler.transform(X_valid)
            X_test = scaler.transform(X_test)


    answer = input('\nStandardize the extra features? (y/n): ').strip().lower()
    if answer  == 'y':
        print('yes')

        scaler_extra = StandardScaler()
        X_train_extra = scaler_extra.fit_transform(df_train[extra_features])
        X_valid_extra = scaler_extra.transform(df_valid[extra_features])
        X_test_extra = scaler_extra.transform(df_test[extra_features])

        X_train = np.hstack([X_train, X_train_extra])
        X_valid = np.hstack([X_valid, X_valid_extra])
        X_test = np.hstack([X_test, X_test_extra])

        leftover_features = log_return_columns

        X_train = np.hstack([X_train, df_train[leftover_features]])
        X_valid = np.hstack([X_valid, df_valid[leftover_features]])
        X_test = np.hstack([X_test, df_test[leftover_features]])

        features += extra_features + leftover_features

    elif answer == 'n':
        print('no')
        extra_features += log_return_columns
        X_train = np.hstack([X_train, df_train[extra_features]])
        X_valid = np.hstack([X_valid, df_valid[extra_features]])
        X_test = np.hstack([X_test, df_test[extra_features]])

        features += extra_features

    # print(df_train[extra_features].head(2))
    #
    # print(df_valid[extra_features].tail(10))
    # print(df_valid["Log_Returns_Tomorrow"].tail(10))
    #
    # print(df_test[extra_features].head())
    # print(df_test["Log_Returns_Tomorrow"].head())


    return X_train, y_train, X_valid, y_valid, X_test, y_test, df_test, features
