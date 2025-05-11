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

indices_layer_sharing = {v: os.path.join(BASE_DIR, "index_data", f"{v}.INDX.csv") for v in selected_indices.values()}


def compute_RSI(df, period):
    delta = df["Adjusted_close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def load_index_data(index_name, TRAIN_START_DATE, TEST_END_DATE):
    file_path = indices_layer_sharing[index_name]

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

    df["SMA_5"] = df["Adjusted_close"].rolling(window=5).mean()
    df["SMA_20"] = df["Adjusted_close"].rolling(window=20).mean()
    df["SMA_50"] = df["Adjusted_close"].rolling(window=50).mean()

    df["EMA_10"] = df["Adjusted_close"].ewm(span=10, adjust=False).mean()
    df["EMA_50"] = df["Adjusted_close"].ewm(span=50, adjust=False).mean()

    df["MA_Crossover"] = df["SMA_5"] > df["SMA_20"]

    df["Index"] = index_name
    df.dropna(inplace=True)

    return df


def layer_sharing_load_daily_data(standardized, TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE):
    # Load data for all selected indices
    dfs = {}
    for index_name in selected_indices.values():
        print(f"Loading data for {index_name}...")
        dfs[index_name] = load_index_data(index_name, TRAIN_START_DATE, TEST_END_DATE)

    # (intersection of all available dates)
    common_dates = sorted(
        set.intersection(*(set(df.index) for df in dfs.values()))
    )
    for idx in dfs:
        dfs[idx] = dfs[idx].loc[common_dates]  # Keep only common dates


    excluded_columns = ["Log_Returns_Tomorrow", "Index"]
    feature_columns = [col for col in dfs[next(iter(dfs))].columns if col not in excluded_columns]


    X_train, y_train = [], []
    X_valid, y_valid = [], []
    X_test,  y_test  = [], []

    for idx, df in dfs.items():
        X = df[feature_columns]
        y = df["Log_Returns_Tomorrow"]

        X_train.append(X.loc[TRAIN_START_DATE:TRAIN_END_DATE])
        y_train.append(y.loc[TRAIN_START_DATE:TRAIN_END_DATE])

        X_valid.append(X.loc[VALID_START_DATE:VALID_END_DATE])
        y_valid.append(y.loc[VALID_START_DATE:VALID_END_DATE])

        X_test.append(X.loc[TEST_START_DATE:TEST_END_DATE])
        y_test.append(y.loc[TEST_START_DATE:TEST_END_DATE])

    if standardized:
        scaler = StandardScaler()

        # Combine all training samples across indices
        df_all_train = pd.concat(X_train, axis=0)
        scaler.fit(df_all_train)

        # Split back into separate DataFrames (same structure as before)
        def transform_list(df_list):
            all_concat = pd.concat(df_list, axis=0)
            transformed = scaler.transform(all_concat)
            output = []
            start = 0
            for df in df_list:
                n = len(df)
                block = transformed[start:start + n]
                output.append(pd.DataFrame(block, index=df.index, columns=df.columns))
                start += n
            return output

        # Apply transformation
        X_train = transform_list(X_train)
        X_valid = transform_list(X_valid)
        X_test = transform_list(X_test)

    for i, index_name in enumerate(dfs.keys()):
        print(f"{index_name}: Train={len(X_train[i])}, Valid={len(X_valid[i])}, Test={len(X_test[i])}")

    df_test_combined = pd.concat([df.loc[TEST_START_DATE:TEST_END_DATE].assign(Index=index) for index, df in dfs.items()], axis=0).sort_index()

    return X_train, y_train, X_valid, y_valid, X_test, y_test, df_test_combined, feature_columns


