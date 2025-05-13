import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

selected_indices = {
    "A": "DJA",  # Dow Jones Industrial Average
    "C": "GSPC",  # S&P 500
    "D": "IXIC",  # NASDAQ Composite
    "E": "NYA",  # NYSE Composite
}

indices_conditional = {v: os.path.join(BASE_DIR, "index_data", f"{v}.INDX.csv") for v in selected_indices.values()}


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
    file_path = indices_conditional[index_name]

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    df = df[df.index >= "1950-01-03"]
    df = df.loc[TRAIN_START_DATE:TEST_END_DATE]

    df["Log_Returns_1"] = np.log(df["Adjusted_close"]) - np.log(df["Adjusted_close"].shift(1))
    df["target"] = df["Log_Returns_1"].shift(-1)

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

    df["MA_Crossover"] = (df["SMA_5"] > df["SMA_20"])

    # Extra metadata
    df["index_id"] = index_name
    df["date"] = df.index
    df["day_of_week"] = df["date"].dt.dayofweek.astype(str)

    df.dropna(inplace=True)

    return df


def load_all_indices_tft(TRAIN_START_DATE, TEST_END_DATE):
    all_dfs = []

    for index_name in selected_indices.values():
        print(f"Loading data for {index_name}...")

        df = load_index_data(index_name, TRAIN_START_DATE, TEST_END_DATE)
        all_dfs.append(df)

    df_all = pd.concat(all_dfs).sort_values(["index_id", "date"]).reset_index(drop=True)

    # Add time_idx (per series)
    df_all["time_idx"] = df_all.groupby("index_id").cumcount()

    return df_all


def return_splits_tft(TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE):
    df_all = load_all_indices_tft(TRAIN_START_DATE, TEST_END_DATE)

    df_all["date"] = pd.to_datetime(df_all["date"])

    df_train = df_all[(df_all["date"] >= pd.to_datetime(TRAIN_START_DATE)) & (df_all["date"] <= pd.to_datetime(TRAIN_END_DATE))]
    df_val = df_all[(df_all["date"] >= pd.to_datetime(VALID_START_DATE)) & (df_all["date"] <= pd.to_datetime(VALID_END_DATE))]
    df_test = df_all[(df_all["date"] >= pd.to_datetime(TEST_START_DATE)) & (df_all["date"] <= pd.to_datetime(TEST_END_DATE))]

    # Columns to be used in time_varying_known_reals
    excluded_columns = ["Adjusted_close", "target", "date", "index_id", "time_idx", "day_of_week"]
    feature_columns = [col for col in df_all.columns if col not in excluded_columns]

    print("TFT splits created:")
    print(f"Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}")
    print(f"Feature columns: {feature_columns}")

    return df_train, df_val, df_test, feature_columns


def load_index_monthly_data(index_name, TRAIN_START_DATE, TEST_END_DATE):
    file_path = indices_conditional[index_name]
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    df = df[df.index >= "1950-01-03"]
    df = df.loc[TRAIN_START_DATE:TEST_END_DATE]

    monthly_df = df.resample("ME").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Adjusted_close": "last",
        "Volume": "sum"
    })

    monthly_df["Log_Returns"] = np.log(monthly_df["Adjusted_close"]) - np.log(monthly_df["Adjusted_close"].shift(1))
    monthly_df["target"] = monthly_df["Log_Returns"].shift(-1)

    monthly_df["Log_Returns_2"] = np.log(monthly_df["Adjusted_close"]) - np.log(monthly_df["Adjusted_close"].shift(2))
    monthly_df["Log_Returns_4"] = np.log(monthly_df["Adjusted_close"]) - np.log(monthly_df["Adjusted_close"].shift(4))
    monthly_df["Log_Returns_6"] = np.log(monthly_df["Adjusted_close"]) - np.log(monthly_df["Adjusted_close"].shift(6))

    monthly_df["Volatility"] = monthly_df["Log_Returns"].rolling(window=3).std()
    monthly_df["RSI_6"] = compute_RSI(monthly_df, period=6)

    monthly_df["SMA_2"] = monthly_df["Adjusted_close"].rolling(window=2).mean()
    monthly_df["SMA_6"] = monthly_df["Adjusted_close"].rolling(window=6).mean()
    monthly_df["SMA_12"] = monthly_df["Adjusted_close"].rolling(window=12).mean()

    monthly_df["EMA_2"] = monthly_df["Adjusted_close"].ewm(span=2, adjust=False).mean()
    monthly_df["EMA_6"] = monthly_df["Adjusted_close"].ewm(span=6, adjust=False).mean()

    monthly_df["MA_Crossover"] = monthly_df["SMA_2"] > monthly_df["SMA_6"]

    # Extra metadata
    monthly_df["index_id"] = index_name
    monthly_df["date"] = monthly_df.index
    monthly_df["month"] = monthly_df.index.month.astype(str)

    monthly_df.dropna(inplace=True)
    return monthly_df

def load_all_indices_monthly(TRAIN_START_DATE, TEST_END_DATE):
    all_dfs = []
    for index_name in selected_indices.values():
        print(f"Loading monthly data for {index_name}...")
        df = load_index_monthly_data(index_name, TRAIN_START_DATE, TEST_END_DATE)
        all_dfs.append(df)

    df_all = pd.concat(all_dfs).sort_values(["index_id", "date"]).reset_index(drop=True)

    # Add time_idx (per series)
    df_all["time_idx"] = df_all.groupby("index_id").cumcount()
    return df_all

def return_splits_monthly(TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE):
    df_all = load_all_indices_monthly(TRAIN_START_DATE, TEST_END_DATE)

    df_all["date"] = pd.to_datetime(df_all["date"])

    df_train = df_all[(df_all["date"] >= pd.to_datetime(TRAIN_START_DATE)) & (df_all["date"] <= pd.to_datetime(TRAIN_END_DATE))]
    df_val = df_all[(df_all["date"] >= pd.to_datetime(VALID_START_DATE)) & (df_all["date"] <= pd.to_datetime(VALID_END_DATE))]
    df_test = df_all[(df_all["date"] >= pd.to_datetime(TEST_START_DATE)) & (df_all["date"] <= pd.to_datetime(TEST_END_DATE))]

    # Columns to be used in time_varying_known_reals
    excluded_columns = ["Adjusted_close", "target", "date", "index_id", "time_idx", "month"]
    feature_columns = [col for col in df_all.columns if col not in excluded_columns]

    print("Monthly DeepAR splits created:")
    print(f"Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}")
    print(f"Feature columns: {feature_columns}")

    return df_train, df_val, df_test, feature_columns