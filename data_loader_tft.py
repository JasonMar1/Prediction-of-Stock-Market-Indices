import os
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

selected_indices = {
    "A": "DJA",
    "C": "GSPC",
    "D": "IXIC",
    "E": "NYA",
}

def compute_RSI(df, period=14):
    delta = df["Adjusted_close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def preprocess_tft_data(start_date, end_date, prediction_shift=1):
    all_data = []

    for symbol in selected_indices.values():
        filepath = os.path.join(BASE_DIR, "index_data", f"{symbol}.INDX.csv")
        df = pd.read_csv(filepath, parse_dates=["Date"])
        df = df[df["Date"] >= pd.to_datetime("1950-01-03")]
        df = df.set_index("Date").loc[start_date:end_date].copy()

        # Feature Engineering
        df["Log_Returns_1"] = np.log(df["Adjusted_close"]) - np.log(df["Adjusted_close"].shift(1))
        df["target"] = df["Log_Returns_1"].shift(-prediction_shift)

        df["Log_Returns_5"] = np.log(df["Adjusted_close"]) - np.log(df["Adjusted_close"].shift(5))
        df["Log_Returns_10"] = np.log(df["Adjusted_close"]) - np.log(df["Adjusted_close"].shift(10))
        df["Log_Returns_20"] = np.log(df["Adjusted_close"]) - np.log(df["Adjusted_close"].shift(20))

        df["Volatility"] = df["Log_Returns_1"].rolling(window=10).std()
        df["RSI_14"] = compute_RSI(df)

        df["SMA_5"] = df["Adjusted_close"].rolling(window=5).mean()
        df["SMA_20"] = df["Adjusted_close"].rolling(window=20).mean()
        df["SMA_50"] = df["Adjusted_close"].rolling(window=50).mean()

        df["EMA_10"] = df["Adjusted_close"].ewm(span=10, adjust=False).mean()
        df["EMA_50"] = df["Adjusted_close"].ewm(span=50, adjust=False).mean()

        df["MA_Crossover"] = (df["SMA_5"] > df["SMA_20"]).astype(int)

        # Static and known features
        df["stock_id"] = symbol
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month

        all_data.append(df)

    full_df = pd.concat(all_data).sort_index()
    full_df.dropna(inplace=True)
    return full_df.reset_index().rename(columns={"index": "date"})
