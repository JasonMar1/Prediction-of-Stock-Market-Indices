import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(standardized):
    """Load dataset directly from CSV and preprocess it."""
    file_path = "index_data/GSPC.INDX.csv"
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    # Ignore all data before 1950-01-03
    df = df[df.index >= "1950-01-03"]

    # Compute Log Returns
    df["Log_Returns"] = np.log(df["Close"]) - np.log(df["Close"].shift(1))
    df["Log_Returns_Tomorrow"] = df["Log_Returns"].shift(-1)  # Target variable

    # Drop missing values (first row & last row will have NaN)
    df.dropna(inplace=True)

    features = ["Open", "High", "Low", "Close", "Adjusted_close", "Volume"]

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
    """Load dataset directly from CSV, aggregate to monthly, and preprocess it."""
    file_path = "index_data/GSPC.INDX.csv"
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    # Ignore all data before 1950-01-03
    df = df[df.index >= "1950-01-03"]

    # Aggregate daily data into monthly data.
    # For prices: first Open, max High, min Low, last Close & Adjusted_close.
    # For Volume: sum the values.
    monthly_df = df.resample("M").agg({
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