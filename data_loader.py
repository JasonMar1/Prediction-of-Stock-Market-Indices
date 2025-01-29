import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle  # To save processed data for reuse

def load_data():
    # Load dataset
    file_path = "index_data/GSPC.INDX.csv"
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    # Ignore all data before 1950-01-03
    df = df[df.index >= "1950-01-03"]

    # Drop missing values
    df.dropna(inplace=True)

    # Create target variables
    df["Close_Tomorrow"] = df["Close"].shift(-1)  # Next day's Close
    df["Target"] = (df["Close_Tomorrow"] > df["Close"]).astype(int)  # Binary classification (1 = Close will be increased, 0 = Close will be decreased)

    # Features and targets
    features = ["Open", "High", "Low", "Close", "Adjusted_close", "Volume"]

    X = df[features]
    y_reg = df["Close_Tomorrow"]  # Regression target
    y_clf = df["Target"]  # Classification target

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save full dataset (without splitting) for flexible training
    with open("processed_data_standardized.pkl", "wb") as f:
        pickle.dump({
            "X": X_scaled,
            "y_reg": y_reg,
            "y_clf": y_clf,
            "dates": df.index
        }, f)

    # Save non-standardized data
    with open("processed_data_non_standardized.pkl", "wb") as f:
        pickle.dump({
            "X": X,
            "y_reg": y_reg,
            "y_clf": y_clf,
            "dates": df.index
        }, f)

    print("Dataset loaded and saved")

if __name__ == "__main__":
    load_data()
