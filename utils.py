import pickle
import pandas as pd


def load_data(standardized):
    """Load full dataset (no pre-split) and return as DataFrame."""
    file_name = "processed_data_standardized.pkl" if standardized else "processed_data_non_standardized.pkl"

    with open(file_name, "rb") as f:
        data = pickle.load(f)

    # Convert to DataFrame for easy filtering
    df = pd.DataFrame(data["X"], index=data["dates"],
                      columns=["Open", "High", "Low", "Close", "Adjusted_close", "Volume"])
    df["y_reg"] = data["y_reg"]

    return df
