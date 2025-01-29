from utils import load_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd


# Date range for training
TRAIN_START_DATE = "2000-01-01"  # Start training from 2000
TRAIN_END_DATE = "2020-03-15"    # Train up to one day before the prediction date
PREDICTION_DATE = "2020-03-16"   # Date to predict

# Load standardized data (Linear Regression requires standardized features)
df_standardized = load_data(standardized=True)

# Load non-standardized data for fetching the actual Close price
df_non_standardized = load_data(standardized=False)

# Filter training data
df_train = df_standardized[(df_standardized.index >= TRAIN_START_DATE) & (df_standardized.index <= TRAIN_END_DATE)]

# Define features and target
features = ["Open", "High", "Low", "Close", "Adjusted_close", "Volume"]
X_train = df_train[features]
y_train = df_train["y_reg"]

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict for March 16, 2020
if PREDICTION_DATE in df_standardized.index:
    X_pred = df_standardized.loc[[PREDICTION_DATE], features]
    predicted_close = model.predict(X_pred)[0]
    print(f"Predicted Close Price for {PREDICTION_DATE}: {predicted_close:.2f}")

    # Fetch the actual Close price from the non-standardized dataset
    try:
        actual_close = df_non_standardized.loc[PREDICTION_DATE, "Close"]
        print(f"Actual Close Price for {PREDICTION_DATE}: {actual_close:.2f}")
        print(f"Prediction Error: {abs(predicted_close - actual_close):.2f}")
    except KeyError:
        print("No ground truth available for the prediction date.")

else:
    print(f"No data available for {PREDICTION_DATE}")

