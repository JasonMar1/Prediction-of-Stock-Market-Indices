from utils import load_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np

# Define training periods
TRAIN_START_DATE_1 = "1950-01-03"  # Oldest available data
TRAIN_START_DATE_2 = "2000-01-01"  # More recent data
TRAIN_END_DATE = "2020-03-14"      # Same end date for both models
TEST_START_DATE = "2020-03-15"     # Test on the same period

# Load standardized data for training
df_standardized = load_data(standardized=True)

# Load non-standardized data for actual Close price comparison
df_non_standardized = load_data(standardized=False)

features = ["Open", "High", "Low", "Close", "Adjusted_close", "Volume"]

# Function to train and evaluate a model
def train_and_evaluate_model(train_start, model_name):
    # Filter training data
    df_train = df_standardized[(df_standardized.index >= train_start) & (df_standardized.index <= TRAIN_END_DATE)]
    df_test = df_standardized[df_standardized.index >= TEST_START_DATE]

    X_train = df_train[features]
    y_train = df_train["y_reg"]
    X_test = df_test[features]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Get actual Close prices (available from the non-standardized dataset)
    y_test = df_non_standardized.loc[X_test.index, "Close"]

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Model: {model_name}")
    print(f"Training Period: {train_start} to {TRAIN_END_DATE}")
    print(f"Test Period: {TEST_START_DATE} onward")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
    print("-" * 50)

    return model, y_pred, y_test

# Train and evaluate both models
model_1, y_pred_1, y_test_1 = train_and_evaluate_model(TRAIN_START_DATE_1, "Long-Term Training (1950-2020)")
model_2, y_pred_2, y_test_2 = train_and_evaluate_model(TRAIN_START_DATE_2, "Short-Term Training (2000-2020)")

# Compare predictions (show first 10)
comparison_df = pd.DataFrame({
    "Actual Close": y_test_1,
    "Predicted (1950-2020)": y_pred_1,
    "Predicted (2000-2020)": y_pred_2
})
print(comparison_df.head(10))
