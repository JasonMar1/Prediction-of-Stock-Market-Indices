from data_loader import load_daily_data_log_returns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import pandas as pd

TRAIN_START_DATE = "2006-01-01"
TRAIN_END_DATE = "2019-12-31"

VALID_START_DATE = "2020-01-01"
VALID_END_DATE = "2023-01-23"

TEST_START_DATE = "2023-01-24"
TEST_END_DATE = "2025-01-24"

X_train, y_train, X_valid, y_valid, X_test, y_test, df_test, features = load_daily_data_log_returns(True, TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE)


model = LinearRegression()
model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = root_mean_squared_error(y_test, y_test_pred)

print("Test Set Evaluation:")
print("---------------------")
print(f"Number of Test Samples: {len(y_test)}")
print(f"Mean Absolute Error (MAE): {test_mae:.4f}")
print(f"Root Mean Square Error (RMSE): {test_rmse:.4f}\n")

results = pd.DataFrame({
    "Predicted_Log_Return": y_test_pred,
    "Actual_Log_Return": y_test
}, index=df_test.index)

print("Sample Predictions:")
print(results.head(30))
