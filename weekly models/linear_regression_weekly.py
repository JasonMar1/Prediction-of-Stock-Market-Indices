from data_loader import load_weekly_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import pandas as pd

TRAIN_START_DATE = "2000-01-01"
TRAIN_END_DATE = "2024-01-23"

TEST_START_DATE = "2024-01-24"
TEST_END_DATE = "2025-01-24"

df_weekly = load_weekly_data(standardized=True)

# Split the data by date
df_train = df_weekly.loc[TRAIN_START_DATE:TRAIN_END_DATE]
df_test = df_weekly.loc[TEST_START_DATE:TEST_END_DATE]

features = ["Open", "High", "Low", "Close", "Adjusted_close", "Volume"]

X_train = df_train[features]
y_train = df_train["y"]

X_test = df_test[features]
y_test = df_test["y"]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print("Test Set Evaluation:")
print("---------------------")
print(f"Number of Test Samples: {len(y_test)}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Square Error (RMSE): {rmse:.4f}\n")

results = pd.DataFrame({
    "Predicted_Log_Return": y_pred,
    "Actual_Log_Return": y_test
}, index=df_test.index)

print("Sample Predictions:")
print(results.head(30))
