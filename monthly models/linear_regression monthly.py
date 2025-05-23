from data_loader import load_monthly_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import pandas as pd

TRAIN_START_DATE = "2006-01-01"
TRAIN_END_DATE = "2019-11-30"

VALID_START_DATE = "2019-12-01"
VALID_END_DATE = "2022-08-31"

TEST_START_DATE = "2022-10-01"  # worst case scenario, having sequence length equal to 3 months + dropping 1 month for data-leakage
TEST_END_DATE = "2025-01-01"

X_train, y_train, X_valid, y_valid, X_test, y_test, df_test, features, index_name = load_monthly_data(True, TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE)


model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate on the test set
y_pred = model.predict(X_test)

# mae, rmse only for the specific date range
fixed_start = "2023-01-01"
fixed_end = "2025-01-01"

results = pd.DataFrame({
    "Predicted_Log_Return": y_pred,
    "Actual_Log_Return": y_test,
    "Index": index_name
}, index=df_test.index)

results["Adjusted_Close"] = results.apply(lambda row: df_test.loc[(df_test.index == row.name) & (df_test["Index"] == row["Index"]), "Adjusted_close"].values[0], axis=1)
results_filtered = results.loc[(results.index >= fixed_start) & (results.index <= fixed_end)]

print(f'results_filtered: {results_filtered}')

mae_loss = mean_absolute_error(results_filtered["Actual_Log_Return"], results_filtered["Predicted_Log_Return"])
rmse_loss = root_mean_squared_error(results_filtered["Actual_Log_Return"], results_filtered["Predicted_Log_Return"])

print(f"MAE: {mae_loss:.6f}")
print(f"RMSE: {rmse_loss:.6f}")

print("\nSample Predictions:")
results_filtered.to_csv(f"BASIC ML MONTHLY PREDICTIONS BEST/predictions_linear_regression_{index_name}.csv")
print(results.head(5))
