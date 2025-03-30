from data_loader import load_daily_data_log_returns
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import pandas as pd


best_score = float("inf")
best_params = None

TRAIN_START_DATE = "2006-01-01"
TRAIN_END_DATE = "2019-11-30"

VALID_START_DATE = "2019-12-01"
VALID_END_DATE = "2022-08-31"

TEST_START_DATE = "2022-10-01"  # worst case scenario, having sequence length equal to 3 months + dropping 1 month for data-leakage
TEST_END_DATE = "2025-01-01"

X_train, y_train, X_valid, y_valid, X_test, y_test, df_test, features = load_daily_data_log_returns(True, TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE)

for C in [0.1, 1, 10, 100]:
    for kernel in ["linear", "poly", "sigmoid", "rbf"]:
        model = SVR(C=C, kernel=kernel)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score = mean_absolute_error(y_test, y_pred)

        if score < best_score:
            best_score = score
            best_params = {"C": C, "kernel": kernel}

print("Best parameters:", best_params)
best_svr = SVR(C=best_params["C"], kernel=best_params["kernel"])
best_svr.fit(X_train, y_train)

# Evaluate on the test set
y_pred = best_svr.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print("\nTest Set Evaluation:")
print("---------------------")
print(f"Number of Test Samples: {len(y_test)}")
print(f"Mean Absolute Error (MAE: {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

results = pd.DataFrame({
    "Predicted_Log_Return": y_pred,
    "Actual_Log_Return": y_test
}, index=df_test.index)

print("\nSample Predictions:")
print(results.head(5))
