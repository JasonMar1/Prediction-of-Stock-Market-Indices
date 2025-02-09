from data_loader import load_daily_data_log_returns
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import pandas as pd

TRAIN_START_DATE = "2000-01-05"
TRAIN_END_DATE = "2023-12-23"

TEST_START_DATE = "2023-12-24"
TEST_END_DATE = "2025-01-24"

df, features = load_daily_data_log_returns(True, TRAIN_START_DATE, TEST_END_DATE)

# Split the data by date
df_train = df.loc[TRAIN_START_DATE:TRAIN_END_DATE]
df_test = df.loc[TEST_START_DATE:TEST_END_DATE]

X_train = df_train[features]
y_train = df_train["y"]

X_test = df_test[features]
y_test = df_test["y"]

svr = SVR()
hyperparameters = {
    "C": [0.1],
    "kernel": ["linear"]
}

tscv = TimeSeriesSplit(n_splits=5)  # TimeSeries CrossValidation 5-fold

grid_search = GridSearchCV(svr, hyperparameters, cv=tscv, scoring="neg_mean_absolute_error", verbose=1)
grid_search.fit(X_train, y_train)

# Best model found by grid search
best_svr = grid_search.best_estimator_
print("\nBest parameters:", grid_search.best_params_)


# best_score = float("inf")
# best_params = None
#
# for C in [0.1, 1, 10, 100]:
#     for kernel in ["linear", "poly", "sigmoid", "rbf"]:
#         model = SVR(C=C, kernel=kernel)
#         model.fit(X_train, y_train)
#
#         # Evaluate on test set
#         y_pred = model.predict(X_test)
#         score = mean_absolute_error(y_test, y_pred)
#
#         if score < best_score:
#             best_score = score
#             best_params = {"C": C, "kernel": kernel}
#
# print("Best parameters:", best_params)
# best_svr = SVR(C=best_params["C"], kernel=best_params["kernel"])
# best_svr.fit(X_train, y_train)

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
