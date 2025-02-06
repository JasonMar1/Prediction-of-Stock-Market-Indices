from data_loader import load_monthly_data
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import pandas as pd

TRAIN_START_DATE = "2000-01-01"
TRAIN_END_DATE = "2023-12-31"

TEST_START_DATE = "2024-01-01"
TEST_END_DATE = "2025-01-24"

df_monthly = load_monthly_data(standardized=True)

# Split the data by date
df_train = df_monthly.loc[TRAIN_START_DATE:TRAIN_END_DATE]
df_test = df_monthly.loc[TEST_START_DATE:TEST_END_DATE]

features = ["Open", "High", "Low", "Close", "Adjusted_close", "Volume"]

X_train = df_train[features]
y_train = df_train["y_reg"]

X_test = df_test[features]
y_test = df_test["y_reg"]

svr = SVR()

# Hyperparameter tuning using Grid Search
param_grid = {
    "C": [0.1,1,10,100],
    "kernel": ["linear", "poly", "sigmoid","rbf"]
}

tscv = TimeSeriesSplit(n_splits=5)

grid_search = GridSearchCV(svr, param_grid, cv=tscv, scoring="neg_mean_absolute_error", verbose=1)
grid_search.fit(X_train, y_train)

# Best model found by grid search
best_svr = grid_search.best_estimator_
print("\nBest parameters:", grid_search.best_params_)

# Evaluate on the test set
y_pred = best_svr.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print("\nTest Set Evaluation:")
print("---------------------")
print(f"Number of Test Samples: {len(y_test)}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

results = pd.DataFrame({
    "Predicted_Log_Return": y_pred,
    "Actual_Log_Return": y_test
}, index=df_test.index)

print("\nSample Predictions:")
print(results.head(200))