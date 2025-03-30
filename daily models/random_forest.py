from data_loader import load_daily_data_log_returns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import pandas as pd

import itertools


TRAIN_START_DATE = "2006-01-01"
TRAIN_END_DATE = "2019-11-30"

VALID_START_DATE = "2019-12-01"
VALID_END_DATE = "2022-08-31"

TEST_START_DATE = "2022-10-01"  # worst case scenario, having sequence length equal to 3 months + dropping 1 month for data-leakage
TEST_END_DATE = "2025-01-01"

X_train, y_train, X_valid, y_valid, X_test, y_test, df_test, features = load_daily_data_log_returns(True, TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE)

# # Initialize the Random Forest regressor
# rf = RandomForestRegressor(random_state=42)

# Define the hyperparameter grid to search over.
param_grid = {
    "n_estimators": [20, 50, 100, 200, 500],      # Number of trees in the forest
    "max_depth": [None, 10, 20, 30],       # Maximum depth of the trees
    "min_samples_split": [2, 5, 10],      # Minimum number of samples required to split an internal node
    "max_features": ["sqrt", "log2", None]  # Number of features to consider when looking for the best split
}

# # Use TimeSeriesSplit for time series cross-validation
# tscv = TimeSeriesSplit(n_splits=5)
#
# # Set up GridSearchCV with the negative mean absolute error as scoring.
# # (negative metric because GridSearchCV maximizes the score.)
# grid_search = GridSearchCV(rf, param_grid, cv=tscv, scoring="neg_mean_absolute_error", verbose=1, n_jobs=-1)
# grid_search.fit(X_train, y_train)
#
# # Print the best parameters found
# print("Best parameters found:", grid_search.best_params_)
# print("Best cross-validation score (negative MAE):", grid_search.best_score_)
#
# # Retrieve the best model from grid search
# best_rf = grid_search.best_estimator_


param_combinations = list(itertools.product(
    param_grid["n_estimators"],
    param_grid["max_depth"],
    param_grid["min_samples_split"],
    param_grid["max_features"]
))

best_mae = float('inf')
best_params = None
best_rf = None


for params in param_combinations:
    n_estimators, max_depth, min_samples_split, max_features = params

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    val_pred = rf.predict(X_valid)
    val_mae = mean_absolute_error(y_valid, val_pred)

    if val_mae < best_mae:
        best_mae = val_mae
        best_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "max_features": max_features
        }
        best_rf = rf


# Evaluate the best model on the test set
y_pred = best_rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print("Best Parameters:", best_params)
print(f"Validation MAE: {best_mae:.4f}")


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
