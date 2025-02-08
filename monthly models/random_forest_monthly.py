from data_loader import load_monthly_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import pandas as pd


TRAIN_START_DATE = "2000-01-01"
TRAIN_END_DATE = "2023-12-31"

TEST_START_DATE = "2024-01-01"
TEST_END_DATE = "2025-01-24"

df_monthly = load_monthly_data(standardized=True)

df_train = df_monthly.loc[TRAIN_START_DATE:TRAIN_END_DATE]
df_test = df_monthly.loc[TEST_START_DATE:TEST_END_DATE]

features = ["Open", "High", "Low", "Close", "Adjusted_close", "Volume"]

X_train = df_train[features]
y_train = df_train["y_reg"]

X_test = df_test[features]
y_test = df_test["y_reg"]

# Initialize the Random Forest regressor
rf = RandomForestRegressor(random_state=42)

# Define the hyperparameter grid to search over.
param_grid = {
    "n_estimators": [20, 50, 100, 200, 500],      # Number of trees in the forest
    "max_depth": [None, 10, 20, 30],       # Maximum depth of the trees
    "min_samples_split": [2, 5, 10],   # Minimum number of samples required to split an internal node
    "max_features": ["sqrt", "log2", None]       # Number of features to consider when looking for the best split (better results using it)
}

# Use TimeSeriesSplit for time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

grid_search = GridSearchCV(rf, param_grid, cv=tscv, scoring="neg_mean_absolute_error", verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best parameters found
print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation score (negative MAE):", grid_search.best_score_)

# Retrieve the best model from grid search
best_rf = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_rf.predict(X_test)
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