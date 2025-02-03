from utils import load_data
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

# Date ranges for splitting the data
TRAIN_START_DATE = "2000-01-01"
TRAIN_END_DATE   = "2024-12-23"
TEST_START_DATE  = "2024-12-24"
TEST_END_DATE    = "2025-01-24"

# Load standardized data (features are already scaled)
df_standardized = load_data(standardized=True)

# Split into training and test sets
df_train = df_standardized[(df_standardized.index >= TRAIN_START_DATE) & (df_standardized.index <= TRAIN_END_DATE)]
df_test  = df_standardized[(df_standardized.index >= TEST_START_DATE) & (df_standardized.index <= TEST_END_DATE)]

# Define features and target
features = ["Open", "High", "Low", "Close", "Adjusted_close", "Volume"]

X_train = df_train[features]
y_train = df_train["y_reg"]

X_test = df_test[features]
y_test = df_test["y_reg"]

# Define the SVR model
svr = SVR()

# Hyperparameter tuning using Grid Search
param_grid = {
    "C": [1],  # Regularization parameter
    "epsilon": [0.01],  # Acceptable error margin
    "kernel": ["linear"]  # Types of kernel functions
}

# Use TimeSeriesSplit for time-sensitive cross-validation
tscv = TimeSeriesSplit(n_splits=5)

grid_search = GridSearchCV(svr, param_grid, cv=tscv, scoring="neg_mean_absolute_error", verbose=2)
grid_search.fit(X_train, y_train)

# Best model found by grid search
best_svr = grid_search.best_estimator_
print("\nBest parameters:", grid_search.best_params_)

# Evaluate on the test set
y_pred = best_svr.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
# rmse = mean_squared_error(y_test, y_pred, squared=False)

print("\nTest Set Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
# print(f"Root Mean Square Error (RMSE): {rmse:.4f}")

# Display sample predictions
results = pd.DataFrame({
    "Predicted_Log_Return": y_pred,
    "Actual_Log_Return": y_test
}, index=df_test.index)

print("\nSample Predictions:")
print(results.head())
