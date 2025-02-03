from utils import load_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

# Date ranges for splitting the data
TRAIN_START_DATE = "2000-01-01"
TRAIN_END_DATE   = "2024-12-23"
TEST_START_DATE  = "2024-12-24"
TEST_END_DATE    = "2025-01-24"

# Load standardized data (features are already scaled by load_data)
df_standardized = load_data(standardized=True)

# Split the data into training and test sets based on date
df_train = df_standardized[(df_standardized.index >= TRAIN_START_DATE) & (df_standardized.index <= TRAIN_END_DATE)]
df_test  = df_standardized[(df_standardized.index >= TEST_START_DATE) & (df_standardized.index <= TEST_END_DATE)]

# Define the features and the target variable
features = ["Open", "High", "Low", "Close", "Adjusted_close", "Volume"]

X_train = df_train[features]
y_train = df_train["y_reg"]

X_test = df_test[features]
y_test = df_test["y_reg"]

# Initialize the Random Forest regressor
rf = RandomForestRegressor(random_state=42)

# Define the hyperparameter grid to search over.
# You can adjust or expand this grid as needed.
param_grid = {
    "n_estimators": [50, 100, 200],      # Number of trees in the forest
    "max_depth": [None, 10, 20, 30],       # Maximum depth of the trees
    "min_samples_split": [2, 5, 10],       # Minimum number of samples required to split an internal node
    "max_features": ["auto", "sqrt"]       # Number of features to consider when looking for the best split
}

# Use TimeSeriesSplit for time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Set up GridSearchCV with the negative mean absolute error as scoring.
# (We use the negative metric because GridSearchCV maximizes the score.)
grid_search = GridSearchCV(rf, param_grid, cv=tscv, scoring="neg_mean_absolute_error", verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best parameters found
print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation score (negative MAE):", grid_search.best_score_)

# Retrieve the best model from grid search
best_rf = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("\nTest Set Evaluation:")
print("---------------------")
print(f"Number of Test Samples: {len(y_test)}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Display sample predictions alongside the actual values
results = pd.DataFrame({
    "Predicted_Log_Return": y_pred,
    "Actual_Log_Return": y_test
}, index=df_test.index)

print("\nSample Predictions:")
print(results.head())
