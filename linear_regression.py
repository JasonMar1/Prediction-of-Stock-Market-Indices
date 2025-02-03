from utils import load_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

# Date ranges for splitting the data
TRAIN_START_DATE = "2000-01-01"   # Training data start
TRAIN_END_DATE   = "2024-11-30"     # Training data end

VAL_START_DATE   = "2024-12-01"     # Validation data start
VAL_END_DATE     = "2024-12-23"     # Validation data end

TEST_START_DATE  = "2024-12-24"     # Test data start
TEST_END_DATE    = "2025-01-24"     # Test data end

df_standardized = load_data(standardized=True)

# Split the data into training, validation, and test sets based on the date
df_train = df_standardized[(df_standardized.index >= TRAIN_START_DATE) & (df_standardized.index <= TRAIN_END_DATE)]
df_val   = df_standardized[(df_standardized.index >= VAL_START_DATE)   & (df_standardized.index <= VAL_END_DATE)]
df_test  = df_standardized[(df_standardized.index >= TEST_START_DATE)  & (df_standardized.index <= TEST_END_DATE)]

# Define the features used for prediction
features = ["Open", "High", "Low", "Close", "Adjusted_close", "Volume"]

# Prepare the training set
X_train = df_train[features]
y_train = df_train["y_reg"]

# Prepare the validation set
X_val = df_val[features]
y_val = df_val["y_reg"]

# Prepare the test set
X_test = df_test[features]
y_test = df_test["y_reg"]

# Train the Linear Regression model on the training set
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the validation set
y_val_pred = model.predict(X_val)
val_mae = mean_absolute_error(y_val, y_val_pred)
val_rmse = np.sqrt(val_mae)

print("Validation Set Evaluation:")
print("--------------------------")
print(f"Number of Validation Samples: {len(y_val)}")
print(f"Mean Absolute Error (MAE): {val_mae:.4f}")
print(f"Root Mean Square Error (RMSE): {val_rmse:.4f}\n")

# (Optional) You could adjust or fine-tune your model based on the validation performance.
# For a simple linear regression, there may be limited tuning to do, but this step is crucial for more complex models.

# Once satisfied, evaluate the model on the test set
y_test_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mae)

print("Test Set Evaluation:")
print("---------------------")
print(f"Number of Test Samples: {len(y_test)}")
print(f"Mean Absolute Error (MAE): {test_mae:.4f}")
print(f"Root Mean Square Error (RMSE): {test_rmse:.4f}\n")

# (Optional) Display the first few predictions alongside the actual log returns for the test set
results = pd.DataFrame({
    "Predicted_Log_Return": y_test_pred,
    "Actual_Log_Return": y_test
}, index=df_test.index)

print("Sample Predictions:")
print(results.head(30))
