from data_loader import load_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import numpy as np
import pandas as pd

TRAIN_START_DATE = "2000-01-01"
TRAIN_END_DATE   = "2024-11-30"

VAL_START_DATE   = "2024-12-01"
VAL_END_DATE     = "2024-12-23"

TEST_START_DATE  = "2024-12-24"
TEST_END_DATE    = "2025-01-24"

df_standardized = load_data(standardized=True)

# Split the data by date
df_train = df_standardized.loc[TRAIN_START_DATE:TRAIN_END_DATE]
df_val = df_standardized.loc[VAL_START_DATE:VAL_END_DATE]
df_test = df_standardized.loc[TEST_START_DATE:TEST_END_DATE]

features = ["Open", "High", "Low", "Close", "Adjusted_close", "Volume"]

X_train = df_train[features]
y_train = df_train["y_reg"]

X_val = df_val[features]
y_val = df_val["y_reg"]

X_test = df_test[features]
y_test = df_test["y_reg"]

model = LinearRegression()
model.fit(X_train, y_train)

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
test_rmse = root_mean_squared_error(y_test, y_test_pred)

print("Test Set Evaluation:")
print("---------------------")
print(f"Number of Test Samples: {len(y_test)}")
print(f"Mean Absolute Error (MAE): {test_mae:.4f}")
print(f"Root Mean Square Error (RMSE): {test_rmse:.4f}\n")

results = pd.DataFrame({
    "Predicted_Log_Return": y_test_pred,
    "Actual_Log_Return": y_test
}, index=df_test.index)

print("Sample Predictions:")
print(results.head(30))
