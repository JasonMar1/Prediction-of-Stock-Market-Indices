from utils import load_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Define training periods
TRAIN_START_DATE_1 = "1950-01-03"
TRAIN_START_DATE_2 = "2000-01-01"
TRAIN_END_DATE = "2020-03-13"
TEST_START_DATE = "2020-03-16"
PREDICTION_DATE = "2020-03-16"

# Load standardized data for model training
df_standardized = load_data(standardized=True)
df_non_standardized = load_data(standardized=False)  # Needed for actual Close price

features = ["Open", "High", "Low", "Close", "Adjusted_close", "Volume"]

def train_and_evaluate_model(train_start, model_name):
    # Filter training data
    df_train = df_standardized[(df_standardized.index >= train_start) & (df_standardized.index <= TRAIN_END_DATE)]
    df_test = df_standardized[df_standardized.index >= TEST_START_DATE]

    X_train = df_train[features]
    y_train = df_train["y_clf"]
    X_test = df_test[features]
    y_test = df_test["y_clf"]

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Model: {model_name}")
    print(f"Training Period: {train_start} to {TRAIN_END_DATE}")
    print(f"Test Period: {TEST_START_DATE} onward")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("-" * 50)

    if PREDICTION_DATE in df_standardized.index:
        index = df_test.index.get_loc(PREDICTION_DATE)
        predicted_label = y_pred[index]
        actual_label = y_test.iloc[index-1]

        # Fetch actual Close price from non-standardized dataset
        actual_close_price = df_non_standardized.loc[PREDICTION_DATE, "Close"]
        actual_close_price2 = df_non_standardized.loc[TRAIN_END_DATE, "Close"]

        print(f"Prediction for {PREDICTION_DATE}: {'UP' if predicted_label == 1 else 'DOWN'}")
        print(f"Actual Market Movement: {'UP' if actual_label == 1 else 'DOWN'}")
        print(f"Actual Close Price on {TRAIN_END_DATE}: {actual_close_price2:.2f}")
        print(f"Actual Close Price on {PREDICTION_DATE}: {actual_close_price:.2f}")

    else:
        print(f"No test data available for {PREDICTION_DATE}")
    return model, y_pred, y_test

model_1, y_pred_1, y_test_1 = train_and_evaluate_model(TRAIN_START_DATE_1, "Logistic Regression (1950-2020)")
model_2, y_pred_2, y_test_2 = train_and_evaluate_model(TRAIN_START_DATE_2, "Logistic Regression (2000-2020)")

# print("\nFirst 50 Predictions:", y_pred_1[:50])
# print("\nFirst 50 True Labels:", y_test_1[:50].values)