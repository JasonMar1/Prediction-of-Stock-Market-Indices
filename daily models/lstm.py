import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
from data_loader import load_data
from itertools import product

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_START_DATE = "2000-01-01"
TRAIN_END_DATE = "2024-01-23"

TEST_START_DATE = "2024-01-24"
TEST_END_DATE = "2025-01-24"

df_standardized = load_data(standardized=True)

# Split the data by date
df_train = df_standardized.loc[TRAIN_START_DATE:TRAIN_END_DATE]
df_test = df_standardized.loc[TEST_START_DATE:TEST_END_DATE]

features = ["Open", "High", "Low", "Close", "Adjusted_close", "Volume"]

X_train = df_train[features].values  # shape: (n_train_samples, num_features)
y_train = df_train["y"].values  # shape: (n_train_samples,)

X_test = df_test[features].values
y_test = df_test["y"].values


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Use output from the last time step. (flattened)
        out = self.fc(out)
        return out


grid_params = {
    "hidden_size": [32, 64, 128],
    "num_layers": [1, 2],
    "dropout": [0.1, 0.2],
    "learning_rate": [0.01, 0.001, 0.0005],
    "batch_size": [32, 64],
    "epochs": [50, 100, 200],
    "sequence_length": [5, 10, 15]
}

param_combinations = list(product(*grid_params.values()))
print(f"Total combinations to try: {len(param_combinations)}")

best_model = None
best_params = None
best_mae = float("inf")

# Note: For each hyperparameter combination, we regenerate sequences using the given sequence_length.
for params in param_combinations:
    hidden_size, num_layers, dropout, learning_rate, batch_size, epochs, seq_length = params
    print(f"\nTraining LSTM with params: {params}")


    # Create sequences using current seq_length.
    def create_sequences(X, y, seq_length):
        xs, ys = [], []
        for i in range(len(X) - seq_length):
            xs.append(X[i:i + seq_length])
            ys.append(y[i + seq_length])
        return np.array(xs), np.array(ys)


    curr_train_seq, curr_train_targets = create_sequences(X_train, y_train, seq_length)
    curr_test_seq, curr_test_targets = create_sequences(X_test, y_test, seq_length)

    # Convert sequences to torch tensors.
    curr_train_seq = torch.tensor(curr_train_seq, dtype=torch.float32).to(device)
    curr_train_targets = torch.tensor(curr_train_targets, dtype=torch.float32).to(device)
    curr_test_seq = torch.tensor(curr_test_seq, dtype=torch.float32).to(device)
    curr_test_targets = torch.tensor(curr_test_targets, dtype=torch.float32).to(device)

    model = LSTMRegressor(input_size=len(features),
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          output_size=1,
                          dropout=dropout).to(device)

    criterion = nn.L1Loss()  # MAE loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create DataLoader for training
    train_dataset = TensorDataset(curr_train_seq, curr_train_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    # Evaluate on current test set
    model.eval()
    predictions = []
    actuals = []
    test_dataset = TensorDataset(curr_test_seq, curr_test_targets)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            predictions.append(outputs.squeeze().cpu().numpy())
            actuals.append(batch_y.cpu().numpy())

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    current_mae = mean_absolute_error(actuals, predictions)
    print(f"MAE for params {params}: {current_mae:.6f}")

    if current_mae < best_mae:
        best_mae = current_mae
        best_model = model
        best_params = params

print(f"\nBest Model Parameters: {best_params}")
print(f"Best MAE: {best_mae:.6f}")

# Use the best sequence length from the best parameters
best_seq_length = best_params[-1]
# Regenerate sequences for final evaluation.
final_train_seq, final_train_targets = create_sequences(X_train, y_train, best_seq_length)
final_test_seq, final_test_targets = create_sequences(X_test, y_test, best_seq_length)

# Convert final test sequences.
final_test_seq = torch.tensor(final_test_seq, dtype=torch.float32).to(device)
final_test_targets = torch.tensor(final_test_targets, dtype=torch.float32).to(device)

# Evaluate on final test set.
best_model.eval()
predictions = []
actuals = []
final_test_dataset = TensorDataset(final_test_seq, final_test_targets)
final_test_loader = DataLoader(final_test_dataset, batch_size=best_params[4],
                               shuffle=False)  # best_params[4] is best batch_size

with torch.no_grad():
    for batch_x, batch_y in final_test_loader:
        outputs = best_model(batch_x)
        predictions.append(outputs.squeeze().cpu().numpy())
        actuals.append(batch_y.cpu().numpy())

predictions = np.concatenate(predictions)
actuals = np.concatenate(actuals)
best_rmse = root_mean_squared_error(actuals, predictions)

print(f"\nFinal Best Model Evaluation:")
print(f"Test MAE: {best_mae:.6f}")
print(f"Test RMSE: {best_rmse:.6f}")

dates = df_test.index[best_seq_length:]
results = pd.DataFrame({
    "Predicted_Log_Return": predictions,
    "Actual_Log_Return": actuals
}, index=dates)

print("\nSample Predictions:")
print(results.head(30))
