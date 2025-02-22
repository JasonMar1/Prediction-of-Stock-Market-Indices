import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
from data_loader import load_daily_data_log_returns
from itertools import product
import matplotlib.pyplot as plt


torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_START_DATE = "2000-01-01"
TRAIN_END_DATE = "2019-12-31"

VALID_START_DATE = "2020-01-01"
VALID_END_DATE = "2023-01-23"

TEST_START_DATE = "2023-01-24"
TEST_END_DATE = "2025-01-24"

X_train, y_train, X_valid, y_valid, X_test, y_test, df_test, features = load_daily_data_log_returns(True, TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Use the output from the last time step
        out = self.fc(out)
        return out


grid_params = {
    "hidden_size": [64],
    "num_layers": [2],
    "dropout": [0.0],
    "learning_rate": [0.0005],
    "batch_size": [64],
    "epochs": [100],
    "sequence_length": [20]
}


# grid_params = {
#     "hidden_size": [32, 64, 128],
#     "num_layers": [1, 2],
#     "dropout": [0.0, 0.2, 0.5],
#     "learning_rate": [0.01, 0.001, 0.0005],
#     "batch_size": [16, 32, 64],
#     "epochs": [100, 200],
#     "sequence_length": [10, 15, 20]
# }

param_combinations = list(product(*grid_params.values()))
print(f"Total combinations: {len(param_combinations)}")

best_model = None
best_params = None
best_mae = float("inf")


def create_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:i + seq_length])
        ys.append(y.iloc[i + seq_length])
    return np.array(xs), np.array(ys)


def get_dataloaders(X_train, y_train, X_valid, y_valid, X_test, y_test, seq_length, batch_size, device):
    train_seq, train_targets = create_sequences(X_train, y_train, seq_length)
    valid_seq, valid_targets = create_sequences(X_valid, y_valid, seq_length)
    test_seq, test_targets = create_sequences(X_test, y_test, seq_length)

    # Convert sequences to torch tensors & create the Dataloaders
    train_dataset = TensorDataset(torch.tensor(train_seq, dtype=torch.float32).to(device), torch.tensor(train_targets, dtype=torch.float32).to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)  # Επηρεάζεται το αποτέλεσμα αν κάνω Shuffle?
                                                                                                    # Αν κάνω drop_last τότε χαλάει αρκετά η συνοχή των δεδομένων μου?

    valid_dataset = TensorDataset(torch.tensor(valid_seq, dtype=torch.float32).to(device), torch.tensor(valid_targets, dtype=torch.float32).to(device))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    test_dataset = TensorDataset(torch.tensor(test_seq, dtype=torch.float32).to(device), torch.tensor(test_targets, dtype=torch.float32).to(device))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, valid_loader, test_loader


def plot_losses(epochs, train_losses, valid_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss", marker='o', linestyle='-')
    plt.plot(range(1, epochs + 1), valid_losses, label="Validation Loss", marker='s', linestyle='--', color='red')
    plt.title("Training & Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss (MAE)")
    plt.legend()
    plt.grid(True)
    plt.show()

# For each hyperparameter combination, regenerate sequences using the given sequence_length.
for params in param_combinations:
    hidden_size, num_layers, dropout, learning_rate, batch_size, epochs, seq_length = params
    print('-' * 100)
    print(f"\nTraining LSTM with params: {params}")

    model = LSTM(input_size=len(features), hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout=dropout).to(device)
    criterion = nn.L1Loss()  # MAE loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, valid_loader, test_loader = get_dataloaders(X_train, y_train, X_valid, y_valid, X_test, y_test, seq_length, batch_size, device)
    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)

            loss = criterion(outputs.view(-1), batch_y.view(-1))
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        avg_train_loss = np.mean(batch_losses)
        train_losses.append(avg_train_loss)

        # Compute validation loss
        model.eval()
        valid_loss = []
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                outputs = model(batch_x)
                loss = criterion(outputs.view(-1), batch_y.view(-1))
                valid_loss.append(loss.item())

        avg_valid_loss = np.mean(valid_loss)
        valid_losses.append(avg_valid_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Valid Loss: {avg_valid_loss:.6f}")

    plot_losses(epochs, train_losses, valid_losses)

    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            predictions.append(outputs.squeeze().cpu().numpy())
            actuals.append(batch_y.cpu().numpy())

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    current_mae = mean_absolute_error(actuals, predictions)
    print(f"MAE for params {params}: {current_mae:.6f}")

    current_rmse = root_mean_squared_error(actuals, predictions)
    print(f"RMSE for params {params}: {current_rmse:.6f}")

    if current_mae < best_mae:
        best_mae = current_mae
        best_model = model
        best_params = params


print('-' * 100)
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

print(f"\nBest Model Evaluation:")
print(f"Test MAE: {best_mae:.6f}")
print(f"Test RMSE: {best_rmse:.6f}")

dates = df_test.index[best_seq_length:]
results = pd.DataFrame({
    "Predicted_Log_Return": predictions,
    "Actual_Log_Return": actuals
}, index=dates)

print("\nSample Predictions:")
print(results.head(10))
