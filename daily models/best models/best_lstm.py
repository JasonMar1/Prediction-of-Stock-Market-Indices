import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader
from data_loader import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_START_DATE = "2000-01-01"
TRAIN_END_DATE = "2024-01-23"

TEST_START_DATE = "2024-01-24"
TEST_END_DATE = "2025-01-24"

df_standardized = load_data(standardized=True)

# Split the data by date
df_train = df_standardized.loc[TRAIN_START_DATE:TRAIN_END_DATE]
df_test = df_standardized.loc[TEST_START_DATE:TEST_END_DATE]

features = ["Open", "High", "Low", "Close", "Volume"]

X_train = df_train[features].values  # shape: (n_train_samples, num_features)
y_train = df_train["y"].values  # shape: (n_train_samples,)

X_test = df_test[features].values
y_test = df_test["y"].values


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Use output from the last time step. (flattened)
        out = self.fc(out)
        return out


def create_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:i + seq_length])
        ys.append(y[i + seq_length])
    return np.array(xs), np.array(ys)

best_seq_length = 15
batch_size = None  # Set your own value

X_train_seq, y_train_seq = create_sequences(X_train, y_train, best_seq_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, best_seq_length)

X_train_seq = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
y_train_seq = torch.tensor(y_train_seq, dtype=torch.float32).to(device)
X_test_seq = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
y_test_seq = torch.tensor(y_test_seq, dtype=torch.float32).to(device)

input_size = len(features)
hidden_size = 32
num_layers = None  # Set your own value
dropout = None  # Set your own value
epochs = 200
learning_rate = 0.0005

model = LSTMRegressor(input_size=len(features),
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      output_size=1,
                      dropout=dropout).to(device)

criterion = nn.L1Loss()  # MAE loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create DataLoader for training
train_dataset = TensorDataset(X_train_seq, y_train_seq)
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

torch.save(model, "best_lstm_model.pth")


def permutation_importance(model, X_test, y_test, batch_size, feature_names, device):
    model.eval()
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32).to(device),
                                 torch.tensor(y_test, dtype=torch.float32).to(device))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    baseline_predictions = []
    actual_values = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            baseline_predictions.append(outputs.squeeze().cpu().numpy())
            actual_values.append(batch_y.cpu().numpy())

    baseline_predictions = np.concatenate(baseline_predictions)
    actual_values = np.concatenate(actual_values)
    baseline_mae = mean_absolute_error(actual_values, baseline_predictions)
    print(f"Baseline MAE: {baseline_mae:.6f}")

    feature_importance = {}

    for feature_idx, feature in enumerate(feature_names):
        X_test_permuted = X_test.copy()
        np.random.shuffle(X_test_permuted[:, :, feature_idx])

        test_dataset_permuted = TensorDataset(torch.tensor(X_test_permuted, dtype=torch.float32).to(device),
                                              torch.tensor(y_test, dtype=torch.float32).to(device))
        test_loader_permuted = DataLoader(test_dataset_permuted, batch_size=batch_size, shuffle=False)

        permuted_predictions = []
        with torch.no_grad():
            for batch_x, _ in test_loader_permuted:
                outputs = model(batch_x)
                permuted_predictions.append(outputs.squeeze().cpu().numpy())

        permuted_predictions = np.concatenate(permuted_predictions)
        permuted_mae = mean_absolute_error(actual_values, permuted_predictions)

        importance_score = permuted_mae - baseline_mae
        feature_importance[feature] = importance_score

        print(f"Feature {feature} Importance: {importance_score:.6f}")

    return feature_importance


best_model = torch.load("best_lstm_model.pth")

feature_importances = permutation_importance(best_model, X_test_seq.cpu().numpy(),
                                             y_test_seq.cpu().numpy(), batch_size, features, device)

sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

plt.figure(figsize=(10, 5))
plt.barh([x[0] for x in sorted_features], [x[1] for x in sorted_features], color="skyblue")
plt.xlabel("Feature Importance (Increase in MAE)")
plt.ylabel("Feature")
plt.title("LSTM Feature Importance (Permutation)")
plt.gca().invert_yaxis()
plt.show()
