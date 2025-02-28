import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from data_loader import load_daily_data_log_returns
import matplotlib.pyplot as plt


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def reset_states(self, batch_size, device):
        # Hidden state and cell state to zeros.
        self.h_state = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        self.c_state = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

    def forward(self, x):
        self.reset_states(x.size(0), x.device)

        out, (self.h_state, self.c_state) = self.lstm(x, (self.h_state, self.c_state))
        out = out[:, -1, :]  # Use the output from the last time step.
        out = self.fc(out)
        return out


def create_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:i + seq_length])
        ys.append(y.iloc[i + seq_length])
    return np.array(xs), np.array(ys)


def get_dataloaders(X_train, y_train, seq_length, batch_size, device):
    train_seq, train_targets = create_sequences(X_train, y_train, seq_length)

    # Convert the sequences, targets to torch tensors & create the Dataloaders
    train_dataset = TensorDataset(torch.tensor(train_seq, dtype=torch.float32).to(device), torch.tensor(train_targets, dtype=torch.float32).to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader


def compute_permutation_importance(model, X_valid_seq, y_valid, feature_names, device, criterion):
    model.eval()

    X_valid_tensor = torch.tensor(X_valid_seq, dtype=torch.float32).to(device)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(X_valid_tensor)
    baseline_mae = criterion(output.view(-1), y_valid_tensor.view(-1)).item()

    importance_scores = {}
    X_valid_permuted = X_valid_seq.copy()  # Copy to modify for each feature

    for i, feature in enumerate(feature_names):
        original = X_valid_permuted[:, :, i].copy()
        np.random.shuffle(X_valid_permuted[:, :, i])  # Shuffle the i-th feature's values across all sequences and time steps

        X_perm_tensor = torch.tensor(X_valid_permuted, dtype=torch.float32).to(device)

        with torch.no_grad():
            permuted_output = model(X_perm_tensor)
        permuted_mae = criterion(permuted_output.view(-1), y_valid_tensor.view(-1)).item()

        importance = permuted_mae - baseline_mae
        importance_scores[feature] = importance

        X_valid_permuted[:, :, i] = original  # Restore original values for the next feature

    return importance_scores


def plot_feature_importance(importance):
    sorted_features = sorted(importance, key=importance.get, reverse=True)  # Descending sort the features by importance
    sorted_importance = [importance[f] for f in sorted_features]

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_features, sorted_importance, color='skyblue')
    plt.xlabel("Features")
    plt.ylabel("Increase in MAE (Importance)")
    plt.title("Permutation Feature Importance on Validation Set")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_START_DATE = "2000-01-01"
TRAIN_END_DATE = "2019-12-31"

VALID_START_DATE = "2020-01-01"
VALID_END_DATE = "2023-01-23"

TEST_START_DATE = "2023-01-24"
TEST_END_DATE = "2025-01-24"

X_train, y_train, X_valid, y_valid, X_test, y_test, df_test, features = load_daily_data_log_returns(True, TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE)


#OPTUNA
hidden_size = 53
num_layers = 3
dropout = 0.1
learning_rate = 0.0001476
batch_size = None  # Set your own value
epochs = 100
sequence_length = 50


print('-' * 100)

model = LSTM(input_size=len(features), hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout=dropout).to(device)
criterion = nn.L1Loss()  # MAE loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loader = get_dataloaders(X_train, y_train, sequence_length, batch_size, device)

for epoch in range(epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs.view(-1), batch_y.view(-1))
        loss.backward()
        optimizer.step()

# --- Compute Permutation Feature Importance on Validation Set ---
valid_seq, valid_targets = create_sequences(X_valid, y_valid, sequence_length)
importance = compute_permutation_importance(model,valid_seq, valid_targets, features, device, criterion)

plot_feature_importance(importance)
print("\nPermutation Feature Importance on Validation Set:")
for feat, score in importance.items():
    print(f"{feat}: {score:.6f}")

