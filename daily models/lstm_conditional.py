import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
from data_loader_conditional import conditional_lstm_load_multiple_indices, combine_and_sort_data
import matplotlib.pyplot as plt


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, num_indices, embedding_dim):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

        self.h_state = nn.Embedding(num_embeddings=num_indices, embedding_dim=embedding_dim)
        self.c_state = nn.Embedding(num_embeddings=num_indices, embedding_dim=embedding_dim)

        # map embedding dimension to LSTM hidden_size
        self.h_state_proj = nn.Linear(embedding_dim, hidden_size)
        self.c_state_proj = nn.Linear(embedding_dim, hidden_size)

    def forward(self, x, index):
        # Retrieve the initial state vectors for the batch & match the LSTM requirements
        h_embedding = self.h_state(index)  # (batch_size, embedding_dim)
        c_embedding = self.c_state(index)

        # map embedding dimension to LSTM hidden_size
        h_proj = self.h_state_proj(h_embedding)  # (batch_size, hidden_size)
        c_proj = self.c_state_proj(c_embedding)  # (batch_size, hidden_size)

        # change to correct size (num_layers, batch_size, hidden_size)
        h_state = h_proj.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_state = c_proj.unsqueeze(0).repeat(self.num_layers, 1, 1)

        # print(f'h_state = {h_state}')
        # print(f'h_state: {h_state.size()}')
        # print(f'c_state: {c_state.size()}')

        out, _ = self.lstm(x, (h_state, c_state))
        out = out[:, -1, :]  # Use the output from the last time step.
        out = self.fc(out)
        return out


def create_sequences(X, y, index_labels, seq_length):
    xs, ys, indices = [], [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:i + seq_length])
        ys.append(y.iloc[i + seq_length])
        indices.append(index_labels.iloc[i + seq_length])  # Store the corresponding index label
    return np.array(xs), np.array(ys), np.array(indices)


def get_dataloaders(X_train, y_train, index_train, X_valid, y_valid, index_valid, X_test, y_test, index_test, seq_length, batch_size, device):
    train_seq, train_targets, train_indices = create_sequences(X_train, y_train, index_train, seq_length)
    valid_seq, valid_targets, valid_indices = create_sequences(X_valid, y_valid, index_valid, seq_length)
    test_seq, test_targets, test_indices = create_sequences(X_test, y_test, index_test, seq_length)

    # Convert the sequences, targets, indices to torch tensors & create the Dataloaders
    train_dataset = TensorDataset(torch.tensor(train_seq, dtype=torch.float32).to(device), torch.tensor(train_targets, dtype=torch.float32).to(device), torch.tensor(train_indices, dtype=torch.long).to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = TensorDataset(torch.tensor(valid_seq, dtype=torch.float32).to(device), torch.tensor(valid_targets, dtype=torch.float32).to(device), torch.tensor(valid_indices, dtype=torch.long).to(device))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(torch.tensor(test_seq, dtype=torch.float32).to(device), torch.tensor(test_targets, dtype=torch.float32).to(device), torch.tensor(test_indices, dtype=torch.long).to(device))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_START_DATE = "2006-01-01"
TRAIN_END_DATE = "2019-11-30"

VALID_START_DATE = "2019-12-01"
VALID_END_DATE = "2022-08-31"

TEST_START_DATE = "2022-10-01"  # worst case scenario, having sequence length equal to 3 months + dropping 1 month for data-leakage
TEST_END_DATE = "2025-01-01"

combined_X_train, combined_y_train, index_train, combined_X_valid, combined_y_valid, index_valid, combined_X_test, combined_y_test, index_test, df_test, features = conditional_lstm_load_multiple_indices('daily', True, TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE)

# Align and Sort Data
X_train, y_train, index_train, X_valid, y_valid, index_valid, X_test, y_test, index_test = combine_and_sort_data( combined_X_train, combined_y_train, index_train, combined_X_valid, combined_y_valid, index_valid, combined_X_test, combined_y_test, index_test)

# OPTUNA
hidden_size = 35
num_layers = 2
dropout = 0.30000000000000004
learning_rate = 0.006812332097152033
batch_size = 112
epochs = 100
sequence_length = 55

print('-' * 100)

model = LSTM(input_size=len(features), hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout=dropout, num_indices=4, embedding_dim=None  # Set your own value).to(device)
criterion = nn.L1Loss()  # MAE loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loader, valid_loader, test_loader = get_dataloaders(X_train, y_train, index_train, X_valid, y_valid, index_valid, X_test, y_test, index_test, sequence_length, batch_size, device)
train_losses = []
valid_losses = []

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=learning_rate * 10,
    steps_per_epoch=len(train_loader),
    epochs=epochs,
    pct_start=0.3,
    anneal_strategy='cos',
    div_factor=10,
    final_div_factor=100,
)

for epoch in range(epochs):
    model.train()
    train_loss = []
    for batch_x, batch_y, batch_index in train_loader:
        optimizer.zero_grad()

        outputs = model(batch_x, batch_index)  # batch_x.shape = (batch_size, seq_length, num_features)
        loss = criterion(outputs.view(-1), batch_y.view(-1))

        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss.append(loss.item())

    avg_train_loss = np.mean(train_loss)
    train_losses.append(avg_train_loss)

    model.eval()
    valid_loss = []
    with torch.no_grad():
        for batch_x, batch_y, batch_index in valid_loader:
            outputs = model(batch_x, batch_index)
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
index_labels = []

with torch.no_grad():
    for batch_x, batch_y, batch_index in test_loader:
        outputs = model(batch_x, batch_index)
        predictions.append(outputs.squeeze().cpu().numpy())
        actuals.append(batch_y.squeeze().cpu().numpy())
        index_labels.extend(batch_index.cpu().numpy())  # Store which index the prediction belongs to

predictions = np.concatenate(predictions)
actuals = np.concatenate(actuals)
index_labels = np.array(index_labels)
print(index_labels)
print('-' * 100)

# mae_loss = mean_absolute_error(actuals, predictions)
# print(f"MAE: {mae_loss:.6f}")
#
# rmse_loss = root_mean_squared_error(actuals, predictions)
# print(f"RMSE: {rmse_loss:.6f}")

# Map index numbers back to index names
index_mapping = {0: "DJA", 1: "GSPC", 2: "IXIC", 3: "NYA"}
index_names = [index_mapping[idx] for idx in index_labels]  # Convert numerical index to name


dates = df_test.index[sequence_length:]
results = pd.DataFrame({"Predicted_Log_Return": predictions, "Actual_Log_Return": actuals,"Index": index_names}, index=dates)

results["Adjusted_Close"] = results.apply(lambda row: df_test.loc[(df_test.index == row.name) & (df_test["Index"] == row["Index"]), "Adjusted_close"].values[0], axis=1)

# mae, rmse only for the specific date range
fixed_start = "2023-01-01"
fixed_end = "2025-01-01"
results_filtered = results.loc[(results.index >= fixed_start) & (results.index <= fixed_end)]

print(f'results_filtered: {results_filtered}')

mae_loss = mean_absolute_error(results_filtered["Actual_Log_Return"], results_filtered["Predicted_Log_Return"])
print(f"MAE: {mae_loss:.6f}")

rmse_loss = root_mean_squared_error(results_filtered["Actual_Log_Return"], results_filtered["Predicted_Log_Return"])
print(f"RMSE: {rmse_loss:.6f}")

results_filtered.to_csv("predictions_conditional_lstm.csv")  # Save predictions for backtesting


print("\nSample Predictions:")
print(results_filtered.head(10))
