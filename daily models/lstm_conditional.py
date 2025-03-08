import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
from data_loader_long import long_lstm_load_multiple_indices, combine_and_sort_data
import matplotlib.pyplot as plt


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, num_indices, embedding_dim):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

        self.index_embedding = nn.Embedding(num_indices, embedding_dim)  # Each index becomes a dense vector

    def forward(self, x, index):
        index_embedded = self.index_embedding(index)  # (batch_size, embedding_dim)

        self.h_state = index_embedded.unsqueeze(0).repeat(self.num_layers, 1, self.hidden_size)
        self.c_state = torch.zeros_like(self.h_state)  # Cell state - μπορει να παρει καποια τιμη σχετικη με το embedding?

        out, _ = self.lstm(x, (self.h_state, self.c_state))
        out = out[:, -1, :]  # Use the output from the last time step.
        out = self.fc(out)
        return out


def create_sequences(X, y, index_labels, seq_length):
    xs, ys, indices = [], [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:i + seq_length])
        ys.append(y.iloc[i + seq_length])
        indices.append(index_labels[i + seq_length])  # Store the corresponding index label
    return np.array(xs), np.array(ys), np.array(indices)


def get_dataloaders(X_train, y_train, index_train, X_valid, y_valid, index_valid, X_test, y_test, index_test, seq_length, batch_size, device):
    train_seq, train_targets, train_indices  = create_sequences(X_train, y_train, index_train, seq_length)
    valid_seq, valid_targets, valid_indices = create_sequences(X_valid, y_valid, index_valid, seq_length)
    test_seq, test_targets, test_indices = create_sequences(X_test, y_test, index_test, seq_length)

    # Convert the sequences, targets to torch tensors & create the Dataloaders
    train_dataset = TensorDataset(torch.tensor(train_seq, dtype=torch.float32).to(device), torch.tensor(train_targets, dtype=torch.float32).to(device)), torch.tensor(train_indices, dtype=torch.long).to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = TensorDataset(torch.tensor(valid_seq, dtype=torch.float32).to(device), torch.tensor(valid_targets, dtype=torch.float32).to(device)), torch.tensor(valid_indices, dtype=torch.long).to(device)

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(torch.tensor(test_seq, dtype=torch.float32).to(device), torch.tensor(test_targets, dtype=torch.float32).to(device)), torch.tensor(test_indices, dtype=torch.long).to(device)
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

TRAIN_START_DATE = "2000-01-01"
TRAIN_END_DATE = "2019-12-31"

VALID_START_DATE = "2020-01-01"
VALID_END_DATE = "2023-01-23"

TEST_START_DATE = "2023-01-24"
TEST_END_DATE = "2025-01-24"

combined_X_train, combined_y_train, combined_X_valid, combined_y_valid, combined_X_test, combined_y_test, df_test, features_set = long_lstm_load_multiple_indices(True, TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE)

# Align and Sort Data
X_train, y_train, X_valid, y_valid, X_test, y_test = combine_and_sort_data(combined_X_train, combined_y_train, combined_X_valid, combined_y_valid, combined_X_test, combined_y_test)

# #OPTUNA
# hidden_size = 53
# num_layers = 3
# dropout = 0.1
# learning_rate = 0.0001476
# batch_size = None  # Set your own value
# epochs = 100
# sequence_length = 50

hidden_size = 340
num_layers = None  # Set your own value
dropout = 0.15000000000000002
learning_rate = 0.0045622646026526196
batch_size = None  # Set your own value
epochs = 200
sequence_length = 60


print('-' * 100)

model = LSTM(input_size=len(features_set), hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout=dropout).to(device)
criterion = nn.L1Loss()  # MAE loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loader, valid_loader, test_loader = get_dataloaders(X_train, y_train, X_valid, y_valid, X_test, y_test, sequence_length, batch_size, device)
train_losses = []
valid_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = []
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        outputs = model(batch_x)  # batch_x.shape = (batch_size, seq_length, num_features)
        loss = criterion(outputs.view(-1), batch_y.view(-1))

        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    avg_train_loss = np.mean(train_loss)
    train_losses.append(avg_train_loss)

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
        actuals.append(batch_y.squeeze().cpu().numpy())

predictions = np.concatenate(predictions)
actuals = np.concatenate(actuals)
print('-' * 100)

mae_loss = mean_absolute_error(actuals, predictions)
print(f"MAE: {mae_loss:.6f}")

rmse_loss = root_mean_squared_error(actuals, predictions)
print(f"RMSE: {rmse_loss:.6f}")


dates = df_test.index[sequence_length:]
results = pd.DataFrame({"Predicted_Log_Return": predictions, "Actual_Log_Return": actuals}, index=dates)

print("\nSample Predictions:")
print(results.head(10))
