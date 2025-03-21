import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
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


def get_dataloaders(X_train, y_train, X_valid, y_valid, X_test, y_test, seq_length, batch_size, device):
    train_seq, train_targets = create_sequences(X_train, y_train, seq_length)
    valid_seq, valid_targets = create_sequences(X_valid, y_valid, seq_length)
    test_seq, test_targets = create_sequences(X_test, y_test, seq_length)

    # Convert the sequences, targets to torch tensors & create the Dataloaders
    train_dataset = TensorDataset(torch.tensor(train_seq, dtype=torch.float32).to(device), torch.tensor(train_targets, dtype=torch.float32).to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = TensorDataset(torch.tensor(valid_seq, dtype=torch.float32).to(device), torch.tensor(valid_targets, dtype=torch.float32).to(device))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(torch.tensor(test_seq, dtype=torch.float32).to(device), torch.tensor(test_targets, dtype=torch.float32).to(device))
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
TRAIN_END_DATE = "2019-12-31"

VALID_START_DATE = "2020-01-01"
VALID_END_DATE = "2023-01-23"

TEST_START_DATE = "2023-01-24"
TEST_END_DATE = "2025-01-24"

X_train, y_train, X_valid, y_valid, X_test, y_test, df_test, features = load_daily_data_log_returns(True, TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE)

print(f"Train samples: {X_train.shape[0]}")
print(f"Validation samples: {X_valid.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")



print(f"y_train samples: {y_train.shape[0]}")
print(f"y_valid samples: {y_valid.shape[0]}")
print(f"y_test samples: {y_test.shape[0]}")


# #OPTUNA
# hidden_size = 53
# num_layers = 3
# dropout = 0.1
# learning_rate = 0.0001476
# batch_size = None  # Set your own value
# epochs = 100
# sequence_length = 50

hidden_size = 108
num_layers = None  # Set your own value
dropout = None  # Set your own value5
learning_rate = 0.0019594494528614313
batch_size = None  # Set your own value
epochs = 100
sequence_length = 50


print('-' * 100)

model = LSTM(input_size=len(features), hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout=dropout).to(device)
criterion = nn.L1Loss()  # MAE loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.50)


train_loader, valid_loader, test_loader = get_dataloaders(X_train, y_train, X_valid, y_valid, X_test, y_test, sequence_length, batch_size, device)
train_losses = []
valid_losses = []

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=learning_rate * 10,  # Peak LR (usually 10x initial LR)
    steps_per_epoch=len(train_loader),  # Number of batches per epoch
    epochs=epochs,  # Total number of epochs
    pct_start=0.3,  # Percentage of cycle spent increasing LR
    anneal_strategy='cos',  # Cosine annealing for smooth decay
    div_factor=10,  # Initial LR is max_lr / div_factor
    final_div_factor=100,  # Final LR is max_lr / final_div_factor
)


for epoch in range(epochs):
    model.train()
    train_loss = []
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        outputs = model(batch_x)  # batch_x.shape = (batch_size, seq_length, num_features)
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
        for batch_x, batch_y in valid_loader:
            outputs = model(batch_x)
            loss = criterion(outputs.view(-1), batch_y.view(-1))
            valid_loss.append(loss.item())

    avg_valid_loss = np.mean(valid_loss)
    valid_losses.append(avg_valid_loss)


    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Valid Loss: {avg_valid_loss:.6f}, Last_LR: {scheduler.get_last_lr()[0]}")
        # print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Valid Loss: {avg_valid_loss:.6f}")

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
print('-' * 100)

mae_loss = mean_absolute_error(actuals, predictions)
print(f"MAE: {mae_loss:.6f}")

rmse_loss = root_mean_squared_error(actuals, predictions)
print(f"RMSE: {rmse_loss:.6f}")


dates = df_test.index[sequence_length:sequence_length + len(predictions)]
results = pd.DataFrame({"Predicted_Log_Return": predictions, "Actual_Log_Return": actuals}, index=dates)

print("\nSample Predictions:")
print(results.head(10))
