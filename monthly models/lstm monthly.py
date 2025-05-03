import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
from data_loader import load_monthly_data
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
    xs, ys, dates = [], [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:i + seq_length])
        ys.append(y.iloc[i + seq_length])
        dates.append(y.index[i + seq_length])  # Save the actual timestamp
    return np.array(xs), np.array(ys), dates


def get_dataloaders(X_train, y_train, X_valid, y_valid, X_test, y_test, seq_length, batch_size, device):
    train_seq, train_targets, _ = create_sequences(X_train, y_train, seq_length)
    valid_seq, valid_targets, _ = create_sequences(X_valid, y_valid, seq_length)
    test_seq, test_targets, test_dates = create_sequences(X_test, y_test, seq_length)

    # Convert the sequences, targets to torch tensors & create the Dataloaders
    train_dataset = TensorDataset(torch.tensor(train_seq, dtype=torch.float32).to(device), torch.tensor(train_targets, dtype=torch.float32).to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = TensorDataset(torch.tensor(valid_seq, dtype=torch.float32).to(device), torch.tensor(valid_targets, dtype=torch.float32).to(device))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(torch.tensor(test_seq, dtype=torch.float32).to(device), torch.tensor(test_targets, dtype=torch.float32).to(device))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader, test_dates


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

X_train, y_train, X_valid, y_valid, X_test, y_test, df_test, features, index_name = load_monthly_data(True, TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE)


pd.set_option('display.max_columns', None)
print(df_test)



# GSPC:
# hidden_size = 295
# num_layers = 3
# dropout = 0.0
# learning_rate = 0.006827668738770926
# batch_size = None  # Set your own value
# epochs = 75
# sequence_length = None  # Set your own value



# GSPC:
# scheduler = optim.lr_scheduler.OneCycleLR(
#     optimizer,
#     max_lr=0.09317123291598554,
#     steps_per_epoch=len(train_loader),
#     epochs=epochs,
#     pct_start=0.27171629170045175,
#     anneal_strategy='cos',
#     div_factor=24,
#     final_div_factor=324,
# )



# NYA:
hidden_size = 385
num_layers = 3
dropout = None  # Set your own value
learning_rate = 0.009998864221516746
batch_size = None  # Set your own value
epochs = 80
sequence_length = None  # Set your own value



# NYA:
# scheduler = optim.lr_scheduler.OneCycleLR(
#     optimizer,
#     max_lr=0.08964934976657521,
#     steps_per_epoch=len(train_loader),
#     epochs=epochs,
#     pct_start=0.4596697233185524,
#     anneal_strategy='cos',
#     div_factor=25,
#     final_div_factor=158,
# )




# IXIC:
# hidden_size = 55
# num_layers = None  # Set your own value
# dropout = 0.30
# learning_rate = 0.0001762726033427556
# batch_size = 112
# epochs = 45
# sequence_length = None  # Set your own value


#IXIC
# scheduler = optim.lr_scheduler.OneCycleLR(
#     optimizer,
#     max_lr=0.001533421161772768,
#     steps_per_epoch=len(train_loader),
#     epochs=epochs,
#     pct_start=0.29055728402173087,
#     anneal_strategy='cos',
#     div_factor=None  # Set your own value,
#     final_div_factor=223,
# )




# DJA
# hidden_size = 393
# num_layers = 2
# dropout = None  # Set your own value5
# learning_rate = 0.006812818193144822
# batch_size = None  # Set your own value
# epochs = 20
# sequence_length = None  # Set your own value


# #DJA
# scheduler = optim.lr_scheduler.OneCycleLR(
#     optimizer,
#     max_lr=0.06343151207296094,
#     steps_per_epoch=len(train_loader),
#     epochs=epochs,
#     pct_start=0.24137089437892872,
#     anneal_strategy='cos',
#     div_factor=None  # Set your own value,
#     final_div_factor=284,
# )


print('-' * 100)

model = LSTM(input_size=len(features), hidden_size=hidden_size, num_layers=num_layers, output_size=1, dropout=dropout).to(device)
criterion = nn.L1Loss()  # MAE loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.50)


train_loader, valid_loader, test_loader, test_dates = get_dataloaders(X_train, y_train, X_valid, y_valid, X_test, y_test, sequence_length, batch_size, device)
train_losses = []
valid_losses = []

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.08964934976657521,
    steps_per_epoch=len(train_loader),
    epochs=epochs,
    pct_start=0.4596697233185524,
    anneal_strategy='cos',
    div_factor=25,
    final_div_factor=158,
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
        # print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Valid Loss: {avg_valid_loss:.6f}, Last_LR: {scheduler.get_last_lr()[0]}")
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
print('-' * 100)

# mae_loss = mean_absolute_error(actuals, predictions)
# print(f"MAE: {mae_loss:.6f}")
#
# rmse_loss = root_mean_squared_error(actuals, predictions)
# print(f"RMSE: {rmse_loss:.6f}")

dates = df_test.index[sequence_length:]
index_names = df_test["Index"].iloc[sequence_length:].tolist()


results = pd.DataFrame({"Predicted_Log_Return": predictions, "Actual_Log_Return": actuals, "Index": index_names}, index=dates)

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


results_filtered.to_csv(f"monthly_predictions_basic_lstm_{index_name}.csv")

print("\nSample Predictions:")
print(results_filtered.head(10))
