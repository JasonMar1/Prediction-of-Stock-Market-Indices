import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
from data_loader_layer_sharing import layer_sharing_load_daily_data


class SharedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size, num_heads):
        super(SharedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Shared LSTM (applied independently per index)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Multihead self-attention across indices
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=False)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (indices, batch_size, sequence_length, input_size)
        indices, batch_size, sequence_length, input_size = x.shape

        outputs = []
        for i in range(indices):
            xi = x[i]  # (batch_size, sequence_length, input_size)
            out, _ = self.lstm(xi)  # (batch_size, sequence_length, hidden_size)
            out = out[:, -1, :]  # last layer's hidden state: (batch_size, hidden_size)
            outputs.append(out)

        out_total = torch.stack(outputs, dim=0)  # (indices, batch_size, hidden_size)

        # Self-attention expects (indices, batch_size, hidden_size)
        attn_output, _ = self.attention(out_total, out_total, out_total)

        # Final dense layer for each index
        x_out = attn_output.permute(1, 0, 2)  # (batch_size, indices, hidden_size)
        x_out = self.fc(x_out)  # (batch_size, indices, 1)

        return x_out.permute(1, 0, 2)  # (indices, batch_size, 1)


def create_sequences(X_list, y_list, seq_length, device):
    all_X, all_y = [], []
    for X_df, y_df in zip(X_list, y_list):

        X_vals, y_vals = X_df.values, y_df.values  # numpy arrays to speed up slicing
        xs, ys = [], []
        for i in range(len(X_vals) - seq_length):
            xs.append(X_vals[i:i + seq_length])
            ys.append(y_vals[i + seq_length])

        xs_np = np.array(xs, dtype=np.float32)
        ys_np = np.array(ys, dtype=np.float32)

        xs_tensor = torch.tensor(xs_np, dtype=torch.float32).to(device)
        ys_tensor = torch.tensor(ys_np, dtype=torch.float32).to(device)

        all_X.append(xs_tensor)
        all_y.append(ys_tensor)

    return torch.stack(all_X), torch.stack(all_y)  # (indices, samples, seq_len, features), (indices, samples)


def get_dataloaders(X_train, y_train, X_valid, y_valid, X_test, y_test, sequence_length, batch_size, device):
    def prepare(X, y, shuffle):
        X_seq, y_seq = create_sequences(X, y, sequence_length, device)
        # Flatten index dimension into batch
        X_flat = X_seq.permute(1, 0, 2, 3)  # (samples, indices, seq_len, features)
        y_flat = y_seq.permute(1, 0)        # (samples, indices)
        dataset = TensorDataset(X_flat, y_flat)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return (
        prepare(X_train, y_train, shuffle=True),
        prepare(X_valid, y_valid, shuffle=False),
        prepare(X_test, y_test, shuffle=False)
    )


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_START_DATE = "2006-01-01"
TRAIN_END_DATE = "2019-11-30"

VALID_START_DATE = "2019-12-01"
VALID_END_DATE = "2022-08-31"

TEST_START_DATE = "2022-10-01"  # worst case scenario, having sequence length equal to 3 months + dropping 1 month for data-leakage
TEST_END_DATE = "2025-01-01"

X_train, y_train, X_valid, y_valid, X_test, y_test, df_test, features = layer_sharing_load_daily_data(True, TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE)

# 500 trials
hidden_size = 494
num_layers = None  # Set your own value
dropout = 0.35
learning_rate = 0.00560631567046721
batch_size = 112
sequence_length = None  # Set your own value0
epochs = 40
num_heads = None  # Set your own value


# # 2000 trials
# hidden_size = 332
# num_layers = 2
# dropout = None  # Set your own value
# learning_rate = 0.001043578040821334
# batch_size = 112
# sequence_length = None  # Set your own value0
# epochs = None  # Set your own value
# num_heads = None  # Set your own value


print('-' * 100)

model = SharedLSTM(input_size=len(features), hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, output_size=1, num_heads=num_heads).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loader, valid_loader, test_loader = get_dataloaders(X_train, y_train, X_valid, y_valid, X_test, y_test, sequence_length, batch_size, device)
train_losses = []
valid_losses = []

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.006098274606304232,
    steps_per_epoch=len(train_loader),
    epochs=epochs,
    pct_start=0.2749320007622838,
    anneal_strategy='cos',
    div_factor=6,
    final_div_factor=None  # Set your own value5,
)

for epoch in range(epochs):
    model.train()
    train_loss = []
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_x = batch_x.permute(1, 0, 2, 3)  # (indices, batch, seq_len, features)
        batch_y = batch_y.permute(1, 0).unsqueeze(-1)  # (indices, batch, 1)

        outputs = model(batch_x)
        loss = criterion(outputs.reshape(-1), batch_y.reshape(-1))

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

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_x = batch_x.permute(1, 0, 2, 3)  # (indices, batch, seq_len, features)
            batch_y = batch_y.permute(1, 0).unsqueeze(-1)  # (indices, batch, 1)

            outputs = model(batch_x)
            loss = criterion(outputs.reshape(-1), batch_y.reshape(-1))
            valid_loss.append(loss.item())

    avg_valid_loss = np.mean(valid_loss)
    valid_losses.append(avg_valid_loss)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_valid_loss:.6f}")


model.eval()
predictions = []
actuals = []

flat_preds, flat_targets, flat_indices = [], [], []


with torch.no_grad():
    for batch_x, batch_y in test_loader:

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_x = batch_x.permute(1, 0, 2, 3)  # (indices, batch, seq_len, features)
        batch_y = batch_y.permute(1, 0).unsqueeze(-1)  # (indices, batch, 1)

        outputs = model(batch_x)
        predictions = outputs.squeeze().cpu().numpy()  # shape: (indices, batch)
        actuals = batch_y.squeeze().cpu().numpy()  # shape: (indices, batch)

        # Handle case where batch size might be 1
        if predictions.ndim == 1:
            predictions = predictions[:, None]
            actuals = actuals[:, None]

        for j in range(predictions.shape[1]):  # loop over batch
            for i, idx in enumerate(["DJA", "GSPC", "IXIC", "NYA"]):  # loop over indices
                flat_preds.append(predictions[i][j])
                flat_targets.append(actuals[i][j])
                flat_indices.append(idx)

# Drop first `sequence_length` rows per index from df_test to match
# Now drop the first `sequence_length` rows per index in df_test
mask = df_test.groupby("Index").cumcount() >= sequence_length
df_test_filtered = df_test[mask]



print(df_test)
print(50*'--')
print(df_test_filtered)


results = pd.DataFrame({
    "Predicted_Log_Return": flat_preds,
    "Actual_Log_Return": flat_targets,
    "Index": flat_indices,
    "Date": df_test_filtered.index
})

# Reset df_test_filtered to make Date a column
df_test_reset = df_test_filtered.reset_index()

# Merge on both Date and Index to bring in Adjusted_Close
merged = pd.merge(results, df_test_reset[["Date", "Index", "Adjusted_close"]], on=["Date", "Index"], how="left")

# Rename and set index
merged = merged.rename(columns={"Adjusted_close": "Adjusted_Close"})
merged.set_index("Date", inplace=True)



# mae, rmse only for the specific date range
fixed_start = "2023-01-01"
fixed_end = "2025-01-01"
results_filtered = merged.loc[(merged.index >= fixed_start) & (merged.index <= fixed_end)]

print(f'results_filtered: {results_filtered}')

mae_loss = mean_absolute_error(results_filtered["Actual_Log_Return"], results_filtered["Predicted_Log_Return"])
rmse_loss = root_mean_squared_error(results_filtered["Actual_Log_Return"], results_filtered["Predicted_Log_Return"])

print(f"MAE: {mae_loss:.6f}")
print(f"RMSE: {rmse_loss:.6f}")

results_filtered.to_csv("predictions_layer_sharing.csv")