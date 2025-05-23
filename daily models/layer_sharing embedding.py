import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
from data_loader_layer_sharing import layer_sharing_load_daily_data


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size, num_heads, num_indices, embedding_dim):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        self.h_state = nn.Embedding(num_embeddings=num_indices, embedding_dim=embedding_dim)
        self.c_state = nn.Embedding(num_embeddings=num_indices, embedding_dim=embedding_dim)

        # map embedding dimension to LSTM hidden_size
        self.h_state_proj = nn.Linear(embedding_dim, hidden_size)
        self.c_state_proj = nn.Linear(embedding_dim, hidden_size)


        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=False)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, index_tensor):
        # x: (indices, batch_size, sequence_length, features)
        indices, batch_size, sequence_length, input_size = x.shape
        outputs = []


        for i in range(indices):
            xi = x[i]  # (batch_size, sequence_length, features)
            index_ids = index_tensor[:, i]  # (batch,)

            h_embedding = self.h_state(index_ids)
            c_embedding = self.c_state(index_ids)

            h_proj = self.h_state_proj(h_embedding)  # (batch_size, hidden_size)
            c_proj = self.c_state_proj(c_embedding)  # (batch_size, hidden_size)

            h_state = h_proj.unsqueeze(0).repeat(self.num_layers, 1, 1)
            c_state = c_proj.unsqueeze(0).repeat(self.num_layers, 1, 1)

            out, _ = self.lstm(xi, (h_state, c_state))  # (batch_size, sequence_length, hidden_size)

            out = out[:, -1, :]  # last layer's hidden state: (batch_size, hidden_size)
            outputs.append(out)

        out_total = torch.stack(outputs, dim=0)  # (indices, batch_size, hidden_size)

        attn_output, _ = self.attention(out_total, out_total, out_total)

        x_out = attn_output.permute(1, 0, 2)  # (batch_size, indices, hidden_size)
        x_out = self.fc(x_out)  # (batch_size, indices, 1)

        return x_out.permute(1, 0, 2)  # (indices, batch_size, 1)



def create_sequences(X_list, y_list, seq_length, device):
    # X_list, y_list: 1 DataFrame ανά index

    all_X, all_y, all_idx = [], [], []

    for idx, (X_df, y_df) in enumerate(zip(X_list, y_list)):
        X_vals, y_vals = X_df.values, y_df.values
        xs, ys, ids = [], [], []

        for i in range(len(X_vals) - seq_length):
            xs.append(X_vals[i:i + seq_length])
            ys.append(y_vals[i + seq_length])
            ids.append(idx)

        xs = torch.tensor(np.array(xs, dtype=np.float32),device=device)
        ys = torch.tensor(np.array(ys, dtype=np.float32), device=device)
        ids= torch.tensor(np.array(ids, dtype=np.int64), device=device)

        all_X.append(xs)
        all_y.append(ys)
        all_idx.append(ids)

    return torch.stack(all_X, dim=0), torch.stack(all_y, dim=0), torch.stack(all_idx, dim=0)


def get_dataloaders(X_train, y_train, X_valid, y_valid, X_test, y_test, seq_length, batch_size, device):
    def prepare(X_list, y_list, shuffle):
        X_seq, y_seq, idx_seq = create_sequences(X_list, y_list, seq_length, device)  # (indices, batch_size, seq_length, features), (indices, batch_size), (indices, batch_size)

        X_flat = X_seq.permute(1, 0, 2, 3)  # (batch_size, indices, seq_len, features)
        y_flat = y_seq.permute(1, 0)        # (batch_size, indices)
        idx_flat = idx_seq.permute(1, 0)    # (batch_size, indices)

        dataset = TensorDataset(X_flat, y_flat, idx_flat)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return prepare(X_train, y_train, shuffle=True), prepare(X_valid, y_valid, shuffle=False), prepare(X_test,  y_test,  shuffle=False)


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_START_DATE = "2006-01-01"
TRAIN_END_DATE = "2019-11-30"

VALID_START_DATE = "2019-12-01"
VALID_END_DATE = "2022-08-31"

TEST_START_DATE = "2022-10-01"  # worst case scenario, having sequence length equal to 3 months + dropping 1 month for data-leakage
TEST_END_DATE = "2025-01-01"

X_train, y_train, X_valid, y_valid, X_test, y_test, df_test, features = layer_sharing_load_daily_data(True, TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE)


hidden_size = 332
num_layers = 2
dropout = None  # Set your own value
learning_rate = 0.001043578040821334
batch_size = 112
sequence_length = None  # Set your own value0
epochs = 5
num_heads = None  # Set your own value


print('-' * 100)

model = LSTM(input_size=len(features), hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, output_size=1, num_heads=num_heads, num_indices=4, embedding_dim=None  # Set your own value).to(device)
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
    for batch_x, batch_y, batch_index in train_loader:
        optimizer.zero_grad()

        batch_x, batch_y, batch_index = batch_x.to(device), batch_y.to(device), batch_index.to(device)
        batch_x = batch_x.permute(1, 0, 2, 3)  # (indices, batch, seq_len, features)
        batch_y = batch_y.permute(1, 0).unsqueeze(-1)  # (indices, batch, 1)

        outputs = model(batch_x, batch_index)
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
        for batch_x, batch_y, batch_index in valid_loader:

            batch_x, batch_y, batch_index = batch_x.to(device), batch_y.to(device), batch_index.to(device)
            batch_x = batch_x.permute(1, 0, 2, 3)  # (indices, batch, seq_len, features)
            batch_y = batch_y.permute(1, 0).unsqueeze(-1)  # (indices, batch, 1)

            outputs = model(batch_x, batch_index)
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
    for batch_x, batch_y, batch_index in test_loader:

        batch_x, batch_y, batch_index = batch_x.to(device), batch_y.to(device), batch_index.to(device)
        batch_x = batch_x.permute(1, 0, 2, 3)  # (indices, batch, seq_len, features)
        batch_y = batch_y.permute(1, 0).unsqueeze(-1)  # (indices, batch, 1)

        outputs = model(batch_x, batch_index)
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

# drop the first 'sequence_length' rows per index in df_test
mask = df_test.groupby("Index").cumcount() >= sequence_length
df_test_filtered = df_test[mask]



# print(df_test)
# print(50*'--')
# print(df_test_filtered)


results = pd.DataFrame({
    "Predicted_Log_Return": flat_preds,
    "Actual_Log_Return": flat_targets,
    "Index": flat_indices,
    "Date": df_test_filtered.index
})

# Reset df_test_filtered to make Date a column
df_test_reset = df_test_filtered.reset_index()

# Merge on Date & Index to bring in Adjusted_close
merged = pd.merge(results, df_test_reset[["Date", "Index", "Adjusted_close"]], on=["Date", "Index"], how="left")

# Rename and set index
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