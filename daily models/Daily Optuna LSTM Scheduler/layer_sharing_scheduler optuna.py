import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from data_loader_layer_sharing import layer_sharing_load_daily_data
import optuna


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


def objective(trial):
    hidden_size = 154
    num_layers = None  # Set your own value
    dropout = 0.1
    learning_rate = 0.0013378263486894257
    batch_size = 112
    sequence_length = None  # Set your own value0
    epochs = 45
    num_heads = None  # Set your own value

    # Scheduler hyperparameters
    max_lr = trial.suggest_float("max_lr", learning_rate * 5, learning_rate * 20)
    pct_start = trial.suggest_float("pct_start", 0.1, 0.5)
    div_factor = trial.suggest_int("div_factor", 5, 25)
    final_div_factor = trial.suggest_int("final_div_factor", 10, 500)

    # Skip invalid configurations (e.g. hidden_size must be divisible by num_heads)
    if hidden_size % num_heads != 0:
        raise optuna.exceptions.TrialPruned()

    model = LSTM(input_size=len(features), hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, output_size=1, num_heads=num_heads, num_indices=4, embedding_dim=None  # Set your own value).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, valid_loader, test_loader = get_dataloaders(X_train, y_train, X_valid, y_valid, X_test, y_test, sequence_length, batch_size, device)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=pct_start,
        anneal_strategy='cos',
        cycle_momentum=False,
        div_factor=div_factor,
        final_div_factor=final_div_factor
    )

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 30

    for epoch in range(epochs):
        model.train()
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

        # Validation
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

        if avg_valid_loss < best_val_loss:
            best_val_loss = avg_valid_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} with best validation loss {best_val_loss:.6f}")
                break

    return best_val_loss


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best hyperparameters:")
print(study.best_trial)
