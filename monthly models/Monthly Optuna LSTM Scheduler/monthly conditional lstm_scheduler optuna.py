import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from data_loader_conditional import conditional_lstm_load_multiple_indices, combine_and_sort_data
import optuna

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_START_DATE = "2006-01-01"
TRAIN_END_DATE = "2019-11-30"

VALID_START_DATE = "2019-12-01"
VALID_END_DATE = "2022-08-31"

TEST_START_DATE = "2022-10-01"  # worst case scenario, having sequence length equal to 3 months + dropping 1 month for data-leakage
TEST_END_DATE = "2025-01-01"


combined_X_train, combined_y_train, index_train, combined_X_valid, combined_y_valid, index_valid, combined_X_test, combined_y_test, index_test, df_test, features = conditional_lstm_load_multiple_indices( 'monthly',
    True, TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE)

# Align and Sort Data
X_train, y_train, index_train, X_valid, y_valid, index_valid, X_test, y_test, index_test = combine_and_sort_data(
    combined_X_train, combined_y_train, index_train, combined_X_valid, combined_y_valid, index_valid, combined_X_test,
    combined_y_test, index_test)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, num_indices, embedding_dim):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

        self.h_state = nn.Embedding(num_embeddings=num_indices, embedding_dim=embedding_dim)
        self.c_state = nn.Embedding(num_embeddings=num_indices, embedding_dim=embedding_dim)

        # Projection layers to map embedding dimension to LSTM hidden_size
        self.h_state_proj = nn.Linear(embedding_dim, hidden_size)
        self.c_state_proj = nn.Linear(embedding_dim, hidden_size)

    def forward(self, x, index):
        # Retrieve the initial state vectors for the batch & match the LSTM requirements
        h_embedding = self.h_state(index)  # (batch_size, embedding_dim)
        c_embedding = self.c_state(index)

        # Project embeddings to match LSTM's hidden_size
        h_proj = self.h_state_proj(h_embedding)  # Shape: (batch_size, hidden_size)
        c_proj = self.c_state_proj(c_embedding)  # Shape: (batch_size, hidden_size)

        h_state = h_proj.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_state = c_proj.unsqueeze(0).repeat(self.num_layers, 1, 1)

        # if h_state == c_state:
        #     print("h_state and c_state are equal")

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


def get_dataloaders(X_train, y_train, index_train, X_valid, y_valid, index_valid, seq_length, batch_size, device):
    train_seq, train_targets, train_indices = create_sequences(X_train, y_train, index_train, seq_length)
    valid_seq, valid_targets, valid_indices = create_sequences(X_valid, y_valid, index_valid, seq_length)

    # Convert the sequences, targets, indices to torch tensors & create the Dataloaders
    train_dataset = TensorDataset(torch.tensor(train_seq, dtype=torch.float32).to(device),
                                  torch.tensor(train_targets, dtype=torch.float32).to(device),
                                  torch.tensor(train_indices, dtype=torch.long).to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = TensorDataset(torch.tensor(valid_seq, dtype=torch.float32).to(device),
                                  torch.tensor(valid_targets, dtype=torch.float32).to(device),
                                  torch.tensor(valid_indices, dtype=torch.long).to(device))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader


def objective(trial):
    hidden_size = None  # Set your own value
    num_layers = None  # Set your own value
    dropout = None  # Set your own value
    learning_rate = None  # Set your own value
    batch_size = None  # Set your own value
    epochs = None  # Set your own value
    seq_length = 2

    patience = 30

    # Scheduler-specific hyperparameters to be tuned
    max_lr = trial.suggest_float("max_lr", learning_rate * 5, learning_rate * 20)
    pct_start = trial.suggest_float("pct_start", 0.1, 0.5)
    div_factor = trial.suggest_int("div_factor", 5, 25)
    final_div_factor = trial.suggest_int("final_div_factor", 10, 500)


    train_loader, valid_loader = get_dataloaders(X_train, y_train, index_train, X_valid, y_valid, index_valid, seq_length, batch_size, device)


    model = LSTM(input_size=len(features), hidden_size=hidden_size, num_layers=num_layers, output_size=1,
                 dropout=dropout,
                 num_indices=4, embedding_dim=None  # Set your own value).to(device)
    criterion = nn.L1Loss()  # MAE loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y, batch_index in train_loader:
            optimizer.zero_grad()

            outputs = model(batch_x, batch_index)  # batch_x.shape = (batch_size, seq_length, num_features)
            loss = criterion(outputs.view(-1), batch_y.view(-1))

            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        valid_loss = []
        with torch.no_grad():
            for batch_x, batch_y, batch_index in valid_loader:
                outputs = model(batch_x, batch_index)
                loss = criterion(outputs.view(-1), batch_y.view(-1))
                valid_loss.append(loss.item())

        avg_valid_loss = np.mean(valid_loss)


        # # Optuna pruning (stops bad trials early)
        # trial.report(avg_valid_loss, epoch)
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()

        # Early stopping
        if avg_valid_loss < best_val_loss:
            best_val_loss = avg_valid_loss
            patience_counter = 0  # Reset patience since a better model is found
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} with best validation loss {best_val_loss:.6f}")
                break  # Stop training

    return best_val_loss


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=500)

print("Best hyperparameters:")
print(study.best_trial)
