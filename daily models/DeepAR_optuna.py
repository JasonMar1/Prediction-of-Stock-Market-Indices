from data_loader_DeepAR import return_splits
import pandas as pd
import numpy as np
import torch
import optuna

from sklearn.metrics import mean_absolute_error, mean_squared_error
from pytorch_forecasting import TimeSeriesDataSet, DeepAR, GroupNormalizer
from pytorch_forecasting.metrics import NormalDistributionLoss
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

seed_everything(42, workers=True)

# Static date splits
TRAIN_START_DATE = "2006-01-01"
TRAIN_END_DATE = "2019-12-31"
VALID_START_DATE = "2020-01-01"
VALID_END_DATE = "2023-01-23"
TEST_START_DATE = "2023-01-24"
TEST_END_DATE = "2025-01-24"

df_train, df_val, df_test, feature_columns = return_splits(
    TRAIN_START_DATE, TRAIN_END_DATE,
    VALID_START_DATE, VALID_END_DATE,
    TEST_START_DATE, TEST_END_DATE
)

prediction_length = 1  # 1-day ahead

# Optuna objective function
def objective(trial):
    print(f"\n🔍 Trial {trial.number} - Trying parameters: {trial.params}")

    max_encoder_length = trial.suggest_int("max_encoder_length", 10, 60)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    hidden_size = trial.suggest_int("hidden_size", 16, 128)
    rnn_layers = trial.suggest_int("rnn_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    max_epochs = trial.suggest_int("max_epochs", 10, 100)

    train_dataset = TimeSeriesDataSet(
        df_train,
        time_idx="time_idx",
        target="target",
        group_ids=["index_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=prediction_length,
        static_categoricals=["index_id"],
        time_varying_known_categoricals=["day_of_week"],
        time_varying_known_reals=feature_columns,
        time_varying_unknown_reals=["target"],
        target_normalizer=GroupNormalizer(groups=["index_id"]),
        allow_missing_timesteps=True,
    )

    val_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset, df_val, stop_randomization=True
    )

    train_loader = train_dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    model = DeepAR.from_dataset(
        train_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        rnn_layers=rnn_layers,
        dropout=dropout,
        loss=NormalDistributionLoss(),
        log_interval=10,
        log_val_interval=0
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=20)
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[early_stop],
        gradient_clip_val=0.1  # limits how big the gradients can get to keep training stable
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    val_loss = trainer.callback_metrics["val_loss"].item()
    return val_loss

# Run Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=300)

# Best result
print("Best hyperparameters found:")
for k, v in study.best_params.items():
    print(f"{k}: {v}")
