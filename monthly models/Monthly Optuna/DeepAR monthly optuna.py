from data_loader_DeepAR import return_splits_monthly
import torch
import optuna

from pytorch_forecasting import TimeSeriesDataSet, DeepAR, GroupNormalizer
from pytorch_forecasting.metrics import NormalDistributionLoss
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping


def objective(trial):
    print("-"*50)
    print(f"\n Trial {trial.number}")

    # Define hyperparameter search space
    hidden_size = trial.suggest_int("hidden_size", 32, 512, log=True)
    rnn_layers = trial.suggest_int("rnn_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 128, step=16)
    max_encoder_length = trial.suggest_int("max_encoder_length", 1, 3)
    epochs = trial.suggest_int("epochs", 10, 100, step=5)
    # epochs = 100

    train_dataset = TimeSeriesDataSet(
        df_train,
        time_idx="time_idx",
        target="target",
        group_ids=["index_id"],  # time step of each observation (sequential per group)
        max_encoder_length=max_encoder_length,
        max_prediction_length=prediction_length,
        static_categoricals=["index_id"],  # Categorical variables that don't change over time
        time_varying_known_categoricals=["month"],  # Categorical variables known in advance and change over time
        time_varying_known_reals=feature_columns,  # Real-valued features that are known at prediction time
        time_varying_unknown_reals=["target"],   # Real-valued features that are not known in advance and must be predicted
        target_normalizer=GroupNormalizer(groups=["index_id"]),  # z-score
        allow_missing_timesteps=True,
    )

    val_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset, df_val, stop_randomization=True
    )

    train_loader = train_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=6, persistent_workers=True)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=6, persistent_workers=True)

    model = DeepAR.from_dataset(
        train_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        rnn_layers=rnn_layers,
        dropout=dropout,
        loss=NormalDistributionLoss(),
        log_interval=-1,
        log_val_interval=-1
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=20)
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[early_stop],
        gradient_clip_val=0.1,  # limits how big the gradients can get to keep training stable
        enable_progress_bar=False,
        num_sanity_val_steps=0  # <- disables the pre-training sanity check
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    val_loss = trainer.callback_metrics["val_loss"].item()

    print(f"Trial {trial.number} finished with val_loss: {val_loss:.5f}")
    return val_loss


# Callback to log current best trial
def log_best_trial(study, trial):
    print(f"Best trial so far: {study.best_trial.number} with val_loss: {study.best_value:.5f}")

if __name__ == "__main__":

    seed_everything(42, workers=True)

    # Static date splits
    TRAIN_START_DATE = "2006-01-01"
    TRAIN_END_DATE = "2019-11-30"

    VALID_START_DATE = "2019-12-01"
    VALID_END_DATE = "2022-08-31"

    TEST_START_DATE = "2022-10-01"  # worst case scenario, having sequence length equal to 3 months + dropping 1 month for data-leakage
    TEST_END_DATE = "2025-01-01"

    df_train, df_val, df_test, feature_columns = return_splits_monthly(
        TRAIN_START_DATE, TRAIN_END_DATE,
        VALID_START_DATE, VALID_END_DATE,
        TEST_START_DATE, TEST_END_DATE
    )

    prediction_length = 1  # 1-month ahead

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, callbacks=[log_best_trial])

    print("Best hyperparameters:")
    print(study.best_trial)