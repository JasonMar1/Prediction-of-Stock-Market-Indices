from data_loader_DeepAR import return_splits
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

from pytorch_forecasting import TimeSeriesDataSet, DeepAR, GroupNormalizer
from pytorch_forecasting.metrics import NormalDistributionLoss
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner



from lightning.pytorch import seed_everything
seed_everything(42, workers=True)


TRAIN_START_DATE = "2006-01-01"
TRAIN_END_DATE = "2019-12-31"

VALID_START_DATE = "2020-01-01"
VALID_END_DATE = "2023-01-23"

TEST_START_DATE = "2023-01-24"
TEST_END_DATE = "2025-01-24"


df_train, df_val, df_test, feature_columns = return_splits(TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE)

max_encoder_length = 15
prediction_length = 1

train_dataset = TimeSeriesDataSet(
    data=df_train,
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
    allow_missing_timesteps=True
)

val_dataset = TimeSeriesDataSet.from_dataset(
    train_dataset,
    df_val,
    stop_randomization=True
)

test_dataset = TimeSeriesDataSet.from_dataset(
    train_dataset,
    df_test,
    stop_randomization=True
)


batch_size = None  # Set your own value
train_loader = train_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_loader = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
test_loader = test_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)


model = DeepAR.from_dataset(
    train_dataset,
    learning_rate=0.001,
    optimizer="adam",
    dropout=0.1,
    hidden_size=64,
    rnn_layers=2,
    loss=NormalDistributionLoss(),
    log_interval=10,
    log_val_interval=0
)

early_stop = EarlyStopping(monitor="val_loss", patience=5)
trainer = Trainer(
    max_epochs=30,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    callbacks=[early_stop],
    gradient_clip_val=0.1  # limits how big the gradients can get to keep training stable
)

# # find the optimal learning rate
# res = Tuner(trainer).lr_find(
#     model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, early_stop_threshold=1000.0, max_lr=0.01,
# )

trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
print(trainer.optimizers)


# 1) Predict median, unpacking all returned values
#    predict(...) returns: (predictions, x, y, index, decoder_lengths)
predictions, x, y, index, decoder_lengths = model.predict(
    test_loader,
    mode="prediction",
    return_x=True,
    return_y=False,               # we don’t actually need y here
    return_index=True,            # returns group index
    return_decoder_lengths=False  # not needed for single-step forecasts
)

# extract numpy arrays
median = predictions[:, 0].cpu().numpy()  # [N] median forecasts
index_ids = x["groups"].squeeze(-1).cpu().numpy()  # [N]
time_idxs = x["decoder_time_idx"][:, 0].cpu().numpy()  # [N]

# build DataFrame of median forecasts
pred_df = pd.DataFrame({
    "index_id": index_ids,
    "time_idx": time_idxs,
    "predicted_median": median
})

# merge with truths
true_df = df_test[["index_id", "time_idx", "date", "target"]].rename(columns={"target": "true"})
index_mapping = {"DJA": 0, "GSPC": 1, "IXIC": 2, "NYA": 3}

# map true_df index_id from string → int
true_df["index_id"] = true_df["index_id"].map(index_mapping)

results_df = pred_df.merge(true_df, on=["index_id", "time_idx"], how="left")


mae  = mean_absolute_error(results_df["true"], results_df["predicted_median"])
rmse = np.sqrt(mean_squared_error(results_df["true"], results_df["predicted_median"]))
print(f"MAE:  {mae:.6f}")
print(f"RMSE: {rmse:.6f}")


results_df[["index_id", "date", "true", "predicted_median"]].to_csv(
    "deepar_predictions.csv", index=False
)
print("Predictions saved to deepar_predictions.csv")