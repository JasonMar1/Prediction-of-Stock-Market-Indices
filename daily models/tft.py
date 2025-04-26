from data_loader_tft import return_splits_tft
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping


if __name__ == "__main__":
    seed_everything(42, workers=True)

    TRAIN_START_DATE = "2006-01-01"
    TRAIN_END_DATE = "2019-12-31"

    VALID_START_DATE = "2020-01-01"
    VALID_END_DATE = "2023-01-23"

    TEST_START_DATE = "2023-01-24"
    TEST_END_DATE = "2025-01-24"

    df_train, df_val, df_test, feature_columns = return_splits_tft(TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE)

    max_encoder_length = 60
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
        allow_missing_timesteps=True,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True
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

    batch_size = 112
    train_loader = train_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=8, persistent_workers=True)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=8, persistent_workers=True)
    test_loader = test_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=8, persistent_workers=True)

    model = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=0.0009144593723706877,
        optimizer="adam",
        dropout=0.5,
        hidden_size=43,
        attention_head_size=1,
        loss=QuantileLoss(),
        output_size=7,  # Προβλέπει quantiles 0.1, 0.2, ..., 0.9
        log_interval=-1,
        log_val_interval=-1
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=5)
    trainer = Trainer(
        max_epochs=2,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[early_stop],
        gradient_clip_val=0.1  # limits how big the gradients can get to keep training stable
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


    # Predict median, unpacking all returned values
    predictions, x, _, _, _ = model.predict(
        test_loader,
        mode="raw",
        return_x=True
    )  # mode="raw" → επιστρέφει όλα τα quantiles (όλο το output tensor) και μετά επιλέγεις ποιο θέλεις.

    from pytorch_forecasting.utils import to_list

    # Step 1: Check if Output object
    if hasattr(predictions, "prediction"):
        predictions = predictions.prediction

    # Step 2: Ensure it's list
    predictions = to_list(predictions)

    # Step 3: Merge
    predictions = torch.cat(predictions, dim=0)

    print(predictions.shape)
    print(predictions[0, 0, :])  # Δείγμα predictions για ένα δείγμα και ένα timestep
    print(predictions[0, 0, 4])

    median = predictions[:, 0, 4].cpu().numpy()
    index_ids = x["groups"].squeeze(-1).cpu().numpy()  # [N]
    time_idxs = x["decoder_time_idx"][:, 0].cpu().numpy()  # [N]

    # build a df of median forecasts
    pred_df = pd.DataFrame({
        "index_id": index_ids,
        "time_idx": time_idxs,
        "predicted_median": median
    })

    # merge with the real returns
    true_df = df_test[["index_id", "time_idx", "date", "target"]].rename(columns={"target": "true"})
    index_mapping = {"DJA": 0, "GSPC": 1, "IXIC": 2, "NYA": 3}

    # map true_df index_id from string to int
    true_df["index_id"] = true_df["index_id"].map(index_mapping)

    results_df = pred_df.merge(true_df, on=["index_id", "time_idx"])

    mae = mean_absolute_error(results_df["true"], results_df["predicted_median"])
    rmse = np.sqrt(mean_squared_error(results_df["true"], results_df["predicted_median"]))
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")

    results_df[["index_id", "date", "time_idx", "true", "predicted_median"]].to_csv("tft_predictions.csv", index=False)
    print("Predictions saved to tft_predictions.csv")
