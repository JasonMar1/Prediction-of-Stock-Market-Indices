from data_loader_tft import return_splits_monthly
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.utils import to_list

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping


if __name__ == "__main__":
    seed_everything(42, workers=True)

    TRAIN_START_DATE = "2006-01-01"
    TRAIN_END_DATE = "2019-11-30"

    VALID_START_DATE = "2019-12-01"
    VALID_END_DATE = "2022-08-31"

    TEST_START_DATE = "2022-10-01"  # worst case scenario, having sequence length equal to 3 months + dropping 1 month for data-leakage
    TEST_END_DATE = "2025-01-01"

    df_train, df_val, df_test, feature_columns = return_splits_monthly(TRAIN_START_DATE, TRAIN_END_DATE, VALID_START_DATE, VALID_END_DATE, TEST_START_DATE, TEST_END_DATE)

    max_encoder_length = 3
    prediction_length = 1

    train_dataset = TimeSeriesDataSet(
        data=df_train,
        time_idx="time_idx",
        target="target",
        group_ids=["index_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=prediction_length,
        static_categoricals=["index_id"],
        time_varying_known_categoricals=["month"],
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

    batch_size = None  # Set your own value
    train_loader = train_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=8, persistent_workers=True)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=8, persistent_workers=True)
    test_loader = test_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=8, persistent_workers=True)

    model = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=None  # Set your own value,
        optimizer="adam",
        dropout=None  # Set your own value,
        hidden_size=None  # Set your own value,
        attention_head_size=None  # Set your own value,
        loss=QuantileLoss(),
        output_size=7,  # Προβλεπει quantiles 0.1, 0.2, ..., 0.9
        log_interval=-1,
        log_val_interval=-1
    )

    def custom_configure_optimizers():
        optimizer = torch.optim.Adam(model.parameters(), lr=None  # Set your own value)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=None  # Set your own value,
            steps_per_epoch=len(train_loader),
            epochs=None  # Set your own value,
            pct_start=None  # Set your own value,
            anneal_strategy='cos',
            cycle_momentum=False,
            div_factor=None  # Set your own value,
            final_div_factor=None  # Set your own value
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    model.configure_optimizers = custom_configure_optimizers

    early_stop = EarlyStopping(monitor="val_loss", patience=30)
    trainer = Trainer(
        max_epochs=None  # Set your own value,
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
    )  # mode="raw" επιστρέφει όλα τα quantiles (όλο το output tensor) και μετά επιλέγω ποιο θέλω.


    if hasattr(predictions, "prediction"):
        predictions = predictions.prediction  # Extract απο το Output object το tensor με τις προβλέψεις

    predictions = to_list(predictions)
    predictions = torch.cat(predictions, dim=0)  # Merge

    # print(predictions.shape)
    # print(predictions[0, 0, :])
    # print(predictions[0, 0, 4])  # 0.5 quantile

    median = predictions[:, 0, 4].cpu().numpy()
    index_ids = x["groups"].squeeze(-1).cpu().numpy()  # [N]
    time_idxs = x["decoder_time_idx"][:, 0].cpu().numpy()  # [N]

    # build a df of median forecasts
    pred_df = pd.DataFrame({
        "index_id": index_ids,
        "time_idx": time_idxs,
        "Predicted_Log_Return": median
    })

    index_mapping = {"DJA": 0, "GSPC": 1, "IXIC": 2, "NYA": 3}
    inverse_index_mapping = {v: k for k, v in index_mapping.items()}

    # merge with the real returns
    true_df = df_test[["index_id", "time_idx", "date", "target", "Adjusted_close"]].rename(columns={"target": "Actual_Log_Return"})

    # map true_df index_id from string to int so the merge with pred_df can happen
    true_df["index_id"] = true_df["index_id"].map(index_mapping)

    pd.set_option('display.max_columns', None)
    results_df = pred_df.merge(true_df, on=["index_id", "time_idx"])

    # Invert the index mapping to get back the names of the indices
    results_df["Index"] = results_df["index_id"].map(inverse_index_mapping)
    results_df.rename(columns={"date": "Date"}, inplace=True)

    print(results_df)

    # mae, rmse only for the specific date range
    fixed_start = "2023-01-01"
    fixed_end = "2025-01-01"
    results_df = results_df[(results_df["Date"] >= fixed_start) & (results_df["Date"] <= fixed_end)]

    mae = mean_absolute_error(results_df["Actual_Log_Return"], results_df["Predicted_Log_Return"])
    rmse = np.sqrt(mean_squared_error(results_df["Actual_Log_Return"], results_df["Predicted_Log_Return"]))
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")

    results_df[["Date", "Predicted_Log_Return", "Index", "Adjusted_close"]].to_csv("monthly_tft_predictions.csv", index=False)
    print("Predictions saved to monthly_tft_predictions.csv")
