import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss
from sklearn.model_selection import train_test_split


from data_loader_tft import preprocess_tft_data

seed_everything(42)


MAX_ENCODER_LENGTH = 60
MAX_PREDICTION_LENGTH = 5
BATCH_SIZE = 64
EPOCHS = 30


df = preprocess_tft_data(
    start_date="2005-01-01",
    end_date="2023-12-31",
    prediction_shift=1
)

# Split data by time
cutoff_train = "2020-01-01"
cutoff_val = "2022-01-01"


train_df = df[df.date < cutoff_train]
val_df = df[(df.date >= cutoff_train) & (df.date < cutoff_val)]
test_df = df[df.date >= cutoff_val]


train_data = TimeSeriesDataSet(
    train_df,
    time_idx="time_idx",
    target="value",
    group_ids=["stock_id"],

    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=MAX_PREDICTION_LENGTH,

    static_categoricals=["stock_id"],
    time_varying_known_categoricals=["day_of_the_week", "month"],
    time_varying_unknown_reals=["target", "Log_Returns_5", "Log_Returns_10", "Log_Returns_20",
        "Volatility", "RSI_14", "SMA_5", "SMA_20", "SMA_50",
        "EMA_10", "EMA_50", "MA_Crossover"],

    target_normalizer=NaNLabelEncoder(),

    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True
)

validation_data = TimeSeriesDataSet(
    val_df,
    time_idx="time_idx",
    target="target",
    group_ids=["stock_id"],

    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=MAX_PREDICTION_LENGTH,

    static_categoricals=["stock_id"],
    time_varying_known_categoricals=["day_of_the_week", "month"],
    time_varying_unknown_reals=["target", "Log_Returns_5", "Log_Returns_10", "Log_Returns_20",
        "Volatility", "RSI_14", "SMA_5", "SMA_20", "SMA_50",
        "EMA_10", "EMA_50", "MA_Crossover"],

    target_normalizer=NaNLabelEncoder(),

    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True
)

test_data = TimeSeriesDataSet(
    test_df,
    time_idx="time_idx",
    target="target",
    group_ids=["stock_id"],

    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=MAX_PREDICTION_LENGTH,

    static_categoricals=["stock_id"],
    time_varying_known_categoricals=["day_of_the_week", "month"],
    time_varying_unknown_reals=["target", "Log_Returns_5", "Log_Returns_10", "Log_Returns_20",
                                "Volatility", "RSI_14", "SMA_5", "SMA_20", "SMA_50",
                                "EMA_10", "EMA_50", "MA_Crossover"],

    target_normalizer=NaNLabelEncoder(),

    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True
)

train_loader = train_data.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
val_loader = validation_data.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
test_loader = test_data.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)


tft = TemporalFusionTransformer.from_dataset(
    train_data,
    learning_rate=0.01,
    hidden_size=16,
    attention_heads=1,
    dropout=0.1,
    loss=QuantileLoss(),
    output_size=7,  # number of quantiles: 0.1 to 0.9
    log_interval=10,
    # reduce_on_plateau_patience=4
)


trainer = Trainer(
    max_epochs=EPOCHS,
    gradient_clip_val=0.1,
    accelerator="auto"
)

trainer.fit(
    model=tft,
    train_dataloader=train_loader,
    val_dataloaders=val_loader
)

tft.save_model("tft_model.ckpt")

best_model = TemporalFusionTransformer.load_from_checkpoint("tft_model.ckpt")

predictions, x = best_model.predict(
    test_loader,
    return_x=True,
    return_index=True
)

pred_df = predictions.detach().cpu().numpy()
index_df = x["decoder_time_idx"]
result = pd.DataFrame(
    pred_df[:, :, 3],  # 0.5 quantile (median)
    columns=[f"t+{i+1}" for i in range(pred_df.shape[1])]
)
result["decoder_time_idx"] = index_df[:, 0].detach().cpu().numpy()
result.to_csv("tft_predictions.csv", index=False)

print("Model trained and predictions saved.")