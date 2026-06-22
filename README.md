# Prediction of Stock Market Indices Using Machine Learning and Deep Learning

A comprehensive research project developed as part of my bachelor's thesis for forecasting stock market index returns using traditional machine learning, recurrent neural networks, probabilistic forecasting models, transformer-based architectures, and custom deep learning approaches.

---

## Overview

This repository contains the implementation and experimental framework developed for my bachelor's thesis focused on forecasting the future returns of major U.S. stock market indices.

The study investigates whether machine learning and deep learning models can identify predictive patterns in historical financial data and technical indicators. In addition to forecasting performance, model predictions are evaluated through portfolio backtesting to assess their practical usefulness in investment decision-making.

The research covers two forecasting horizons:

- **Daily Forecasting** (next trading day)
- **Monthly Forecasting** (next month)

The project combines:

- Financial feature engineering
- Traditional machine learning algorithms
- Recurrent neural network architectures
- Transformer-based forecasting models
- Probabilistic forecasting methods
- Hyperparameter optimization
- Portfolio backtesting and risk analysis

---

## Research Objectives

The thesis aims to investigate the following research questions:

- Can machine learning algorithms effectively predict future stock index returns?
- Do deep learning models outperform traditional machine learning approaches?
- Does training a single model across multiple indices improve forecasting performance?
- Can attention mechanisms capture relationships between financial markets?
- Do improvements in forecasting accuracy translate into superior portfolio performance?
- Which forecasting horizon is more predictable: daily or monthly?

---

## Stock Market Indices

The study focuses on four major U.S. stock market indices:

| Symbol | Index |
|----------|----------|
| DJA | Dow Jones Industrial Average |
| GSPC | S&P 500 |
| IXIC | NASDAQ Composite |
| NYA | NYSE Composite |

Historical OHLCV data is stored locally and spans multiple decades of market activity.

---

## Target Variable

For all experiments, the prediction target is the next-period logarithmic return:

```
Log_Return_t = log(Pt) - log(Pt-1)
```
Where Pt denotes the adjusted closing price

Daily models predict next-day returns, while monthly models predict next-month returns.

---

## Feature Engineering

A comprehensive feature engineering pipeline was developed separately for daily and monthly forecasting.

### Market Features

- Open
- High
- Low
- Close
- Adjusted Close
- Volume

### Return Features

#### Daily Forecasting

- Log_Return_1
- Log_Return_5
- Log_Return_10
- Log_Return_20

#### Monthly Forecasting

- Log_Return_1
- Log_Return_2
- Log_Return_4
- Log_Return_6

### Technical Indicators

- Relative Strength Index (RSI)
- Simple Moving Averages (SMA)
- Exponential Moving Averages (EMA)
- Moving Average Crossovers
- Rolling Volatility

### Temporal Features

Used by DeepAR and Temporal Fusion Transformer:

- Day of Week
- Month
- Time Index

### Data Scaling

Continuous variables are standardized using statistics, calculated exclusively from the training dataset to prevent data leakage.

---

## Implemented Models

### Traditional Machine Learning Models

#### Linear Regression
#### Support Vector Regression (SVR)
#### Random Forest Regression

---

### Deep Learning Models

#### LSTM

A standard Long Short-Term Memory network used as the primary recurrent neural network baseline.

#### Long LSTM

A shared LSTM model trained on multiple stock market indices, where each index is treated as an independent time series.

Unlike the Wide LSTM, each index retains its own separate feature sequence. However, all sequences are used jointly to train a single shared model, enabling parameter sharing across different markets.

This allows the model to learn generalized temporal patterns that can transfer between indices while still producing index-specific predictions for each series.

#### Wide LSTM

A multi-index forecasting setup where a single LSTM model is trained using synchronized feature vectors from multiple stock market indices (DJA, GSPC, IXIC, NYA).

At each time step, the input consists of concatenated features from all indices, allowing the model to capture potential correlations between markets.

The model produces four simultaneous outputs, one per index, corresponding to the next-day/month log return prediction for each market.

#### Conditional LSTM

An enhanced multi-index LSTM architecture that extends the Long LSTM by introducing index-aware conditioning through learned embeddings.

Each stock market index (DJA, GSPC, IXIC, NYA) is mapped to a unique embedding vector, which is learned during training.

Unlike standard LSTM models that initialize hidden and cell states with zeros, the Conditional LSTM uses the index embedding to initialize the hidden and cell states.

This allows the model to adapt its internal dynamics based on the specific market being processed, enabling it to learn both shared temporal patterns and index-specific behavior.

---

### Advanced Time-Series Deep Learning Models

#### DeepAR

A probabilistic forecasting model implemented using PyTorch Forecasting.

Features:

- Autoregressive forecasting
- Probabilistic predictions
- Multi-series learning
- Uncertainty estimation
- Support for temporal covariates

#### Temporal Fusion Transformer (TFT)

A state-of-the-art attention-based architecture designed specifically for time-series forecasting.

Features:

- Variable Selection Networks
- Multi-Head Attention
- Temporal Feature Fusion
- Interpretable Forecasting Mechanisms
- Multi-series forecasting capability

---

### Custom Layer Sharing LSTM

One of the primary contributions of this thesis is the development of a custom **Layer Sharing LSTM** architecture.

The model combines:

- Shared LSTM layers across multiple indices
- Learned index embeddings
- Index-conditioned hidden state initialization
- Multi-head self-attention
- Shared representation learning

The architecture is designed to capture both:

- Common market dynamics
- Index-specific characteristics

while exploiting relationships between multiple financial markets simultaneously.

---

## Hyperparameter Optimization

Deep learning models are optimized using **Optuna**.

The optimization process explores parameters such as:

- Hidden Size
- Number of Layers
- Dropout Rate
- Learning Rate
- Batch Size
- Sequence Length
- Number of Attention Heads
- Training Epochs

Learning rate scheduling is performed using **OneCycleLR**.

> **Note on Hyperparameter Values:**
> 
> The final hyperparameter configurations for all models have been intentionally omitted from this repository and are initialized as `None` throughout the codebase.
> Arriving at the final configuration required extensive experimentation, iterative tuning, and significant compute time as part of a formal academic research process. Sharing them publicly would undermine the research effort behind this work.
> If you wish to reproduce or extend this research, hyperparameter optimization can be performed using the provided Optuna scripts.
---

## Data Pipeline

The forecasting workflow follows a strict chronological process to eliminate look-ahead bias.

```text
Raw Historical Data
        │
        ▼
Feature Engineering
        │
        ▼
Train / Validation / Test Split
        │
        ▼
Feature Scaling
        │
        ▼
Model Training
        │
        ▼
Prediction Generation
        │
        ▼
Backtesting
        │
        ▼
Performance Evaluation
```

**All datasets are split chronologically rather than randomly to preserve temporal structure.**

---

## Backtesting Framework

Forecasts are transformed into investment decisions and evaluated using a portfolio simulation framework.

Three portfolio management strategies are implemented:

### Full Rebalancing

Portfolio weights are completely adjusted at each rebalance period according to model predictions.

### Fixed Percentage

A fixed allocation percentage is invested based on the generated forecasts.

### Hybrid Strategy

A combination of Full Rebalancing and Fixed Percentage approaches designed to balance responsiveness and transaction efficiency.

---

## Evaluation Metrics

### Forecasting Metrics

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Directional Accuracy

### Financial Performance Metrics

- Expected Return
- Volatility
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Final Portfolio Value

Models are evaluated not only on forecasting accuracy but also on their ability to generate profitable and risk-adjusted investment strategies.

---

## Key Conclusions

The experimental study revealed that:

- **Deep learning generally outperforms classical ML** on both MAE and RMSE, though Random Forest remains competitive, particularly at the monthly horizon.
- **Multi-index architectures improve generalization.** The Conditional LSTM, which conditions hidden-state initialization on learned index embeddings, consistently produced strong and balanced results across both forecasting horizons.
- **Full Rebalancing underperforms** in all scenarios due to high cumulative transaction costs that erode returns regardless of prediction quality.
- **Best daily combination: Hybrid strategy + Basic LSTM** — highest Expected Return (20.73%), Sharpe Ratio (1.64), and Sortino Ratio (1.60) with an annual portfolio return of 22.10%.
- **Best monthly combination: Hybrid strategy + Conditional LSTM** — best balance of return (16.94% expected, 16.96% annual) and risk control (Sharpe 1.61, Sortino 2.01, Max Drawdown 6.86%). For risk-averse investors, **Fixed Percentage + Conditional LSTM** achieved the lowest Volatility (7.45%), lowest Max Drawdown (4.47%), and highest Sortino Ratio (2.23).
- **Monthly forecasting is more stable** than daily forecasting, producing lower relative errors and superior risk-adjusted metrics in portfolio simulations.
- **Forecasting accuracy alone is an incomplete proxy for investment performance.** Several models with higher MAE/RMSE still outperformed lower-error models in backtesting, underscoring the importance of direction accuracy and strategy design.
---

## Repository Structure

```text
├── index_data/
│   ├── DJA.INDX.csv
│   ├── GSPC.INDX.csv
│   ├── IXIC.INDX.csv
│   ├── NYA.INDX.csv
│
├── daily models/
│   ├── Daily Optuna/
│   ├── Daily Predictions/
│   └── backtesting/
├── monthly models/
│   ├── Monthly Optuna/
│   ├── Monthly Predictions/
│   └── backtesting/
├── data_loader.py
├── data_loader_long.py
├── data_loader_wide.py
├── data_loader_conditional.py
├── data_loader_layer_sharing.py
├── data_loader_DeepAR.py
└── data_loader_tft.py
```

---

## Key Contributions

- Comparative analysis of traditional machine learning and deep learning models for stock market forecasting.
- Daily and monthly forecasting framework.
- Extensive financial feature engineering pipeline.
- Multi-index forecasting through Conditional LSTM.
- Development of a custom Layer Sharing LSTM architecture with self-attention.
- Hyperparameter optimization using Optuna.
- Comprehensive portfolio backtesting framework.
- Evaluation based on both predictive performance and investment outcomes.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-Learn
- PyTorch
- PyTorch Forecasting
- Optuna
- Matplotlib

---

## Disclaimer

This repository is intended exclusively for academic and research purposes.

The implemented forecasting models are experimental and should not be interpreted as financial advice. Historical performance does not guarantee future investment results.

---

## Author

**Jason Marinopoulos**

Bachelor Thesis – Computer Science Department

Aristotle University of Thessaloniki

GitHub: https://github.com/JasonMar1
