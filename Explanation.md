# T Bot: Mathematical and Structural Architecture

This document provides a technical overview of the structure and mathematics powering **T Bot**, a neural trading system.

## 1. Structural Architecture

The project has been refactored into a highly modular, decoupled architecture following clean separation of concerns. This allows components like data fetching and live execution to operate independently of heavy model training.

### 1.1 Core Modules
*   **`module_1_data.py` (Data Streamer):** Handles secure connections and data acquisition via `ccxt` (Binance). Standardizes raw exchange data into analyzable Pandas DataFrames and respects API rate limits to prevent IP blocks.
*   **`module_2_features.py` (Feature Engineer):** The data synthesis pipeline. Responsible for technical indicator calculation, target variable generation, temporal window sliding (batch generation), and critical scaling logic. Maintains strict train/test data separation to avoid data leakage.
*   **`module_3_model.py` (Model Engine):** Contains the PyTorch neural network definition (LSTM). Operates the training loop, loss calculation, backpropagation, and state-dict loading/saving.
*   **`module_4_execution.py` (Execution Manager):** Translates AI output into actionable real-world outcomes. Includes predefined risk-management logic (Stop-Loss/Take-Profit orders) and a "Circuit Breaker" to halt operations after multiple execution failures.

### 1.2 Execution Pipelines
*   **`main.py` (Asynchronous Live Inference):** The production loop. Employs `asyncio` to track multiple coins concurrently without blocking. It avoids live training; instead, it dynamically loads pre-computed normalizer scalers and pre-trained neural network weights (`.pth` files), processing minimal data through the model to achieve fast execution latency.
*   **`train_offline.py` (Historical Training Workflow):** An offline standalone script built for data-hogging. It fetches vast amounts of data, processes the intensive feature engineering, and brute-force trains the network without worrying about live market latency, exporting its learned states.

---

## 2. Mathematical and Statistical Aspects

At its core, the bot casts market forecasting as an autoregressive continuous regression problem.

### 2.1 Feature Engineering (Inputs)
The model looks at 3 stationary technical indicators per given candle to achieve size-independent, comparative analysis across all price levels:
*   **Momentum (RSI):** Bounded $0 \to 100$. A 14-period Relative Strength Index.
*   **Volatility (Normalized ATR):** Rather than absolute Price ATR, it is normalized: $\frac{ATR_{14}}{Close\_Price}$.
*   **Trend (EMA Distance):** Prevents raw value bias by measuring percentage deviation rather than absolute difference: $\frac{Close\_Price - EMA_{50}}{EMA_{50}}$.

### 2.2 Target Variable (Ground Truth)
The network does not predict raw asset prices (which are non-stationary and cause explosive gradients). Instead, it predicts **Forward Log Returns**, defined mathematically as:
$$y_{t} = \ln{\left(\frac{Close_{t+1}}{Close_{t}}\right)}$$

When evaluating live, the model unpacks the predicted return to find its percentage move and an absolute price target.
$$Percentage\_Move = (e^{y\_pred} - 1) \times 100$$
$$Target\_Price = Current\_Price \times e^{y\_pred}$$

### 2.3 Data Transformation & Tensors
*   **Z-Score Standardization:** The 3 chosen features go through `StandardScaler` to force a mean of 0 and standard deviation of 1.
    *   *Critical Detail:* During live inference (`is_training=False`), the live data is standardized only using the parameters fitted during offline training, avoiding lookahead bias and data leakage.
*   **Sliding Window (3D Tensor):** The data undergoes dimensional reshaping for the LSTM. Using a sliding window size of $N=60$ periods, the data becomes a 3-Dimensional tensor of shape:
    `(Samples, Time Steps (60), Features (3))`
    This gives the LSTM a sequential "memory" of the last 60 moments up to the target.

### 2.4 Model Topology and Mathematics
The bot uses a PyTorch-based Long Short-Term Memory (LSTM) network capable of capturing long dependencies in time-series data without suffering vanishing gradients:
*   **Deep LSTM Layer:** 2 stacked layers with a hidden dimension size of 64 nodes to compress temporal patterns.
*   **Dropout:** Configured to drop $20\%$ of nodes per forward pass during training to prevent the network from memorizing noise (overfitting).
*   **Dense Linear Layer:** Compresses the 64 hidden factors from the *final* sequence step into 1 regressed float (the predicted Log Return).
*   **Loss Function & Optimization:** Utilizes **Mean Squared Error (MSE)** to massively penalize large predictive mistakes, optimized by the generic adaptive gradient descent algorithm, **Adam**, heavily tuned with a $0.001$ learning rate.

### 2.5 Structured Risk Mathematics
Once a decision passes a low-volatility threshold (Expected move $> 0.01\%$), `module_4_execution.py` mathematically locks the trade:
1.  **Sizing:** Dynamically offsets fixed Binance limits by enforcing ~$15 block positions ($15 / Current Price).
2.  **Risk/Reward Limits:** Hard-codes rigid limits away from the entry price for `BUY` positions (inverted for `SELL`):
    *   Stop Loss (Damage Control): $5\%$ below entry $(-0.05)$
    *   Take Profit Layers: Scaled linearly at $+20\%, +25\%, +30\%$.
