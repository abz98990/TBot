# **Deep Learning Spot Trading Architecture (Neural Trading Bot v3.0)**

An end-to-end, highly autonomous cryptocurrency trading system governed by a Long Short-Term Memory (LSTM) neural network. Built with **Asynchronous Parallelism**, the architecture processes multidimensional time-series data to predict the **probability of market direction** (Classification) and executes trades on the Binance Testnet with strict institutional-grade risk management.

## **🏗 System Architecture**

The project is built on the principle of **strict decoupling**. The predictive neural network possesses zero awareness of how trades are executed, and the execution engine possesses zero awareness of how predictions are made.

### **1. Data Ingestion Subsystem (`module_1_data.py`)**
The sensory input of the architecture. Connects securely to the Binance API using `ccxt` to extract raw OHLCV candlestick data. Fetches historical data asynchronously without blocking the main trading loops.

### **2. Feature Engineering Pipeline (`module_2_features.py`)**
The mathematical synthesizer. Transforms raw, non-stationary price data into a format readable by neural networks:
* **Technical Synthesis:** Calculates indicators representing Momentum (RSI), Volatility (ATR%), and Trend (EMA Distance) using `pandas-ta`.  
* **Classification Target Engineering:** Calculates binary labels (1 for UP, 0 for DOWN) to train the model on market direction.
* **Normalization:** Applies Z-Score standardization via `StandardScaler` to prevent exploding gradients.  
* **Tensor Transformation:** Employs a sliding-window algorithm to reshape 2D DataFrames into 3D Tensors (`[Samples, Time Steps, Features]`).

### **3. Deep Learning Engine (`module_3_model.py`)**
The intellectual core. Built in PyTorch, this module defines a multi-layer LSTM network. 
* **Classification Architecture:** Optimized using Binary Cross Entropy Loss (`BCEWithLogitsLoss`) and outputs a clear probability (0% to 100% confidence) via a `Sigmoid` activation function.
* **Continuous Online Learning:** The AI calibrates itself live in production, re-training for 3 epochs on the latest market data at the close of every single candle.

### **4. Execution Router (`module_4_execution.py`)**
The physical actor. Takes the pure probability forecast and translates it into rigid financial logic.
* **Probability Thresholds:** Dispatches `BUY` orders if AI confidence > 55%, and `SELL` orders if confidence < 45%.
* **Time-Based Exit Strategy:** Enforces a rigid "Single-Candle Exit". Existing positions are automatically market-closed at the end of the timeframe before the next prediction is made.
* **Capital Protection:** Automatically dispatches a protective 5% Limit Stop-Loss order into the exchange order book the moment a trade is executed.

### **5. The CLI Orchestrator (`main.py`)**
The central nervous system. Authenticates securely, handles multi-coin parallel tracking using `asyncio`, manages the Auto-Trade toggle, and maintains a strict quantitative tracking ledger (Accuracy and Cumulative PnL).

### **6. Real-Time Visualization (`dashboard.py`)**
A live-updating `matplotlib` graphical interface spawned as a subprocess. Plots Asset Price, LSTM Confidence vs Actual Outcomes, Rolling Accuracy, and Cumulative Net PnL in real-time.

## **🚀 Installation & Setup**

### **Prerequisites**
You need Python 3.8+ and dedicated **Binance Spot Testnet** API Keys. Do not use Mainnet keys.

### **1. Install Dependencies**
Ensure your environment has the required scientific computing, ML, and graphing libraries:  
```bash
pip install ccxt pandas pandas-ta scikit-learn torch numpy matplotlib
```

### **2. Offline Bootstrapping**
Before running live inference, you must initialize the neural network weights and scaler limits:
```bash
python train_offline.py
```
*Follow the prompts to fetch historical data and train the initial classification model for 150 epochs.*

### **3. Run Live Operations**
Navigate to the project directory and execute the orchestrator:  
```bash
python main.py
```

1. **Configure:** Provide a list of trading pairs (e.g., BTC/USDT, ETH/USDT) and a timeframe (e.g., 15m, 1h).  
2. **Dashboard:** Elect to launch the real-time visualization tracker.
3. **Execution Mode:** Choose to manually authorize trades with a 10s timeout, or engage the **Auto-Trade** toggle for fully autonomous trading.
4. **Hibernate & Adapt:** The bot will monitor multiple coins in parallel, trade, exit at candle close, recalibrate its brain on the new data, and hibernate until the next cycle.

## **⚠️ Disclaimer**
This software is for educational and research purposes only. Cryptocurrency markets are highly volatile and adversarial. The 5% Stop Loss mitigates risk but does not guarantee absolute loss prevention due to exchange slippage. Always test strategies thoroughly on the Testnet before deploying real capital.