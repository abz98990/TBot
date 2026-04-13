# **Deep Learning Spot Trading Architecture (Neural Trading Bot)**

An end-to-end, strictly decoupled cryptocurrency trading system governed by a Long Short-Term Memory (LSTM) neural network. Designed to predict rolling log returns by processing multidimensional, stationary time-series tensors, and execute trades on the Binance Spot Testnet with strict, institutional-grade risk management.

## **🏗 System Architecture**

The project is built on the principle of **strict decoupling**. The predictive neural network possesses zero awareness of how trades are executed, and the execution engine possesses zero awareness of how predictions are made.

### **1\. Data Ingestion Subsystem (module\_1\_data.py)**

The sensory input of the architecture. Connects securely to the Binance API using ccxt to extract raw OHLCV (Open, High, Low, Close, Volume) candlestick data across dynamically selected timeframes.

### **2\. Feature Engineering Pipeline (module\_2\_features.py)**

The mathematical synthesizer. Transforms raw, non-stationary price data into a format readable by neural networks:

* **Technical Synthesis:** Calculates a lean diet of indicators representing Momentum (RSI), Volatility (ATR%), and Trend (EMA Distance) using pandas-ta.  
* **Target Engineering:** Calculates forward log returns.  
* **Normalization:** Applies Z-Score scaling to prevent exploding gradients.  
* **Tensor Transformation:** Employs a sliding-window algorithm to reshape 2D DataFrames into 3D Tensors (\[Samples, Time Steps, Features\]).

### **3\. Deep Learning Engine (module\_3\_model.py)**

The intellectual core. Built in PyTorch, this module defines and trains a multi-layer LSTM network. It processes historical 3D tensors, mitigates overfitting via Dropout layers, and outputs a continuous numerical prediction (Mean Squared Error optimized) representing the expected log return of the next candle.

### **4\. Execution Router (module\_5\_execution.py)**

The physical actor. Takes the pure mathematical forecast and translates it into rigid financial logic.

* **Risk Management:** Enforces a strict 5% Stop Loss (SL) and tiered Take Profit (TP) targets at 20%, 25%, and 30%.  
* **Circuit Breaker:** Implements a 3-strike kill switch. If 3 consecutive trades fail, the system halts to protect capital.  
* **Order Routing:** Formats, signs, and dispatches API payloads to the Binance Testnet.

### **5\. The CLI Orchestrator (main.py)**

The central nervous system. Authenticates the user securely (hiding API secrets), prompts for operational parameters (coins, timeframe), coordinates the handoff between all modules, and presents the **Human-in-the-Loop** prediction dashboard.

## **🚀 Installation & Setup**

### **Prerequisites**

You need Python 3.8+ and dedicated **Binance Spot Testnet** API Keys. Do not use Mainnet keys.

### **1\. Install Dependencies**

Ensure your environment has the required scientific computing and exchange libraries:  
pip install ccxt pandas pandas-ta scikit-learn torch numpy

### **2\. Run the System**

Navigate to the project directory and execute the orchestrator:  
python main.py

### **3\. Operational Flow**

1. **Authenticate:** Enter your Binance Testnet API Key and Secret.  
2. **Configure:** Provide a list of trading pairs (e.g., BTC/USDT) and a timeframe (e.g., 15m, 1h).  
3. **Train:** The system will ingest historical data, normalize it, and train the LSTM live in your terminal over 15 epochs.  
4. **Authorize:** The AI will present a Prediction Dashboard showing the current rate, target rate, and expected percentage move. **You** maintain ultimate sovereignty and must type y to green-light the live trade execution.

## **⚠️ Disclaimer**

This software is for educational and research purposes only. Cryptocurrency markets are highly volatile and adversarial. The 5% Stop Loss mitigates risk but does not guarantee absolute loss prevention due to exchange slippage. Always test strategies thoroughly on the Testnet before deploying real capital.