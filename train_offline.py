import os
import sys

from module_1_data import DataStreamer
from module_2_features import FeatureEngineer
from module_3_model import ModelEngine
from main import load_or_prompt_credentials

def run_offline_training():
    print("=" * 60)
    print(" NEURAL TRADING BOT - OFFLINE TRAINING PHASE ".center(60, "="))
    print("=" * 60)

    # 1. Secure Credential Input
    api_key, api_secret = load_or_prompt_credentials()

    # 2. Coin Selection
    print("\n--- Training Parameters ---")
    symbols = input("Enter coins to train on (comma separated, e.g. BTC/USDT): ").strip().upper().split(',')
    symbols = [s.strip() for s in symbols if s.strip()]
    if not symbols:
        symbols = ['BTC/USDT']
    
    timeframe = input("Enter timeframe (e.g., 15m, 1h, 1d) [Default: 1h]: ").strip() or "1h"
    if timeframe.isnumeric():
        timeframe += 'm'

    # 3. Initialization
    print("\n[SYSTEM] Initializing Components...")
    streamer = DataStreamer(api_key, api_secret, testnet=True)
    engineer = FeatureEngineer(window_size=60)
    
    # We fetch a larger dataset for offline training.
    print("\n[SYSTEM] Commencing Bulk Data Fetch (Can take several minutes)...")
    limit = 1000 # Could be expanded using pagination logic to fetch 10s of thousands
    
    historical_data = streamer.fetch_historical_candles(
        symbols=symbols,
        timeframe=timeframe,
        limit=limit
    )

    for coin, df in historical_data.items():
        print(f"\n--- Training Engine on {coin} ---")
        if df.empty:
            continue

        # Feature Engineering Pipeline
        df_features = engineer.apply_technical_indicators(df)
        df_targets = engineer.engineer_target_variable(df_features)
        
        # We explicitly set is_training=True so it FITS the StandardScaler
        df_normalized = engineer.normalize_data(df_targets, is_training=True)
        X, y = engineer.create_3d_tensor(df_normalized)

        # Build Model
        ai_engine = ModelEngine(input_size=3)
        
        # Train aggressively
        ai_engine.train(X, y, epochs=150, batch_size=32)

        # Define filepaths dynamically based on coin name (replaces / with _)
        coin_clean = coin.replace('/', '_')
        model_filepath = os.path.join("models", f"{coin_clean}_lstm_weights.pth")
        scaler_filepath = os.path.join("models", f"{coin_clean}_scaler.pkl")
        
        # Securely save the weights and scaler for live trading
        ai_engine.save_model(filepath=model_filepath)
        engineer.save_scaler(filepath=scaler_filepath)

if __name__ == "__main__":
    try:
        run_offline_training()
    except KeyboardInterrupt:
        print("\n\n[SYSTEM] Process manually aborted by user.")
        sys.exit(0)
