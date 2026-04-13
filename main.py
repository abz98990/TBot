# main.py
import getpass
import sys
from module_1_data import DataStreamer
from module_2_features import FeatureEngineer  # NEW: Importing Module 2
from module_3_model import ModelEngine


def run_cli():
    print("=" * 50)
    print(" NEURAL TRADING BOT CLI v1.1 ".center(50, "="))
    print("=" * 50)

    # 1. Secure Credential Input
    print("\n--- Exchange Authentication ---")
    api_key = "CYDdL2sD4wsBy1g1mte1OieivnbBpuxwN63s0RoyYtxRLHjffGabECjvXmBcYacW"
    api_secret = "MSQRV7BnrVv28bJ6DkxtckXSpu8jkqZ38XuG8ASUjoueoMaAKJ7y31OqhggTV6NG"

    if not api_key or not api_secret:
        print("[FATAL] Credentials cannot be empty. Exiting.")
        sys.exit(1)

    # 2. Coin Selection
    print("\n--- Strategy Parameters ---")
    coins_input = input("Enter coins to trade (comma separated, max 5): ")
    selected_coins = [coin.strip().upper() for coin in coins_input.split(',')]
    timeframe = input("Enter timeframe (e.g., 15m, 1h, 1d) [Default: 1h]: ").strip() or "1h"

    # Fix common formatting error gracefully
    if timeframe.isnumeric():
        timeframe += 'm'

    # 3. System Initialization
    print("\n[SYSTEM] Initializing Core Architecture...")
    try:
        # Initialize Modules
        streamer = DataStreamer(api_key, api_secret, testnet=True)
        engineer = FeatureEngineer(window_size=60)  # Looking back 60 candles

        # Fetch Data
        print(f"\n[SYSTEM] Commencing Data Ingestion for {timeframe} timeframe...")
        historical_data = streamer.fetch_historical_candles(
            symbols=selected_coins,
            timeframe=timeframe,
            limit=500
        )

        # 4. Feature Engineering Pipeline (NEW)
        for coin, df in historical_data.items():
            print(f"\n--- Processing Pipeline for {coin} ---")

            # Run the synthesis pipeline
            df_features = engineer.apply_technical_indicators(df)
            df_targets = engineer.engineer_target_variable(df_features)
            df_normalized = engineer.normalize_data(df_targets, is_training=True)

            # Generate the final LSTM inputs
            X, y = engineer.create_3d_tensor(df_normalized)

            # ... inside the historical_data loop ...

            # Generate the final LSTM inputs
            X, y = engineer.create_3d_tensor(df_normalized)

            # --- NEW CODE: MODULE 3 INTEGRATION ---
            # Initialize the Brain (input_size=3 because we have 3 features)
            ai_engine = ModelEngine(input_size=3)

            # Train the bot!
            ai_engine.train(X, y, epochs=15, batch_size=16)

        # ... rest of file ...
    except Exception as e:
        print(f"\n[FATAL] System execution failed: {e}")


if __name__ == "__main__":
    try:
        run_cli()
    except KeyboardInterrupt:
        print("\n[SYSTEM] Process aborted by user.")
        sys.exit(0)