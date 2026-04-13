import sys
from datetime import datetime, timedelta
import numpy as np
from module_1_data import DataStreamer
from module_2_features import FeatureEngineer  # NEW: Importing Module 2
from module_3_model import ModelEngine
from module_4_execution import ExecutionManager


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
    coins_input = "BTC/USDT"
    selected_coins = [coin.strip().upper() for coin in coins_input.split(',')]
    timeframe = "30m"

    # Fix common formatting error gracefully
    if timeframe.isnumeric():
        timeframe += 'm'

    # ... [Skip down to where the system initializes] ...

    # 3. System Initialization
    print("\n[SYSTEM] Initializing Core Architecture...")
    try:
        # Initialize Modules
        streamer = DataStreamer(api_key, api_secret, testnet=True)
        engineer = FeatureEngineer(window_size=60)

        # --- NEW: Initialize the Execution Router ---
        executor = ExecutionManager(streamer.exchange)

        # Fetch Data
        print(f"\n[SYSTEM] Commencing Data Ingestion for {timeframe} timeframe...")
        historical_data = streamer.fetch_historical_candles(
            symbols=selected_coins,
            timeframe=timeframe,
            limit=500
        )

        for coin, df in historical_data.items():
            print(f"\n--- Processing Pipeline for {coin} ---")

            df_features = engineer.apply_technical_indicators(df)
            df_targets = engineer.engineer_target_variable(df_features)
            df_normalized = engineer.normalize_data(df_targets, is_training=True)

            X, y = engineer.create_3d_tensor(df_normalized)

            ai_engine = ModelEngine(input_size=3)
            ai_engine.train(X, y, epochs=300, batch_size=32)

            # --- LIVE PREDICTION ---
            latest_window = X[-1]
            print("\n[INFERENCE] Asking AI for next candle prediction...")
            predicted_return = ai_engine.predict_next_candle(latest_window)

            # 1. Price Math Translations
            predicted_pct = (np.exp(predicted_return) - 1) * 100
            current_price = df['close'].iloc[-1]
            predicted_target_price = current_price * np.exp(predicted_return)

            # 2. Time Math Translations
            # Extract the number and the unit from the timeframe string (e.g., "15" and "m")
            tf_val = int(timeframe[:-1])
            tf_unit = timeframe[-1].lower()

            if tf_unit == 'm':
                delta = timedelta(minutes=tf_val)
            elif tf_unit == 'h':
                delta = timedelta(hours=tf_val)
            elif tf_unit == 'd':
                delta = timedelta(days=tf_val)
            else:
                delta = timedelta(hours=1)  # Fallback

            # Calculate actual clock times
            current_time = datetime.now()
            target_time = current_time + delta

            # Format times to be easily readable (HH:MM:SS)
            time_fmt = "%H:%M:%S"
            t_now_str = current_time.strftime(time_fmt)
            t_target_str = target_time.strftime(time_fmt)

            # 3. The Prediction Dashboard
            print(f"\n" + "=" * 60)
            print(f" 🤖 AI PREDICTION REPORT: {coin} ".center(60, "="))
            print(f"=" * 60)
            print(f"Current Rate ({t_now_str})   : ${current_price:.4f}")
            print(f"Target Rate  ({t_target_str})   : ${predicted_target_price:.4f}")
            print(f"Expected Move                : {predicted_pct:+.4f}%")
            print(f"=" * 60)

            # --- THE FINAL HANDOFF TO MODULE 5 (HUMAN-IN-THE-LOOP) ---
            if predicted_pct > 0.1 or predicted_pct < -0.1:
                signal_direction = 'BUY' if predicted_pct > 0.1 else 'SELL'

                print(f"\n🚨 ACTIONABLE SIGNAL DETECTED: {signal_direction} 🚨")

                auth = input(
                    f"[AUTHORIZATION REQUIRED] Execute {signal_direction} order at ${current_price:.2f}? (y/n): ").strip().lower()

                if auth == 'y':
                    print("\n[SYSTEM] Authorization accepted. Engaging Execution Router...")
                    executor.process_signal(coin, signal_direction, current_price)
                else:
                    print("\n[SYSTEM] Authorization denied. Trade aborted. Standing down.")

            else:
                print(f"\n[ACTION] Market is flat (Move < 0.1%). HOLD. No execution required.")

    except Exception as e:
        print(f"\n[FATAL] System execution failed: {e}")

if __name__ == "__main__":
    try:
        run_cli()
    except KeyboardInterrupt:
        print("\n[SYSTEM] Process aborted by user.")
        sys.exit(0)