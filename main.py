import os
import json
import getpass
import sys
import time
import numpy as np
from datetime import datetime, timedelta

from module_1_data import DataStreamer
from module_2_features import FeatureEngineer
from module_3_model import ModelEngine
from module_4_execution import ExecutionManager

CREDENTIALS_FILE = os.path.join("config", "api_keys.json")


def load_or_prompt_credentials():
    """Auto-reads credentials from a file, or prompts and saves them for future use."""
    # 1. Try to auto-read the keys
    if os.path.exists(CREDENTIALS_FILE):
        try:
            with open(CREDENTIALS_FILE, 'r') as f:
                keys = json.load(f)
                print(f"[SYSTEM] Loaded API credentials automatically from {CREDENTIALS_FILE}")
                return keys.get("api_key", ""), keys.get("api_secret", "")
        except Exception as e:
            print(f"[ERROR] Failed to read {CREDENTIALS_FILE}: {e}")

    # 2. Fallback to manual input if file doesn't exist
    print("\n--- Exchange Authentication ---")
    api_key = input("Enter Binance API Key: ").strip()
    api_secret = getpass.getpass("Enter Binance API Secret (Hidden): ").strip()

    if not api_key or not api_secret:
        print("[FATAL] Credentials cannot be empty. Exiting.")
        sys.exit(1)

    # 3. Auto-save the keys for future iterations
    try:
        os.makedirs("config", exist_ok=True)
        with open(CREDENTIALS_FILE, 'w') as f:
            json.dump({"api_key": api_key, "api_secret": api_secret}, f, indent=4)
        print(f"[SYSTEM] Credentials securely saved to {CREDENTIALS_FILE} for future use.")
    except Exception as e:
        print(f"[WARNING] Could not save credentials: {e}")

    return api_key, api_secret


def run_cli():
    print("=" * 60)
    print(" NEURAL TRADING BOT CLI v2.0 (LIVE INFERENCE) ".center(60, "="))
    print("=" * 60)

    # 1. Secure Credential Input (Auto-Read Mechanism)
    api_key, api_secret = load_or_prompt_credentials()

    # 2. Coin Selection
    print("\n--- Strategy Parameters ---")
    coins_input = input("Enter coins to trade (comma separated, max 5. e.g. BTC/USDT): ")
    selected_coins = [coin.strip().upper() for coin in coins_input.split(',')]
    timeframe = input("Enter timeframe (e.g., 15m, 1h, 1d) [Default: 1h]: ").strip() or "1h"

    if timeframe.isnumeric():
        timeframe += 'm'

    # 3. System Initialization
    print("\n[SYSTEM] Initializing Core Architecture...")
    try:
        streamer = DataStreamer(api_key, api_secret, testnet=True)
        engineer = FeatureEngineer(window_size=60)
        executor = ExecutionManager(streamer.exchange)

        # Determine Sleep Duration Based on Timeframe
        tf_val = int(timeframe[:-1])
        tf_unit = timeframe[-1].lower()
        if tf_unit == 'm':
            sleep_seconds = tf_val * 60
        elif tf_unit == 'h':
            sleep_seconds = tf_val * 3600
        elif tf_unit == 'd':
            sleep_seconds = tf_val * 86400
        else:
            sleep_seconds = 3600  # Fallback 1 hour

        print(f"\n[SYSTEM] ENTERING AUTONOMOUS LIVE INFERENCE PHASE.")
        print(f"[SYSTEM] Bot will cycle and refresh data every {timeframe}.")

        # ==============================================================
        # THE CONTINUOUS LIVE LOOP
        # ==============================================================
        while True:
            try:
                cycle_time = datetime.now().strftime('%H:%M:%S')
                print(f"\n{'=' * 60}")
                print(f" INFERENCE CYCLE INITIATED AT {cycle_time} ".center(60, "="))
                print(f"{'=' * 60}")

                # Fetch the absolute latest market data
                historical_data = streamer.fetch_historical_candles(
                    symbols=selected_coins,
                    timeframe=timeframe,
                    limit=500
                )

                # Process each coin
                for coin, df in historical_data.items():
                    print(f"\n--- Processing Pipeline for {coin} ---")

                    # 1. Synthesize & Normalize
                    df_features = engineer.apply_technical_indicators(df)
                    df_targets = engineer.engineer_target_variable(df_features)
                    df_normalized = engineer.normalize_data(df_targets, is_training=True)
                    X, y = engineer.create_3d_tensor(df_normalized)

                    # 2. Rolling Window Retraining
                    # We train inside the loop so the model dynamically adapts to live market shifts
                    ai_engine = ModelEngine(input_size=3)
                    ai_engine.train(X, y, epochs=15, batch_size=16)

                    # 3. Live Inference
                    latest_window = X[-1]
                    print("\n[INFERENCE] Asking AI for next candle prediction...")
                    predicted_return = ai_engine.predict_next_candle(latest_window)

                    # Math Translations
                    predicted_pct = (np.exp(predicted_return) - 1) * 100
                    current_price = df['close'].iloc[-1]
                    predicted_target_price = current_price * np.exp(predicted_return)

                    # Time Translations
                    current_time = datetime.now()
                    target_time = current_time + timedelta(seconds=sleep_seconds)
                    t_now_str = current_time.strftime("%H:%M:%S")
                    t_target_str = target_time.strftime("%H:%M:%S")

                    # The Prediction Dashboard
                    print(f"\n" + "=" * 60)
                    print(f" 🤖 AI PREDICTION REPORT: {coin} ".center(60, "="))
                    print(f"=" * 60)
                    print(f"Current Rate ({t_now_str})   : ${current_price:.4f}")
                    print(f"Target Rate  ({t_target_str})   : ${predicted_target_price:.4f}")
                    print(f"Expected Move                : {predicted_pct:+.4f}%")
                    print(f"=" * 60)

                    # 4. Human-in-the-Loop Execution Router
                    if predicted_pct > 0.1 or predicted_pct < -0.1:
                        signal_direction = 'BUY' if predicted_pct > 0.1 else 'SELL'
                        print(f"\n🚨 ACTIONABLE SIGNAL DETECTED: {signal_direction} 🚨")

                        auth = input(
                            f"[AUTHORIZATION REQUIRED] Execute {signal_direction} order at ${current_price:.4f}? (y/n): ").strip().lower()

                        if auth == 'y':
                            print("\n[SYSTEM] Authorization accepted. Engaging Execution Router...")
                            executor.process_signal(coin, signal_direction, current_price)
                        else:
                            print("\n[SYSTEM] Authorization denied. Trade aborted. Standing down.")

                    else:
                        print(f"\n[ACTION] Market is flat (Move < 0.1%). HOLD. No execution required.")

                # Sleep sequence to wait for the next candle to form
                print(f"\n[SYSTEM] Cycle Complete. Entering hibernation for {timeframe}...")
                time.sleep(sleep_seconds)

            except Exception as loop_error:
                # If a network timeout or error occurs, don't crash. Sleep for 60 seconds and try again.
                print(f"\n[ERROR] Live Loop encountered a critical disruption: {loop_error}")
                print("[SYSTEM] Initiating 60-second recovery sleep before attempting next cycle...")
                time.sleep(60)

    except Exception as e:
        print(f"\n[FATAL] System execution failed during initialization: {e}")


if __name__ == "__main__":
    try:
        run_cli()
    except KeyboardInterrupt:
        # Graceful exit if you press Ctrl+C
        print("\n\n[SYSTEM] Process manually aborted by user. Shutting down Live Loop.")
        sys.exit(0)