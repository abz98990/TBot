import os
import json
import sys
import time
import asyncio
import numpy as np
import msvcrt
from datetime import datetime, timedelta

from module_1_data import DataStreamer
from module_2_features import FeatureEngineer
from module_3_model import ModelEngine
from module_4_execution import ExecutionManager

CREDENTIALS_FILE = os.path.join("config", "api_keys.json")


def load_or_prompt_credentials():
    """Auto-reads credentials from a file, or prompts and saves them for future use."""
    if os.path.exists(CREDENTIALS_FILE):
        try:
            with open(CREDENTIALS_FILE, 'r') as f:
                keys = json.load(f)
                print(f"[SYSTEM] Loaded API credentials automatically from {CREDENTIALS_FILE}")
                return keys.get("api_key", ""), keys.get("api_secret", "")
        except Exception as e:
            print(f"[ERROR] Failed to read {CREDENTIALS_FILE}: {e}")

    print("\n--- Exchange Authentication ---")
    api_key = "CYDdL2sD4wsBy1g1mte1OieivnbBpuxwN63s0RoyYtxRLHjffGabECjvXmBcYacW" # Demo Key
    api_secret = "MSQRV7BnrVv28bJ6DkxtckXSpu8jkqZ38XuG8ASUjoueoMaAKJ7y31OqhggTV6NG"

    if not api_key or not api_secret:
        print("[FATAL] Credentials cannot be empty. Exiting.")
        sys.exit(1)

    try:
        os.makedirs("config", exist_ok=True)
        with open(CREDENTIALS_FILE, 'w') as f:
            json.dump({"api_key": api_key, "api_secret": api_secret}, f, indent=4)
        print(f"[SYSTEM] Credentials securely saved to {CREDENTIALS_FILE} for future use.")
    except Exception as e:
        print(f"[WARNING] Could not save credentials: {e}")

    return api_key, api_secret


async def async_input(prompt: str, timeout: int = 10):
    """Non-blocking Windows CLI input with timeout. Stops UI threads from hanging indefinitely."""
    print(prompt, end='', flush=True)
    start_time = time.time()
    response = ""
    while True:
        if msvcrt.kbhit():
            char = msvcrt.getwche()
            if char in ('\r', '\n'):
                print()
                return response
            elif char == '\b':
                response = response[:-1]
                print(" \b", end="", flush=True)
            else:
                response += char
        
        if time.time() - start_time > timeout:
            print("\n[SYSTEM] Input timed out (no response). Continuing execution...")
            return None
            
        await asyncio.sleep(0.05)


async def track_coin_loop(coin, timeframe, sleep_seconds, streamer, executor):
    """The highly autonomous, asynchronous inference loop isolated per coin."""
    print(f"[SYSTEM] Booting isolated Tracker Thread for {coin}...")
    
    ai_engine = ModelEngine(input_size=3)
    engineer = FeatureEngineer(window_size=60)
    
    coin_clean = coin.replace('/', '_')
    model_filepath = os.path.join("models", f"{coin_clean}_lstm_weights.pth")
    scaler_filepath = os.path.join("models", f"{coin_clean}_scaler.pkl")
    
    # Pre-loading the weights rather than training from scratch every hour!
    ai_engine.load_weights(model_filepath)
    engineer.load_scaler(scaler_filepath)

    while True:
        try:
            cycle_time = datetime.now().strftime('%H:%M:%S')
            
            # 1. Fetch data off the main thread so we don't halt other coin loops
            historical_data = await asyncio.to_thread(
                streamer.fetch_historical_candles, [coin], timeframe, 500
            )
            df = historical_data.get(coin)
            
            if df is None or df.empty:
                print(f"[WARNING] Invalid data returned for {coin}, sleeping...")
                await asyncio.sleep(60)
                continue

            # 2. Synthesize & Normalize
            df_features = engineer.apply_technical_indicators(df)
            df_targets = engineer.engineer_target_variable(df_features)
            
            # is_training=False ensures we ONLY apply the historical scaler and don't re-fit!
            df_normalized = engineer.normalize_data(df_targets, is_training=False)
            X, y = engineer.create_3d_tensor(df_normalized)

            # 3. Live Inference
            latest_window = X[-1]
            predicted_return = await asyncio.to_thread(ai_engine.predict_next_candle, latest_window)

            predicted_pct = (np.exp(predicted_return) - 1) * 100
            current_price = df['close'].iloc[-1]
            predicted_target_price = current_price * np.exp(predicted_return)

            t_now_str = datetime.now().strftime("%H:%M:%S")

            print(f"\n" + "=" * 60)
            print(f"AI PREDICTION REPORT [{t_now_str}]: {coin} ".center(60, "="))
            print(f"=" * 60)
            print(f"Current Rate: ${current_price:.6f}")
            print(f"Target Rate : ${predicted_target_price:.6f}")
            print(f"Expected Move: {predicted_pct:+.4f}%")
            print(f"=" * 60)

            # 4. Real-time Async Execution Router
            if predicted_pct > 0.1 or predicted_pct < -0.1:
                signal_direction = 'BUY' if predicted_pct > 0.1 else 'SELL'
                print(f"\n[{coin}] ACTIONABLE SIGNAL DETECTED: {signal_direction}")

                auth = await async_input(
                    f"[AUTHORIZATION REQUIRED] Execute {signal_direction} order on {coin} at ${current_price:.4f}? (y/n within 10s): ", 
                    timeout=10
                )

                if auth and auth.strip().lower() == 'y':
                    print(f"\n[SYSTEM] Authorization accepted for {coin}. Engaging Execution Router...")
                    await asyncio.to_thread(executor.process_signal, coin, signal_direction, current_price)
                else:
                    print(f"[SYSTEM] Authorization denied or timed out for {coin}. Trade aborted.")
            else:
                print(f"\n[ACTION] Market is flat for {coin} (Move < 0.1%). HOLD. No execution required.")

            print(f"[{coin}] Hibernating for {timeframe} until next candle...")
            await asyncio.sleep(sleep_seconds)

        except Exception as loop_error:
            print(f"\n[ERROR] Disruptions on {coin} loop: {loop_error}")
            await asyncio.sleep(60)


async def main_async():
    print("=" * 60)
    print(" NEURAL TRADING BOT v3.0 (ASYNC INFERENCE) ".center(60, "="))
    print("=" * 60)

    api_key, api_secret = load_or_prompt_credentials()

    print("\n--- Strategy Parameters ---")
    coins_input = input("Enter coins to trade (comma separated, max 5. e.g. BTC/USDT): ")
    selected_coins = [coin.strip().upper() for coin in coins_input.split(',')]
    if not selected_coins or selected_coins == ['']:
        selected_coins = ['BTC/USDT']
        
    timeframe = input("Enter timeframe (e.g., 15m, 1h, 1d) [Default: 1h]: ").strip() or "1h"
    if timeframe.isnumeric():
        timeframe += 'm'

    print("\n[SYSTEM] Initializing Core Architecture...")
    streamer = DataStreamer(api_key, api_secret, testnet=True)
    executor = ExecutionManager(streamer.exchange)

    tf_val = int(timeframe[:-1])
    tf_unit = timeframe[-1].lower()
    if tf_unit == 'm': sleep_seconds = tf_val * 60
    elif tf_unit == 'h': sleep_seconds = tf_val * 3600
    elif tf_unit == 'd': sleep_seconds = tf_val * 86400
    else: sleep_seconds = 3600

    print(f"\n[SYSTEM] ENTERING AUTONOMOUS LIVE INFERENCE PHASE.")
    
    # Launch parallel tracking loops for multiple coins
    tasks = []
    for coin in selected_coins:
        tasks.append(track_coin_loop(coin, timeframe, sleep_seconds, streamer, executor))
        
    # Wait indefinitely as the loops run in parallel
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\n[SYSTEM] Process manually aborted by user. Shutting down Live Loop.")
        sys.exit(0)