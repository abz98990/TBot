import os
import json
import sys
import time
import asyncio
import subprocess
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


async def track_coin_loop(coin, timeframe, sleep_seconds, streamer, executor, auto_trade=False):
    """The highly autonomous, asynchronous inference loop isolated per coin."""
    print(f"[SYSTEM] Booting isolated Tracker Thread for {coin}...")
    
    ai_engine = ModelEngine(input_size=6)
    engineer = FeatureEngineer(window_size=60)
    
    coin_clean = coin.replace('/', '_')
    model_filepath = os.path.join("models", f"{coin_clean}_lstm_weights.pth")
    scaler_filepath = os.path.join("models", f"{coin_clean}_scaler.pkl")
    
    # Pre-loading the weights rather than training from scratch every hour!
    ai_engine.load_weights(model_filepath)
    engineer.load_scaler(scaler_filepath)

    prediction_queue = []
    open_position = None
    initial_entry_price = None
    cumulative_net_pnl = 0.0

    log_file = os.path.join("logs", f"{coin_clean}_performance.csv")
    os.makedirs("logs", exist_ok=True)

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

            current_price = df['close'].iloc[-1]

            # --- Position Management (4-Candle Exit) ---
            if open_position is not None:
                open_position['candles_held'] += 1
                if open_position['candles_held'] >= 4:
                    print(f"\n[SYSTEM] 4 Candles elapsed. Closing existing position for {coin}...")
                    exit_price = await asyncio.to_thread(executor.close_position, coin, open_position)
                    if exit_price and exit_price > 0:
                        current_price = exit_price # Use the actual exit price for our PnL calc
                        
                    # Calculate trade PnL
                    entry_price = open_position['entry_price']
                    if open_position['side'] == 'buy':
                        trade_pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    else:
                        trade_pnl_pct = ((entry_price - current_price) / entry_price) * 100
                        
                    cumulative_net_pnl += trade_pnl_pct
                    print(f"[TRACKING] Trade PnL: {trade_pnl_pct:+.2f}% | Cumulative Net PnL: {cumulative_net_pnl:+.2f}%")
                    
                    if initial_entry_price is not None:
                        asset_deviation = ((current_price - initial_entry_price) / initial_entry_price) * 100
                        print(f"[TRACKING] Asset deviation since FIRST trade: {asset_deviation:+.2f}%")
                    
                    open_position = None
                else:
                    print(f"\n[SYSTEM] Holding position for {coin}... ({open_position['candles_held']}/4 candles)")
            # ------------------------------------------------

            # --- Quantitative Analysis ---
            for p in prediction_queue:
                p['candles_elapsed'] += 1
                
                # Check accuracy for predictions that have reached 4 candles and haven't been evaluated yet
                if p['candles_elapsed'] >= 4 and p['actual_class'] == '':
                    actual_class = 1 if current_price > p['price_then'] else 0
                    predicted_class = 1 if p['prob'] > 0.5 else 0
                    is_correct = int(predicted_class == actual_class)
                    
                    p['actual_class'] = actual_class
                    p['is_correct'] = is_correct
                    
                    print(f"\n[QUANTITATIVE ANALYSIS] {coin}")
                    print(f"Prediction from 4 candles ago: {p['prob']*100:.2f}% (UP)")
                    print(f"Actual Outcome (4-candle)    : {'UP' if actual_class == 1 else 'DOWN'}")
                    print(f"Prediction Correct?          : {'YES' if is_correct else 'NO'}")
            
            # Keep only the last 100 predictions in memory to prevent leak
            if len(prediction_queue) > 100:
                prediction_queue.pop(0)
            # -----------------------------

            # 2. Synthesize & Normalize
            df_features = engineer.apply_technical_indicators(df)
            
            df_train = df_features.copy()
            df_train = engineer.engineer_target_variable(df_train)
            df_train = engineer.normalize_data(df_train, is_training=False)
            X_train, y_train = engineer.create_3d_tensor(df_train)

            # Continuous Calibration (Online Learning)
            print(f"[MODEL] Calibrating {coin} with latest actuals...")
            await asyncio.to_thread(ai_engine.train, X_train, y_train, epochs=3, batch_size=32, verbose=False)
            await asyncio.to_thread(ai_engine.save_model, model_filepath)

            # 3. Live Inference
            df_infer = df_features.copy()
            df_infer[engineer.feature_columns] = engineer.scaler.transform(df_infer[engineer.feature_columns])
            
            feature_data = df_infer[engineer.feature_columns].values
            latest_window = feature_data[-engineer.window_size:]
            
            predicted_prob = await asyncio.to_thread(ai_engine.predict_next_candle, latest_window)

            # Enqueue the new prediction
            prediction_queue.append({
                'cycle_time': cycle_time,
                'prob': predicted_prob,
                'price_then': current_price,
                'rsi': df_features['RSI_14'].iloc[-1],
                'macd_hist': df_features['MACDh_12_26_9'].iloc[-1],
                'adx': df_features['ADX_14'].iloc[-1],
                'candles_elapsed': 0,
                'actual_class': '',
                'is_correct': '',
                'cumulative_net_pnl': cumulative_net_pnl
            })
            
            # Rewrite the entire CSV with the rolling history so dashboard updates instantly
            with open(log_file, "w") as f:
                f.write("timestamp,last_price,predicted_prob,actual_class,accuracy,cumulative_pnl,rsi,macd_hist,adx\n")
                for p in prediction_queue:
                    f.write(f"{p['cycle_time']},{p['price_then']},{p['prob']:.4f},{p['actual_class']},{p['is_correct']},{p['cumulative_net_pnl']:.4f},{p['rsi']:.2f},{p['macd_hist']:.4f},{p['adx']:.2f}\n")

            t_now_str = datetime.now().strftime("%H:%M:%S")

            print(f"\n" + "=" * 60)
            print(f"AI CLASSIFICATION REPORT [{t_now_str}]: {coin} ".center(60, "="))
            print(f"=" * 60)
            print(f"Current Rate: ${current_price:.6f}")
            print(f"Confidence (UP): {predicted_prob*100:.2f}%")
            print(f"=" * 60)

            # 4. Real-time Async Execution Router
            if open_position is not None:
                print(f"\n[ACTION] Already holding a position for {coin}. Skipping new signals.")
            elif predicted_prob > 0.55 or predicted_prob < 0.45:
                signal_direction = 'BUY' if predicted_prob > 0.55 else 'SELL'
                print(f"\n[{coin}] HIGH CONFIDENCE SIGNAL DETECTED: {signal_direction}")

                if auto_trade:
                    auth = 'y'
                    print(f"[API] Auto-Trade is ACTIVE. Bypassing manual authorization.")
                else:
                    auth = await async_input(
                        f"[AUTHORIZATION REQUIRED] Execute {signal_direction} order on {coin} at ${current_price:.4f}? (y/n within 10s): ", 
                        timeout=10
                    )

                if auth and auth.strip().lower() == 'y':
                    print(f"\n[SYSTEM] Authorization accepted for {coin}. Engaging Execution Router...")
                    
                    # Binance requires trades to have a minimum value of usually $10 (MIN_NOTIONAL)
                    # We dynamically calculate the coin quantity to equate to exactly $15.00 USD securely.
                    trade_qty = 15.0 / current_price
                    
                    position_info = await asyncio.to_thread(executor.process_signal, coin, signal_direction, current_price, trade_qty)
                    if position_info:
                        position_info['candles_held'] = 0
                        open_position = position_info
                        if initial_entry_price is None:
                            initial_entry_price = position_info['entry_price']
                else:
                    print(f"[SYSTEM] Authorization denied or timed out for {coin}. Trade aborted.")
            else:
                print(f"\n[ACTION] AI is uncertain ({predicted_prob*100:.2f}%). HOLD. No execution required.")

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
    
    launch_dash = input("\nLaunch Real-Time Dashboard alongside bot? (y/n) [Default: y]: ").strip().lower()
    if launch_dash != 'n':
        target_csv = f"logs/{selected_coins[0].replace('/', '_')}_performance.csv"
        print(f"[SYSTEM] Spawning dashboard.py in a background process...")
        # Open in a new console window on Windows
        creationflags = subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
        subprocess.Popen([sys.executable, "dashboard.py", target_csv], creationflags=creationflags)

    auto_trade_input = input("\nEnable Auto-Trading (bypass 10s manual confirmation)? (y/n) [Default: n]: ").strip().lower()
    auto_trade = auto_trade_input == 'y'

    # Launch parallel tracking loops for multiple coins
    tasks = []
    for coin in selected_coins:
        tasks.append(track_coin_loop(coin, timeframe, sleep_seconds, streamer, executor, auto_trade=auto_trade))
        
    # Wait indefinitely as the loops run in parallel
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\n[SYSTEM] Process manually aborted by user. Shutting down Live Loop.")
        sys.exit(0)