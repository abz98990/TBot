# module_1_data.py
import ccxt
import pandas as pd
import time


class DataStreamer:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """Initializes the exchange connection securely."""
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,  # CRITICAL: Prevents Binance from banning your IP
        })
        if testnet:
            self.exchange.set_sandbox_mode(True)
            print("[SYSTEM] Connected to Binance TESTNET.")
        else:
            print("[SYSTEM] WARNING: Connected to Binance MAINNET. Real funds at risk.")

    def fetch_historical_candles(self, symbols: list, timeframe: str = '15m', limit: int = 1000) -> dict:
        """
        Fetches OHLCV data for multiple coins.
        Returns a dictionary of Pandas DataFrames.
        """
        market_data = {}

        for symbol in symbols:
            try:
                print(f"[DATA] Fetching {limit} candles for {symbol} on {timeframe} timeframe...")

                # Fetch data from exchange
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

                # Convert pure arrays to a structured Pandas DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                market_data[symbol] = df

                # Polite delay to respect exchange rate limits when looping multiple coins
                time.sleep(self.exchange.rateLimit / 1000)

            except Exception as e:
                print(f"[ERROR] Failed to fetch data for {symbol}: {e}")

        return market_data