import asyncio
import pandas as pd
from module_1_data import DataStreamer
from module_2_features import FeatureEngineer

async def main():
    streamer = DataStreamer("", "", testnet=False)
    engineer = FeatureEngineer()
    
    data = streamer.fetch_historical_candles(['ETH/USDT'], '1h', 500)
    df = data['ETH/USDT']
    
    df['RSI_14'] = df.ta.rsi(length=14)
    atr = df.ta.atr(length=14)
    df['ATR_pct'] = atr / df['close']
    ema_50 = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA_50_dist'] = (df['close'] - ema_50) / ema_50
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    adx = df.ta.adx(length=14)
    df = pd.concat([df, adx], axis=1)

    print(f"Total rows before dropna: {len(df)}")
    
    # Check NaNs per column
    print("NaN counts per column:")
    print(df.isna().sum())
    
    df.dropna(inplace=True)
    print(f"Total rows after dropna: {len(df)}")

if __name__ == '__main__':
    asyncio.run(main())
