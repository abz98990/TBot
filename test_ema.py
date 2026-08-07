import asyncio
from module_1_data import DataStreamer
from module_2_features import FeatureEngineer
import pandas as pd

async def run():
    streamer = DataStreamer(None, None, testnet=True)
    engineer = FeatureEngineer()
    
    data = streamer.fetch_historical_candles(['ETH/USDT'], '1h', 500)
    df = data['ETH/USDT']
    
    ema_50 = df.ta.ema(length=50)
    print(f"Type of ema_50: {type(ema_50)}")
    if isinstance(ema_50, pd.DataFrame):
        print(f"Columns: {ema_50.columns}")
        
    df['EMA_50_dist'] = (df['close'] - ema_50) / ema_50
    print("Success")

if __name__ == '__main__':
    asyncio.run(run())
