import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import pandas as pd
from pathlib import Path
from typing import List, Tuple

class StockDataset(Dataset):
    def __init__(self, csv_file: Path, time_format="%Y-%m-%d %H:%M:%S"):
        """
        Args:
            csv_file (Path): Path to the CSV file containing stock data.
            time_format (str): Format of the time string in the dataset.
        """
        # Load the CSV into a Pandas DataFrame
        self.data = pd.read_csv(csv_file).dropna()
        self.trend_mapping = {f'u{i}': i for i in range(1, 101)}  # Map upward trends
        self.trend_mapping.update({f'd{i}': 100 + i for i in range(1, 101)})  # Map downward trends
        self.trend_mapping.update({'</s>': 0}) # Special token which is BOS or SEP
        self.time_format = time_format
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        # Get the row
        row = self.data.iloc[idx]
        
        # HLOC Features (Tensor of size (16,))
        hloc_array = row[['open', 'high', 'low', 'close', 'close_open_diff', 'high_low_range', 'midprice', 
            'typical_price', 'log_return', 'open_close_return', 'sma_100', 'ema_100', 'atr', 'rolling_std', 
            'roc_20', 'rsi_20']].tolist()
        hloc_array = [i.tolist() for i in hloc_array]
        hloc_features = torch.tensor(hloc_array, dtype=torch.float32)
        
        # Trend Index (Scalar)
        trend_index = self.trend_mapping[row['trends']]
        
        # Time Feature (Extracted from 'date')
        time_str = row['date']
        time_dt = datetime.strptime(time_str, self.time_format)
        time_features = torch.tensor([time_dt.minute, time_dt.hour, time_dt.weekday(), time_dt.day, time_dt.month], dtype=torch.float32)
        
        return {"hloc": hloc_features, "trend_index": trend_index, "time": time_features}

if __name__ == "__main__":
    ds = StockDataset(csv_file=Path(Path.cwd(), "data/nifty_trend_data.csv"))
    print(len(ds))
    print(ds[0])
    dl = DataLoader(ds, batch_size=4)
    print(next(iter(dl)))