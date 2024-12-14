import torch
torch.manual_seed(42)
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import pandas as pd
from pathlib import Path
from typing import Dict

class StockDataset(Dataset):
    def __init__(self, csv_file: Path, max_seq_length: int, frac: float, time_format="%Y-%m-%d %H:%M:%S"):
        self.data = pd.read_csv(csv_file).dropna()
        self.data = self.data.iloc[0: int(len(self.data) * frac)]
        self.trend_mapping = {f'u{i}': i for i in range(1, 101)}  # Map upward trends
        self.trend_mapping.update({f'd{i}': 100 + i for i in range(1, 101)})  # Map downward trends
        self.trend_mapping.update({'</s>': 0})  # Special token which is BOS or SEP
        self.time_format = time_format
        self.Tmax = max_seq_length  

    def __len__(self) -> int:
        # Ensure we don't go out of bounds with the sequence
        return len(self.data) - 2 * self.Tmax + 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc_seq = self.data.iloc[idx:idx + self.Tmax]
        dec_seq = self.data.iloc[idx + self.Tmax:idx + 2 * self.Tmax]
        target_dec_seq = self.data.iloc[idx + self.Tmax + 1:idx + 2 * self.Tmax + 1]
        
        # HLOC Features (Tensor of size (T, 16))
        enc_hloc_features = torch.tensor(enc_seq[[
            'open', 'high', 'low', 'close', 'close_open_diff', 'high_low_range', 'midprice', 
            'typical_price', 'log_return', 'open_close_return', 'sma_100', 'ema_100', 'atr', 
            'rolling_std', 'roc_20', 'rsi_20'
        ]].values, dtype=torch.float32)  # Shape: (T, 16)
        dec_hloc_features = torch.tensor(dec_seq[[
            'open', 'high', 'low', 'close', 'close_open_diff', 'high_low_range', 'midprice', 
            'typical_price', 'log_return', 'open_close_return', 'sma_100', 'ema_100', 'atr', 
            'rolling_std', 'roc_20', 'rsi_20'
        ]].values, dtype=torch.float32)  # Shape: (T, 16)
        dec_hloc_target_features = torch.tensor(target_dec_seq[[
            'open', 'high', 'low', 'close', 'close_open_diff', 'high_low_range', 'midprice', 
            'typical_price', 'log_return', 'open_close_return', 'sma_100', 'ema_100', 'atr', 
            'rolling_std', 'roc_20', 'rsi_20'
        ]].values, dtype=torch.float32)  # Shape: (T, 16)

        # Trend Index (Tensor of size (T,))
        enc_trend_indices = torch.tensor([self.trend_mapping[trend] for trend in enc_seq['trends']], dtype=torch.int32)  # Shape: (T,)
        dec_trend_indices = torch.tensor([self.trend_mapping[trend] for trend in dec_seq['trends']], dtype=torch.int32)  # Shape: (T,)
        dec_trend_target_indices = torch.tensor([self.trend_mapping[trend] for trend in target_dec_seq['trends']], dtype=torch.int64)  # Shape: (T,)

        # Time Features (Tensor of size (T, 5))
        enc_time_features = torch.tensor(
            [[
                datetime.strptime(row['date'], self.time_format).minute,
                datetime.strptime(row['date'], self.time_format).hour,
                datetime.strptime(row['date'], self.time_format).weekday(),
                datetime.strptime(row['date'], self.time_format).day,
                datetime.strptime(row['date'], self.time_format).month
            ] for _, row in enc_seq.iterrows()], dtype=torch.float32
        )  # Shape: (T, 5)
        dec_time_features = torch.tensor(
            [[
                datetime.strptime(row['date'], self.time_format).minute,
                datetime.strptime(row['date'], self.time_format).hour,
                datetime.strptime(row['date'], self.time_format).weekday(),
                datetime.strptime(row['date'], self.time_format).day,
                datetime.strptime(row['date'], self.time_format).month
            ] for _, row in dec_seq.iterrows()], dtype=torch.float32
        )  # Shape: (T, 5)
        dec_time_target_features = torch.tensor(
            [[
                datetime.strptime(row['date'], self.time_format).minute,
                datetime.strptime(row['date'], self.time_format).hour,
                datetime.strptime(row['date'], self.time_format).weekday(),
                datetime.strptime(row['date'], self.time_format).day,
                datetime.strptime(row['date'], self.time_format).month
            ] for _, row in target_dec_seq.iterrows()], dtype=torch.float32
        )  # Shape: (T, 5)

        return {
            "enc_hloc_features": enc_hloc_features, "enc_trend_indices": enc_trend_indices, "enc_time_features": enc_time_features,
            "dec_hloc_features": dec_hloc_features, "dec_trend_indices": dec_trend_indices, "dec_time_features": dec_time_features,
            "dec_hloc_target_features": dec_hloc_target_features[:, :4], "dec_trend_target_indices": dec_trend_target_indices
        }

# Example Usage
if __name__ == "__main__":
    # Path to the CSV file
    csv_path = Path(Path.cwd(), "data/nifty_trend_data.csv")
    
    # Initialize Dataset
    ds = StockDataset(csv_file=csv_path, max_seq_length=10, frac=1.0)  # T = 10
    print(f"Dataset length: {len(ds)}")
    print("First example:", {k: v.shape for k, v in ds[0].items()})

    # DataLoader with batch size 4
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    batch = next(iter(dl))
    print("Batch example:")
    print({key: value.shape for key, value in batch.items()})  # Print shapes
    
    # Check outofindex error
    print("Last example:", {k: v.shape for k, v in ds[-1].items()})
