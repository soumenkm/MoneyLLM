import torch, pickle
torch.manual_seed(42)
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import pandas as pd
from pathlib import Path
from typing import Dict

class StockDataset(Dataset):
    def __init__(self, pkl_file: Path, max_seq_length: int, frac: float, time_format="%Y-%m-%d %H:%M:%S"):
        print("Preparing dataset...")
        self.data = self._get_data(pkl_file=pkl_file, frac=frac, time_format=time_format)
        self.trend_mapping = {i: self.quantile[i-1] for i in range(1, 201)}  # Map trends
        self.trend_mapping.update({0: 0.0})  # Special token which is BOS or SEP
        self.time_format = time_format
        self.Tmax = max_seq_length - 1  
    
    def _get_data(self, pkl_file: Path, frac: float, time_format: str) -> pd.DataFrame:
        data = pickle.load(open(pkl_file, "rb"))
        self.quantile = data.loc[0, "quantile"]
        data.drop(columns=['quantile'], inplace=True)
        data.dropna(inplace=True)
        data = data.iloc[0: int(len(data) * frac)]
        
        self.hloc_feature_cols = ['open', 'high', 'low', 'close', 'close_open_diff', 'high_low_range',
            'midprice', 'typical_price', 'log_return', 'open_close_return',
            'sma_100', 'ema_100', 'atr', 'rolling_std', 'roc_20', 'rsi_20'
        ]
        mean = data[self.hloc_feature_cols].mean()
        std = data[self.hloc_feature_cols].std()
        std = std.replace(0, 1)
        data[self.hloc_feature_cols] = (data[self.hloc_feature_cols] - mean) / std
        
        data['minute'] = data['date'].apply(lambda x: datetime.strptime(x, time_format).minute)
        data['hour'] = data['date'].apply(lambda x: datetime.strptime(x, time_format).hour)
        data['weekday'] = data['date'].apply(lambda x: datetime.strptime(x, time_format).weekday())
        data['day'] = data['date'].apply(lambda x: datetime.strptime(x, time_format).day)
        data['month'] = data['date'].apply(lambda x: datetime.strptime(x, time_format).month)

        self.time_feature_cols = ['minute', 'hour', 'weekday', 'day', 'month']
        time_mean = data[self.time_feature_cols].mean()
        time_std = data[self.time_feature_cols].std()
        time_std = time_std.replace(0, 1)
        data[self.time_feature_cols] = (data[self.time_feature_cols] - time_mean) / time_std
        
        return data
    
    def __len__(self) -> int:
        # Ensure we don't go out of bounds with the sequence
        return len(self.data) - 2 * self.Tmax - 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc_seq = self.data.iloc[idx:idx + self.Tmax] # (T-1)
        dec_seq = self.data.iloc[idx + self.Tmax:idx + 2 * self.Tmax] # (T-1)
        target_dec_seq = self.data.iloc[idx + self.Tmax + 1:idx + 2 * self.Tmax + 2] # (T)
        
        # HLOC Features (Tensor of size (T-1, 16))
        enc_hloc_features = torch.tensor(enc_seq[self.hloc_feature_cols].values, dtype=torch.float32)  # Shape: (T-1, 16)
        dec_hloc_features = torch.tensor(dec_seq[self.hloc_feature_cols].values, dtype=torch.float32)  # Shape: (T-1, 16)
        dec_hloc_target_features = torch.tensor(target_dec_seq[self.hloc_feature_cols].values, dtype=torch.float32)  # Shape: (T, 16)
        
        # Prepend start token for HLOC features
        enc_hloc_features = torch.cat([
            torch.zeros((1, enc_hloc_features.shape[1]), dtype=torch.float32),  # Zero placeholder
            enc_hloc_features
        ], dim=0) # Shape: (T, 16)
        dec_hloc_features = torch.cat([
            torch.zeros((1, dec_hloc_features.shape[1]), dtype=torch.float32),  # Zero placeholder
            dec_hloc_features
        ], dim=0) # Shape: (T, 16)

        # Trend Index (Tensor of size (T,))
        enc_trend_indices = torch.tensor([0] + enc_seq['trends'].tolist(), dtype=torch.int64)  # Shape: (T,)
        dec_trend_indices = torch.tensor([0] + dec_seq['trends'].tolist(), dtype=torch.int64)  # Shape: (T,)
        dec_trend_target_indices = torch.tensor(target_dec_seq['trends'].tolist(), dtype=torch.float32)  # Shape: (T,)
   
        # Time Features (Tensor of size (T, 5))
        enc_time_features = torch.cat([
            torch.zeros((1, 5), dtype=torch.float32),  # Zero placeholder
            torch.tensor(enc_seq[self.time_feature_cols].values, dtype=torch.float32)
        ], dim=0)  # Shape: (T, 5)

        dec_time_features = torch.cat([
            torch.zeros((1, 5), dtype=torch.float32),  # Zero placeholder
            torch.tensor(dec_seq[self.time_feature_cols].values, dtype=torch.float32)
        ], dim=0)  # Shape: (T, 5)

        return {
            "enc_hloc_features": enc_hloc_features, "enc_trend_indices": enc_trend_indices, "enc_time_features": enc_time_features,
            "dec_hloc_features": dec_hloc_features, "dec_trend_indices": dec_trend_indices, "dec_time_features": dec_time_features,
            "dec_hloc_target_features": dec_hloc_target_features[:, :4], "dec_trend_target_indices": dec_trend_target_indices
        }

# Example Usage
if __name__ == "__main__":
    # Path to the pkl file
    pkl_path = Path(Path.cwd(), "data/nifty_trend_data.pkl")
    
    # Initialize Dataset
    ds = StockDataset(pkl_file=pkl_path, max_seq_length=10, frac=0.001)  # T = 10
    print(f"Dataset length: {len(ds)}")
    print("First example:", {k: v.shape for k, v in ds[0].items()})

    # DataLoader with batch size 4
    dl = DataLoader(ds, batch_size=4, shuffle=True, drop_last=True)
    batch = next(iter(dl))
    print("Batch example:")
    print({key: value.shape for key, value in batch.items()})  # Print shapes
    print({key: value.dtype for key, value in batch.items()}) 
    
    # Check outofindex error
    print("Last example:", {k: v.shape for k, v in ds[-1].items()})
    a = [batch for batch in dl]
