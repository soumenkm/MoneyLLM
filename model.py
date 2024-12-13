import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import pandas as pd
from pathlib import Path
from typing import List, Tuple

class Preprocessor(torch.nn.Module):
    def __init__(self, device: str, embedding_dim: int, num_trends: int = 201, num_hloc_features: int = 16):
        super(Preprocessor, self).__init__()
        self.device = device
        self.d = embedding_dim
        self.num_trends = num_trends
        self.num_hloc_features = num_hloc_features
        self.trend_embed = torch.nn.Embedding(num_embeddings=self.num_trends, embedding_dim=self.d).to(self.device)
        self.hloc_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.num_hloc_features, out_features=self.num_hloc_features * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(in_features=self.num_hloc_features * 4, out_features=self.d)
        ).to(self.device)
        self.cross_attn = torch.nn.MultiheadAttention(embed_dim=self.d, num_heads=8, batch_first=True).to(self.device)
    
    def forward(self, hloc_features: torch.Tensor, trend_indices: torch.Tensor) -> torch.Tensor:
        assert hloc_features.dim() == 3, "hloc_features must have 3 dimensions (batch_size, seq_len, num_hloc_features)"
        assert hloc_features.shape[-1] == self.num_hloc_features, f"hloc_features last dimension must be {self.num_hloc_features}"
        assert trend_indices.dim() == 2, "trend_indices must have 2 dimensions (batch_size, seq_len)"
        
        x1 = self.hloc_mlp(hloc_features.to(self.device)) # (b, T, d)
        x2 = self.trend_embed(trend_indices.to(self.device)) # (b, T, d)
        z = self.cross_attn(query=x1, key=x2, value=x2)[0] # (b, T, d)
        return z

class TimeEmbedding(torch.nn.Module):
    def __init__(self, device: str, embedding_dim: int, num_time_features: int):
        super(TimeEmbedding, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.num_time_features = num_time_features

        # Time2Vec learnable parameters
        self.linear = torch.nn.Linear(num_time_features, embedding_dim).to(self.device)  # Linear term
        self.periodic_weights = torch.nn.Parameter(torch.randn(num_time_features, embedding_dim)).to(self.device)  # Periodic weights
        self.bias = torch.nn.Parameter(torch.randn(embedding_dim)).to(self.device)  # Bias term

    def forward(self, time_features: torch.Tensor) -> torch.Tensor:
        assert time_features.dim() == 3, "time_features must have 3 dimensions (batch_size, seq_len, num_time_features)"
        assert time_features.shape[-1] == self.num_time_features, f"time_features last dimension must be {self.num_time_features}"
        
        linear_term = self.linear(time_features.to(self.device))  # (b, T, d)
        periodic_term = torch.sin(torch.matmul(time_features.to(self.device), self.periodic_weights) + self.bias)  # (b, T, d)
        time_embedding = linear_term + periodic_term # (b, T, d)
        return time_embedding    

class SinusoidalPositionalEmbedding(torch.nn.Module):
    def __init__(self, device: str, max_seq_len: int, embedding_dim: int):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.device = device
        self.seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.register_buffer("positional_encodings", self._generate_sinusoidal_embeddings(self.seq_len, embedding_dim))

    def _generate_sinusoidal_embeddings(self, seq_len: int, embedding_dim: int) -> torch.Tensor:
        position = torch.arange(seq_len).unsqueeze(1).to(self.device)  # (T, 1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).to(self.device) * -(torch.log(torch.tensor(10000.0).to(self.device)) / embedding_dim))  # (d/2, 1)
        sinusoidal_embeddings = torch.zeros(seq_len, embedding_dim).to(self.device)
        sinusoidal_embeddings[:, 0::2] = torch.sin(position * div_term).to(self.device)  # Even indices
        sinusoidal_embeddings[:, 1::2] = torch.cos(position * div_term).to(self.device)  # Odd indices
        return sinusoidal_embeddings.unsqueeze(0)  # Add batch dimension (1, T, d)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        assert positions.dim() == 2, "positions must have 2 dimensions (batch_size, seq_len)"
        assert positions.shape[-1] <= self.seq_len, f"positions last dimension must be less than or equal to {self.seq_len}"
        
        batch_size = positions.shape[0]
        return self.positional_encodings[:, :positions.shape[-1], :].repeat(batch_size, 1, 1)  # (b, T, d)
    
if __name__ == "__main__":
    pos = torch.arange(start=0, end=256, step=1).unsqueeze(0).repeat(4, 1).to("cuda") # (b, T)
    pos_emb = SinusoidalPositionalEmbedding(device="cuda", max_seq_len=512, embedding_dim=768)
    y = pos_emb(pos)
    print(y)
    print(y.shape)
    
    # time = torch.rand(size=(4, 256, 5)).to("cuda")
    # time_emb = TimeEmbedding(device="cuda", embedding_dim=768, num_time_features=5)
    # y = time_emb(time)
    # print(y)
    # print(y.shape)