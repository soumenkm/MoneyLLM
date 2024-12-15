import torch
torch.manual_seed(42)
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import pandas as pd
from pathlib import Path
from typing import List, Tuple

class Preprocessor(torch.nn.Module):
    def __init__(self, device: str, embedding_dim: int, num_trends: int=201, num_hloc_features: int=16):
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

class MLP(torch.nn.Module):
    def __init__(self, device: str, hidden_dim: int, dropout_prob: float=0.2):
        super(MLP, self).__init__()
        self.device = device
        self.d = hidden_dim
        self.up_proj = torch.nn.Linear(in_features=self.d, out_features=4*self.d).to(self.device)
        self.silu = torch.nn.SiLU()
        self.down_proj = torch.nn.Linear(in_features=4*self.d, out_features=self.d).to(self.device)
        self.dropout = torch.nn.Dropout(dropout_prob)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[-1] == self.d, f"inputs.shape must be (b, T, {self.d})"
        assert list(inputs.shape).__len__() == 3, "inputs rank must be 3"
        
        x = self.up_proj(inputs.to(self.device)) # (b, T, 4d)
        x = self.silu(x) # (b, T, 4d)
        x = self.down_proj(x) # (b, T, d)
        z = self.dropout(x) # (b, T, d)
        return z

class TransformerEncoder(torch.nn.Module):
    def __init__(self, device: str, embedding_dim: int, max_seq_length: int, num_heads: int=8, dropout_prob: float=0.2):
        super(TransformerEncoder, self).__init__()
        self.device = device
        self.d = embedding_dim
        self.Tmax = max_seq_length
        self.h = num_heads
        self.p = dropout_prob
        
        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=(self.d,)).to(self.device)
        self.mhsa = torch.nn.MultiheadAttention(embed_dim=self.d, num_heads=self.h, dropout=self.p, batch_first=True).to(self.device)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=(self.d,)).to(self.device)
        self.mlp = MLP(device=self.device, hidden_dim=self.d, dropout_prob=self.p).to(self.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[-1] == self.d, f"inputs.shape must be (b, T, {self.d})"
        assert list(inputs.shape).__len__() == 3, "inputs rank must be 3"
        
        x = inputs.to(self.device) # (b, T, d)
        x1 = self.layernorm1(x) # (b, T, d)
        x2 = self.mhsa(query=x1, key = x1, value=x1, is_causal=False)[0] # (b, T, d)
        y = x2 + x # (b, T, d)
        y1 = self.layernorm2(y) # (b, T, d)
        y2 = self.mlp(y1) # (b, T, d)
        z = y2 + y # (b, T, d)
        return z
   
class TransformerDecoder(torch.nn.Module):
    def __init__(self, device: str, embedding_dim: int, max_seq_length: int, num_heads: int=8, dropout_prob: float=0.2):
        super(TransformerDecoder, self).__init__()
        self.device = device
        self.d = embedding_dim
        self.Tmax = max_seq_length
        self.h = num_heads
        self.p = dropout_prob
        
        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=(self.d,)).to(self.device)
        self.mhsa = torch.nn.MultiheadAttention(embed_dim=self.d, num_heads=self.h, dropout=self.p, batch_first=True).to(self.device)
        self.mhca = torch.nn.MultiheadAttention(embed_dim=self.d, num_heads=self.h, dropout=self.p, batch_first=True).to(self.device)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=(self.d,)).to(self.device)
        self.layernorm3 = torch.nn.LayerNorm(normalized_shape=(self.d,)).to(self.device)
        self.mlp = MLP(device=self.device, hidden_dim=self.d, dropout_prob=self.p).to(self.device)

    def forward(self, inputs: dict) -> torch.Tensor:
        decoder_inputs = inputs["dec_inputs"]
        encoder_features = inputs["enc_outputs"]
        assert decoder_inputs.shape[-1] == self.d, f"decoder_inputs.shape must be (b, T, {self.d})"
        assert list(decoder_inputs.shape).__len__() == 3, "decoder_inputs rank must be 3"
        assert encoder_features.shape[-1] == self.d, f"encoder_features.shape must be (b, T, {self.d})"
        assert list(encoder_features.shape).__len__() == 3, "encoder_features rank must be 3"
        
        x = decoder_inputs.to(self.device) # (b, T, d)
        x_enc = encoder_features.to(self.device) # (b, T, d)
        x1 = self.layernorm1(x) # (b, T, d)
        seq_len = x1.shape[1] # T
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(self.device)  # (T, T)
        x2 = self.mhsa(query=x1, key = x1, value=x1, attn_mask=causal_mask, is_causal=True)[0] # (b, T, d)
        y = x2 + x # (b, T, d)
        y1 = self.layernorm2(y) # (b, T, d)
        
        x3 = self.mhca(query=y1, key=x_enc, value=x_enc, is_causal=False)[0] # (b, T, d)
        y2 = x3 + y1 # (b, T, d)
        y3 = self.layernorm3(y2) # (b, T, d)
        y4 = self.mlp(y3) # (b, T, d)
        z = y4 + y3 # (b, T, d)
        return z  
    
class Transformer(torch.nn.Module):
    def __init__(self, device: str, config: dict):
        super(Transformer, self).__init__()
        self.device = device
        self.config = config
        self.d = self.config["embedding_dim"]
        
        self.preprocessor = Preprocessor(device=self.device, embedding_dim=self.d, num_trends=self.config["num_trends"], num_hloc_features=self.config["num_hloc_features"]).to(self.device)
        self.pos_embedder = SinusoidalPositionalEmbedding(device=self.device, max_seq_len=self.config["max_seq_length"], embedding_dim=self.d).to(self.device)
        self.time_embedder = TimeEmbedding(device=self.device, embedding_dim=self.d, num_time_features=self.config["num_time_features"]).to(self.device)
        
        self.encoder_layers = torch.nn.ModuleList([TransformerEncoder(
            device=self.device, embedding_dim=self.d, max_seq_length=self.config["max_seq_length"], num_heads=self.config["num_heads"], dropout_prob=self.config["dropout_prob"]
        ) for _ in range(self.config["num_layers"])]).to(self.device)
        self.decoder_layers = torch.nn.ModuleList([TransformerDecoder(
            device=self.device, embedding_dim=self.d, max_seq_length=self.config["max_seq_length"], num_heads=self.config["num_heads"], dropout_prob=self.config["dropout_prob"]
        ) for _ in range(self.config["num_layers"])]).to(self.device)
        
        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=(self.d,)).to(self.device)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=(self.d,)).to(self.device)
        self.dropout = torch.nn.Dropout(p=self.config["dropout_prob"]).to(self.device)
        self.hloc_output = torch.nn.Linear(in_features=self.d, out_features=self.config["num_pred_hloc"]).to(self.device)
        self.trend_output = torch.nn.Linear(in_features=self.d, out_features=1).to(self.device)
         
    def forward(self, enc_hloc_features: torch.Tensor, enc_trend_indices: torch.Tensor, enc_time_features: torch.Tensor,
                dec_hloc_features: torch.Tensor, dec_trend_indices: torch.Tensor, dec_time_features: torch.Tensor, 
                dec_hloc_target_features: torch.Tensor, dec_trend_target_indices: torch.Tensor) -> torch.Tensor:
        assert enc_trend_indices.shape[0] == dec_trend_indices.shape[0], f"batch size should be same in both encoder and decoder inputs"
        
        b, Te = enc_trend_indices.shape
        b, Td = dec_trend_indices.shape 
        enc_hloc_features = enc_hloc_features.to(self.device)
        enc_trend_indices = enc_trend_indices.to(self.device)
        enc_time_features = enc_time_features.to(self.device)
        dec_hloc_features = dec_hloc_features.to(self.device)
        dec_trend_indices = dec_trend_indices.to(self.device)
        dec_time_features = dec_time_features.to(self.device)
        
        x1 = self.preprocessor(hloc_features=enc_hloc_features, trend_indices=enc_trend_indices) # (b, Te, d)
        enc_pos_tokens = torch.arange(Te).unsqueeze(0).repeat(b, 1).to(self.device) # (b, Te)
        x2 = self.pos_embedder(enc_pos_tokens) # (b, Te, d)
        x3 = self.time_embedder(enc_time_features) # (b, Te, d)
        x = x1 + x2 + x3 # (b, Te, d)
        x = self.dropout(x) # (b, Te, d)
        for layer in self.encoder_layers:
            x = layer(x) # (b, Te, d)
        z1 = self.layernorm1(x) # (b, Te, d)
        
        y1 = self.preprocessor(hloc_features=dec_hloc_features, trend_indices=dec_trend_indices) # (b, Td, d)
        dec_pos_tokens = torch.arange(Td).unsqueeze(0).repeat(b, 1).to(self.device) # (b, Td)
        y2 = self.pos_embedder(dec_pos_tokens) # (b, Td, d)
        y3 = self.time_embedder(dec_time_features) # (b, Td, d)
        y = y1 + y2 + y3 # (b, Td, d)
        y = self.dropout(y) # (b, Td, d)
        for layer in self.decoder_layers:
            y = layer({"dec_inputs": y, "enc_outputs": z1}) # (b, Td, d)
        z2 = self.layernorm2(y) # (b, Td, d)
        
        out1 = self.hloc_output(z2) # (b, Td, 4)
        out2 = self.trend_output(z2) # (b, Td, 1)
        
        if dec_trend_target_indices is not None:
            assert dec_hloc_target_features.dim() == 3, "dec_hloc_features must have 3 dimensions (batch_size, seq_len, num_pred_hloc)"
            assert dec_hloc_target_features.shape[-1] == self.config["num_pred_hloc"], f"dec_hloc_target_features last dimension must be {self.config['num_pred_hloc']}"
            dec_hloc_target_features = dec_hloc_target_features.to(self.device)
            loss1 = torch.nn.functional.huber_loss(out1, dec_hloc_target_features) * self.config["hloc_loss_weight"]
        else:
            loss1 = 0
        if dec_trend_target_indices is not None:
            assert dec_trend_target_indices.dim() == 2, "dec_trend_target_indices must have 2 dimensions (batch_size, seq_len)"
            dec_trend_target_indices = dec_trend_target_indices.to(self.device)
            loss2 = torch.nn.functional.huber_loss(out2.squeeze(-1), dec_trend_target_indices) * self.config["trend_loss_weight"]
        else:
            loss2 = 0
        
        loss = loss1 + loss2
        return {"hloc_logits": out1, "trend_logits": out2, "loss": loss, "hloc_loss": loss1.item(), "trend_loss": loss2.item()}
    
    def calc_num_params(self) -> None:
        num_params = sum([i.numel() for i in self.parameters() if i.requires_grad])
        print(f"Number of trainable parameters: {num_params}")
 
if __name__ == "__main__":
    config = {
        "embedding_dim": 64, 
        "max_seq_length": 512, 
        "num_layers": 6, 
        "num_heads": 4, 
        "dropout_prob": 0.2, 
        "num_trends": 201, 
        "num_hloc_features": 16,
        "num_time_features": 5,
        "num_pred_hloc": 4,
        "trend_loss_weight": 0.5,
        "hloc_loss_weight": 0.5
    }
    enc_hloc_features = torch.rand(size=(4, 256, 16))
    enc_trend_indices = torch.randint(low=0, high=201, size=(4, 256))
    enc_time_features = torch.rand(size=(4, 256, 5))
    dec_hloc_features = torch.rand(size=(4, 128, 16))
    dec_trend_indices = torch.randint(low=0, high=201, size=(4, 128))
    dec_time_features = torch.rand(size=(4, 128, 5))
    hloc_target_features = torch.rand(size=(4, 128, 4))
    trend_target_indices = torch.randint(low=0, high=201, size=(4, 128))
    model = Transformer(device="cuda", config=config).to("cuda")
    out = model.forward(enc_hloc_features, enc_trend_indices, enc_time_features,
                        dec_hloc_features, dec_trend_indices, dec_time_features,
                        hloc_target_features, trend_target_indices)
    print(out)
    model.calc_num_params()
    