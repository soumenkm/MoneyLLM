import os, json, pickle, torch, wandb, tqdm
if __name__ == "__main__":
    wandb.login()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
torch.manual_seed(42)
from pathlib import Path
from dataset import StockDataset
from transformers import get_linear_schedule_with_warmup
from model import Transformer
import numpy as np
from typing import List, Tuple, Union, Any
from torch.utils.data import DataLoader, Subset, Dataset

class Trainer:
    def __init__(self, device: torch.device, config: dict):
        self.config = config
        self.device = device

        self.ds = StockDataset(pkl_file=self.config["dataset_path"], max_seq_length=self.config["model_config"]["max_seq_length"], time_format=self.config["time_format"], frac=self.config["frac"])
        self.train_ds = Subset(dataset=self.ds, indices=range(0, int(0.8 * len(self.ds))))
        self.eval_ds = Subset(dataset=self.ds, indices=range(int(0.8 * len(self.ds)), len(self.ds)))

        self.batch_size = self.config["batch_size"]
        self.num_steps = len(self.train_ds)//(self.batch_size)
        self.num_epochs = self.config["num_epochs"]
        self.project_name = "transformer_pretrain"
        os.environ["WANDB_PROJECT"] = self.project_name
        self.run_name = f"{self.config['frac']:.2f}_{self.config['initial_lr']:.1e}"
        self.output_dir = Path(Path.cwd(), f"outputs/ckpt/{self.project_name}/{self.run_name}")
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        self.model = Transformer(device=self.device, config=self.config["model_config"]).to(self.device) 
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.config['initial_lr'], weight_decay=self.config["weight_decay"], betas=self.config["adam_betas"])
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=int(0.01 * self.num_steps), 
                                                         num_training_steps=int(self.num_epochs * self.num_steps))

        self.train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=False, drop_last=True)
        self.eval_dl = DataLoader(self.eval_ds, batch_size=self.batch_size, shuffle=False, drop_last=True)
        self.wandb_log = self.config["wandb_log"]
        
        if self.wandb_log:
            wandb.init(project=self.project_name, name=self.run_name, config=self.config)
            wandb.watch(self.model, log="all")
            wandb.define_metric("train/step")
            wandb.define_metric("eval/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("eval/*", step_metric="eval/step")
    
    def _find_norm(self) -> float:
        norm = 0
        for val in self.model.parameters():
            if val.requires_grad:
                k = val.grad if val.grad is not None else torch.tensor(0.0, device=self.device)
                norm += (k ** 2).sum().item()
        norm = norm ** 0.5  
        return norm
    
    def _save_checkpoint(self, ep: int) -> None:
        checkpoint = {
            "epoch": ep, 
            "model_state": self.model.state_dict(), 
            "opt_state": self.optimizer.state_dict(),
            "config": self.config,
            "output_dir": self.output_dir,
            "project_name": self.project_name,
            "run_name": self.run_name
        }   
        checkpoint_path = Path(self.output_dir, f"ckpt_ep_{ep}.pth")            
        torch.save(checkpoint, checkpoint_path)
        print(f"[SAVE] ep: {ep}/{self.num_epochs-1}, checkpoint saved at: {checkpoint_path}")
    
    def _forward_batch(self, batch: dict, is_train: bool) -> dict:
        if is_train:
            self.model.train()
            with torch.amp.autocast(device_type=str(self.device)):
                out = self.model(**batch)
            out["hloc_logits"].requires_grad_(True)
            out["trend_logits"].requires_grad_(True)
            assert out["hloc_logits"].requires_grad == True
            assert out["trend_logits"].requires_grad == True
        else:
            self.model.eval()
            with torch.no_grad(), torch.amp.autocast(device_type=str(self.device)):
                out = self.model(**batch)       
        return out 

    def _optimize_batch(self, loss: torch.Tensor) -> None:  
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()     
        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.config["max_grad_norm"], norm_type=2.0)
        self.optimizer.step()
        self.scheduler.step()
    
    def _optimize_dataloader(self, ep: int) -> None:  
        with tqdm.tqdm(iterable=self.train_dl, desc=f"[TRAIN] ep: {ep}/{self.num_epochs-1}", total=len(self.train_dl), unit="step", colour="green") as pbar:
            for i, batch in enumerate(pbar):    
                out = self._forward_batch(batch=batch, is_train=True)
                loss = out["loss"] 
                gn = self._find_norm()
                lr = self.optimizer.param_groups[0]['lr']   
                self._optimize_batch(loss=loss)
            
                if self.wandb_log:
                    wandb.log({
                        "train/loss": loss.item(), 
                        "train/hloc_loss": out["hloc_loss"], 
                        "train/trend_loss": out["trend_loss"], 
                        "train/learning_rate": lr, 
                        "train/grad_norm": gn, 
                        "train/epoch": ep, 
                        "train/step": self.train_step})
                    self.train_step += 1
                pbar.set_postfix({"loss": f"{loss.item():.3f}", "huber_loss": f"{out['hloc_loss']:.3f}", "trend_loss": f'{out["trend_loss"]:.3f}', "lr": f"{lr:.3e}", "gn": f"{gn:.3f}"})                        
    
    def _validate_dataloader(self, ep: int) -> None:
        with tqdm.tqdm(iterable=self.eval_dl, desc=f"[VAL] ep: {ep}/{self.num_epochs-1}", total=len(self.eval_dl), unit="step", colour="green") as pbar:
            for i, batch in enumerate(pbar):    
                out = self._forward_batch(batch=batch, is_train=False)  
                loss = out["loss"]

                if self.wandb_log:
                    wandb.log({
                        "eval/loss": loss.item(), 
                        "eval/hloc_loss": out["hloc_loss"], 
                        "eval/trend_loss": out["trend_loss"], 
                        "eval/epoch": ep, 
                        "eval/step": self.eval_step})
                    self.eval_step += 1
                pbar.set_postfix({"loss": f"{loss.item():.3f}", "huber_loss": f"{out['hloc_loss']:.3f}", "CE_loss": f'{out["trend_loss"]:.3f}'})                        
    
    def train(self) -> None:
        print(self.model)
        self.model.calc_num_params()
        self.train_step = 0
        self.eval_step = 0
        for ep in range(self.num_epochs):
            self._optimize_dataloader(ep=ep)
            self._validate_dataloader(ep=ep)
            self._save_checkpoint(ep=ep)
        if self.wandb_log:
            wandb.finish()
    
    @staticmethod
    def load_model(device: torch.device, checkpoint_path: Path) -> dict:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint["config"]
        model = Transformer(device=device, config=config["model_config"])
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"], strict=False)
        
        print(f"Model loaded from checkpoint: {checkpoint_path}")
        return {"model": model, "config_data": config}

def main(device: torch.device) -> None:
    config = {
        "dataset_path": Path(Path.cwd(), "data/nifty_trend_data.pkl"),
        "time_format": "%Y-%m-%d %H:%M:%S",
        "model_config": {
            "embedding_dim": 128, 
            "max_seq_length": 120, 
            "num_layers": 12, 
            "num_heads": 8, 
            "dropout_prob": 0.2, 
            "num_trends": 201, 
            "num_hloc_features": 16,
            "num_time_features": 5,
            "num_pred_hloc": 4,
            "hloc_loss_weight": 1,
            "trend_loss_weight": 1
        },
        "num_epochs": 5, "batch_size": 16, "frac": 1.0,
        "initial_lr": 1e-5, "max_grad_norm": 10.0, "weight_decay": 0.1,
        "adam_betas": (0.95, 0.999), "num_ckpt_per_epoch": 1,
        "wandb_log": True
    }
    trainer = Trainer(device=device, config=config)
    trainer.train()
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}...")
    
    main(device=device)