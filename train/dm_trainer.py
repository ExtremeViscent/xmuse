# trainer_dit1d_f32.py
import os, math, time
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from model.dit import DiT1D  # DiT1D(input_dim, hidden_size, depth, num_heads, prompt_dim, prompt_dropout_prob, ...)
from data.dm_dataloader import ParamMeta, ParamAudioDataset

# ---------------- diffusion schedule ----------------
def make_beta_schedule(T: int, schedule: str = "cosine"):
    if schedule == "linear":
        beta_1, beta_T = 1e-4, 2e-2
        betas = torch.linspace(beta_1, beta_T, T)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        return betas, alphas_bar
    elif schedule == "cosine":
        # Improved DDPM (Nichol & Dhariwal, 2021)
        def f(t):
            s = 0.008
            return torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        ts = torch.linspace(0, 1, T + 1)
        alphas_bar = f(ts) / f(ts[0])
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        betas = betas.clamp(1e-8, 0.999)
        return betas, torch.cumprod(1.0 - betas, dim=0)
    else:
        raise ValueError(f"unknown schedule: {schedule}")

def cosine_with_warmup(warmup_steps: int, total_steps: int):
    def fn(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return fn

# ---------------- EMA ----------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.shadow.items():
            if k in msd:
                msd[k].copy_(v)

# ---------------- helpers ----------------
def sample_timesteps(B, T, device):
    return torch.randint(low=0, high=T, size=(B,), device=device)

def compute_x_t(x0, eps, a_bar_t):
    return a_bar_t.sqrt() * x0 + (1.0 - a_bar_t).sqrt() * eps

def target_v(x0, eps, a_bar_t):
    return a_bar_t.sqrt() * eps - (1.0 - a_bar_t).sqrt() * x0

def prepare_batch(batch, device):
    x0 = batch["params"].to(device=device, dtype=torch.float32)     # (B, D)
    prompt = batch["features"].to(device=device, dtype=torch.float32)  # (B, 2, 512)
    return x0, prompt

# ---------------- config ----------------
@dataclass
class TrainConfig:
    # data
    batch_size: int = 512               # 256–1024
    num_workers: int = 0
    # diffusion
    T: int = 1000
    schedule: str = "cosine"
    pred_type: Literal["v", "eps"] = "v"
    # optim
    lr: float = 1e-4
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    # lr schedule
    warmup_steps: int = 3000
    total_steps: int = 100_000          # 先 50k–100k；稳定后可到 150k
    # ema
    ema_decay: float = 0.9999
    # misc
    ckpt_dir: str = "./checkpoints"
    log_interval: int = 100
    val_interval: int = 2000
    save_interval: int = 5000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- trainer ----------------
class Trainer:
    def __init__(self, model: nn.Module, train_set, val_set, cfg: TrainConfig):
        self.model = model.to(cfg.device).float()   # 强制 float32
        self.cfg = cfg

        self.train_loader = DataLoader(
            train_set, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True, drop_last=True
        )
        self.val_loader = DataLoader(
            val_set, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True, drop_last=False
        )

        self.optimizer = AdamW(self.model.parameters(), lr=cfg.lr,
                               betas=cfg.betas, weight_decay=cfg.weight_decay)
        self.lr_sched = LambdaLR(self.optimizer,
                                 lr_lambda=cosine_with_warmup(cfg.warmup_steps, cfg.total_steps))

        betas, alphas_bar = make_beta_schedule(cfg.T, cfg.schedule)
        self.betas = betas.to(cfg.device)
        self.alphas_bar = alphas_bar.to(cfg.device)

        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        self.step = 0

        # EMA
        self.ema = EMA(self.model, decay=cfg.ema_decay)

    def save(self, tag: str):
        path = os.path.join(self.cfg.ckpt_dir, f"step_{self.step:07d}_{tag}.pt")
        torch.save({
            "step": self.step,
            "model": self.model.state_dict(),
            "opt": self.optimizer.state_dict(),
            "sched": self.lr_sched.state_dict(),
            "ema": self.ema.shadow,
            "cfg": self.cfg.__dict__
        }, path)
        print(f"[CKPT] saved -> {path}")

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_mse, count = 0.0, 0
        for batch in self.val_loader:
            x0, prompt = prepare_batch(batch, self.cfg.device)
            B, D = x0.shape
            t = sample_timesteps(B, self.cfg.T, self.cfg.device)
            eps = torch.randn_like(x0)
            a_bar_t = self.alphas_bar[t].view(B, 1)
            x_t = compute_x_t(x0, eps, a_bar_t)

            out = self.model(x_t, t, prompt)   # (B, C_out, D); 训练期内部已做 prompt dropout=0.1

            if self.cfg.pred_type == "v":
                tgt = target_v(x0, eps, a_bar_t)
                pred = out[:, 0, :]
            else:
                tgt = eps
                pred = out[:, 0, :]

            mse = torch.mean((pred - tgt) ** 2).item()
            total_mse += mse * B
            count += B
        self.model.train()
        return total_mse / max(1, count)

    def train(self):
        self.model.train()
        last_log = time.time()

        print("Starting training...")
        # 外层循环：直到达到总步数
        with tqdm(total=self.cfg.total_steps, desc="Training Progress", ncols=100) as pbar:
            while self.step < self.cfg.total_steps:
                for batch in self.train_loader:
                    self.step += 1

                    x0, prompt = prepare_batch(batch, self.cfg.device)
                    B, D = x0.shape
                    t = sample_timesteps(B, self.cfg.T, self.cfg.device)
                    eps = torch.randn_like(x0)
                    a_bar_t = self.alphas_bar[t].view(B, 1)
                    x_t = compute_x_t(x0, eps, a_bar_t)

                    self.optimizer.zero_grad(set_to_none=True)

                    out = self.model(x_t, t, prompt)  # (B, C_out, D)

                    if self.cfg.pred_type == "v":
                        tgt = target_v(x0, eps, a_bar_t)
                        pred = out[:, 0, :]
                    else:
                        tgt = eps
                        pred = out[:, 0, :]

                    loss = torch.mean((pred - tgt) ** 2)
                    loss.backward()

                    if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

                    self.optimizer.step()
                    self.lr_sched.step()
                    self.ema.update(self.model)

                    # tqdm 动态更新
                    pbar.set_postfix({
                        "loss": f"{loss.item():.6f}",
                        "lr": f"{self.optimizer.param_groups[0]['lr']:.3e}"
                    })
                    pbar.update(1)

                    # 日志
                    if self.step % self.cfg.log_interval == 0:
                        dt = time.time() - last_log
                        last_log = time.time()
                        lr = self.optimizer.param_groups[0]["lr"]
                        print(f"[{self.step:7d}/{self.cfg.total_steps}] "
                            f"loss={loss.item():.6f} lr={lr:.3e} dt={dt:.2f}s")

                    # 验证
                    if self.step % self.cfg.val_interval == 0:
                        # 1) 备份当前训练权重
                        backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

                        # 2) 用 EMA 权重评估
                        self.ema.copy_to(self.model)
                        val_mse = self.validate()
                        print(f"[VAL] step={self.step} mse={val_mse:.6e}")

                        # 3) 恢复训练权重（避免污染训练）
                        self.model.load_state_dict(backup, strict=True)

                    # 保存
                    if self.step % self.cfg.save_interval == 0:
                        self.save("regular")

                    if self.step >= self.cfg.total_steps:
                        break

            # 训练完成后保存最终模型
            self.save("final")


# ---------------- usage example ----------------
if __name__ == "__main__":
    cache_path = "cache/param_audio_dataset.pt"
    dataset = torch.load(cache_path, weights_only=False)
    train_size = 100000
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    print(f"Train set size: {len(train_set)}, Val set size: {len(val_set)}")

    model = DiT1D(
        input_dim=212, hidden_size=256, depth=8, num_heads=8,
        prompt_dim=512, prompt_dropout_prob=0.1, learn_sigma=False
    )

    cfg = TrainConfig(
        batch_size=128, T=1000, schedule="cosine", pred_type="v",
        lr=1e-4, warmup_steps=1000, total_steps=20000,
        ema_decay=0.9999, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    trainer = Trainer(model, train_set, val_set, cfg)
    trainer.train()
