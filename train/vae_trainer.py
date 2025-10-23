from typing import Tuple, List, Optional, Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from data.dataloader import RandomParamDataset
from model.simple_vae import get_vae_model

from .loss import gaussian_nll_from_params, multi_ce_loss, kl_gaussian


class VAETrainer:
    def __init__(
        self,
        config_path: str,
        save_path: str = "vae_model.pth",
        num_samples: int = 2048,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,         # 更稳；CUDA上再用 bf16/fp16 + autocast
        checkpoint_path: Optional[str] = None,
        use_autocast: bool = True,
        grad_clip: float = 1.0,
        kl_warmup_steps: int = 20000,               # KL 退火步数
    ):
        self.config_path = config_path
        self.save_path = save_path
        self.device = device
        self.dtype = dtype
        self.use_autocast = use_autocast
        self.grad_clip = grad_clip
        self.kl_warmup_steps = kl_warmup_steps
        self.global_step = 0

        # dataset
        self.dataset = RandomParamDataset(
            config_path=config_path,
            num_samples=num_samples,
            device=torch.device("cpu"),   # DataLoader 通常先放 CPU，step 再搬到 device
            dtype=torch.float32           # 原始数据保持 float32
        )

        # model
        self.model = get_vae_model(self.dataset, "default").to(device=self.device, dtype=self.dtype)

        # AMP scaler（fp16 时启用；bf16 不需要 scaler）
        self.scaler = GradScaler(enabled=(self.use_autocast and self.device.type == "cuda" and self.dtype == torch.float16))

        if checkpoint_path is not None:
            self.load_model(checkpoint_path)

    # ---------- IO ----------
    def get_dataloader(self, batch_size: int, shuffle: bool = True, num_workers: int = 0):
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=(num_workers > 0)
        )

    def save_model(self, path: Optional[str] = None):
        path = path or self.save_path
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: Optional[str] = None):
        path = path or self.save_path
        sd = torch.load(path, map_location="cpu")
        self.model.load_state_dict(sd)
        self.model.to(self.device, dtype=self.dtype)

    # ---------- utils ----------
    @staticmethod
    def _to_indices_from_onehot(one_hot: torch.Tensor) -> torch.Tensor:
        # one_hot: (B, K)
        return one_hot.argmax(dim=1)

    def _prep_batch(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回：
          x_cont:  (B, n_cont)  归一化后的连续输入（[-1,1]）
          x_cat:   (B, n_disc)  每列是类别 index
          tgt_cont:(B, n_cont)  连续目标（与 x_cont 同尺度）
          tgt_cat: (B, n_disc)  离散目标 index
        """
        # raw tensors on cpu -> device
        cont_raw = data["continuous"].to(self.device, dtype=torch.float32)  # (B, n_cont)
        disc_raw = data["categorical"].to(self.device)                      # 可能是 (B, sum b_i) one-hot 或 (B, n_disc) index

        # --- continuous: min-max到[-1,1] ---
        cont_min = self.dataset.cont_min_vals.to(self.device)
        cont_max = self.dataset.cont_max_vals.to(self.device)
        eps = 1e-6
        scale = (cont_max - cont_min).clamp_min(eps)
        x_cont = 2.0 * (cont_raw - cont_min) / scale - 1.0
        tgt_cont = x_cont.detach()  # NLL 在同尺度上重构

        # --- discrete: 兼容 one-hot 或 index ---
        if disc_raw.dim() == 2 and disc_raw.size(1) == sum(self.dataset.cat_sizes):
            # 拼接 one-hot -> 拆列
            disc_sizes = self.dataset.cat_sizes
            idx_list = []
            start = 0
            for k in disc_sizes:
                end = start + k
                idx_list.append(self._to_indices_from_onehot(disc_raw[:, start:end]))
                start = end
            x_cat = torch.stack(idx_list, dim=1)   # (B, n_disc)
        else:
            # 已经是 index 形式
            x_cat = disc_raw.to(torch.long)        # (B, n_disc)

        tgt_cat = x_cat.detach()

        # cast 到 model dtype
        x_cont = x_cont.to(self.dtype)
        # x_cat 必须是 long，不能转 dtype
        return x_cont, x_cat, tgt_cont.to(self.dtype), tgt_cat

    def _beta(self) -> float:
        # 线性退火到 1.0
        if self.kl_warmup_steps <= 0:
            return 1.0
        return min(1.0, self.global_step / float(self.kl_warmup_steps))

    # ---------- train step ----------
    def one_step(self, data: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        self.model.train()
        optimizer.zero_grad(set_to_none=True)

        x_cont, x_cat, tgt_cont, tgt_cat = self._prep_batch(data)

        use_amp = self.use_autocast and self.device.type == "cuda" and (self.dtype in (torch.float16, torch.bfloat16))
        with autocast(enabled=use_amp, dtype=self.dtype if self.dtype in (torch.float16, torch.bfloat16) else None):
            cont_params, disc_logits_list, mu, logvar = self.model(x_cont, x_cat)

            loss_cont = gaussian_nll_from_params(cont_params, tgt_cont)
            loss_disc = multi_ce_loss(disc_logits_list, tgt_cat)
            loss_kl   = kl_gaussian(mu, logvar)
            beta = self._beta()
            total_loss = loss_cont + loss_disc + beta * loss_kl

        if self.scaler.is_enabled():
            self.scaler.scale(total_loss).backward()
            if self.grad_clip is not None:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            optimizer.step()

        self.global_step += 1

        return {
            "total_loss": float(total_loss.detach().cpu()),
            "recon_loss_cont": float(loss_cont.detach().cpu()),
            "recon_loss_disc": float(loss_disc.detach().cpu()),
            "kl_loss": float(loss_kl.detach().cpu()),
            "beta": float(beta),
        }

    # ---------- train loop ----------
    def train(
        self,
        epochs: int,
        batch_size: int,
        learning_rate: float = 2e-4,
        num_workers: int = 0,
        weight_decay: float = 0.01,
        log_every: int = 50,
        save_every: Optional[int] = None,
    ):
        dataloader = self.get_dataloader(batch_size=batch_size, num_workers=num_workers)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        for epoch in range(1, epochs + 1):
            running = {"total_loss": 0.0, "recon_loss_cont": 0.0, "recon_loss_disc": 0.0, "kl_loss": 0.0}
            for it, batch in enumerate(dataloader, start=1):
                stats = self.one_step(batch, optimizer)
                for k in running:
                    running[k] += stats[k]

                if it % log_every == 0:
                    n = log_every
                    print(
                        f"Epoch {epoch} | iter {it:05d} | "
                        f"loss {running['total_loss']/n:.4f} | "
                        f"cont {running['recon_loss_cont']/n:.4f} | "
                        f"disc {running['recon_loss_disc']/n:.4f} | "
                        f"kl {running['kl_loss']/n:.4f} | beta {stats['beta']:.3f}"
                    )
                    for k in running: running[k] = 0.0

            if save_every and (epoch % save_every == 0):
                self.save_model(f"{self.save_path}.ep{epoch}")

        self.save_model()


if __name__ == "__main__":
    trainer = VAETrainer(
        config_path="config.json",
        save_path="weights/vae_model.pth",
        num_samples=2048,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        kl_warmup_steps=2000,
    )

    trainer.train(
        epochs=50,
        batch_size=64,
        learning_rate=2e-4,
        num_workers=4,
        weight_decay=0.01,
        log_every=5,
        save_every=10,
    )
