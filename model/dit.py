# DiT-1D: [B,D] -> [B,S=D,H] + prompt[B,2,512]
import math
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp


def modulate(x, shift, scale):
    # x: (B, S, H), shift/scale: (B, H)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------- Timestep ----------------
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) *
                          torch.arange(0, half, device=t.device, dtype=torch.float32) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):  # (B,)
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


# ---------------- Prompt ----------------
class PromptEmbedder(nn.Module):
    """
    prompt: (B, P=2, E=512) -> (B, H)
    支持 classifier-free guidance：训练期按概率置零；或用 force_drop_mask 强制置零。
    """
    def __init__(self, prompt_dim: int, hidden_size: int, dropout_prob: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(prompt_dim, hidden_size, bias=True)
        self.dropout_prob = float(dropout_prob)

    def forward(self, prompt, train: bool, force_drop_mask=None):
        h = self.proj(prompt).mean(dim=1)  # (B,H)，对 P=2 做平均池化
        if (self.dropout_prob > 0 and train) or (force_drop_mask is not None):
            if force_drop_mask is None:
                drop = torch.rand(h.shape[0], device=h.device) < self.dropout_prob
            else:
                drop = force_drop_mask.to(torch.bool).to(h.device)
            h = torch.where(drop[:, None], torch.zeros_like(h), h)
        return h


# ---------------- 1D sin-cos PosEnc ----------------
def get_1d_sincos_pos_embed(embed_dim, length):
    assert embed_dim % 2 == 0
    pos = np.arange(length, dtype=np.float32)
    omega = 1.0 / (10000 ** (np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)))
    out = np.einsum('m,d->md', pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1).astype(np.float32)  # (S,H)
    return emb


# ---------------- Blocks ----------------
class DiTBlock(nn.Module):
    """adaLN-Zero 调制的块"""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp   = Mlp(in_features=hidden_size, hidden_features=mlp_hidden,
                         act_layer=approx_gelu, drop=0)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):  # x:(B,S,H), c:(B,H)
        s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = self.adaLN(c).chunk(6, dim=1)
        x = x + g_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), s_msa, sc_msa))
        x = x + g_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), s_mlp, sc_mlp))
        return x


class FinalPerToken(nn.Module):
    """逐 token 输出： (B,S,H) -> (B,S,out_channels)"""
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(hidden_size, out_channels, bias=True)
        self.ada  = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):  # x:(B,S,H), c:(B,H)
        sh, sc = self.ada(c).chunk(2, dim=1)
        x = modulate(self.norm(x), sh, sc)
        return self.proj(x)  # (B,S,C_out)


# ---------------- Core: 1D DiT ----------------
class DiT1D(nn.Module):
    """
    输入：x ∈ R^{B×D}（每一维当作一个 token，序列长度 S=D）
         prompt ∈ R^{B×2×512}
         t ∈ Z^{B}
    输出： (B, out_channels, D)，其中 out_channels = 2 if learn_sigma else 1
    """
    def __init__(
        self,
        input_dim: int,                 # D
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        prompt_dim: int = 512,
        prompt_dropout_prob: float = 0.0,
        learn_sigma: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.learn_sigma = learn_sigma
        self.in_channels = 1
        self.out_channels = 2 if learn_sigma else 1

        # 将标量升维为 token：共享 Linear(1->H)
        self.value_proj = nn.Linear(1, hidden_size, bias=True)

        # 条件（时间 + prompt）
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.p_embedder = PromptEmbedder(prompt_dim, hidden_size, prompt_dropout_prob)

        # 固定 1D sin-cos 位置编码 (S=D)
        self.pos_embed = nn.Parameter(torch.zeros(1, input_dim, hidden_size), requires_grad=False)

        # Transformer 堆叠
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        # 逐 token 输出
        self.final = FinalPerToken(hidden_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)

        # 位置编码
        pe = get_1d_sincos_pos_embed(self.hidden_size, self.input_dim)  # (S,H)
        self.pos_embed.data.copy_(torch.from_numpy(pe).unsqueeze(0))     # (1,S,H)

        # t/p 投影
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.p_embedder.proj.weight, std=0.02)
        if self.p_embedder.proj.bias is not None:
            nn.init.constant_(self.p_embedder.proj.bias, 0)

        # adaLN Zero（与原 DiT 保持）
        for blk in self.blocks:
            nn.init.constant_(blk.adaLN[-1].weight, 0)
            nn.init.constant_(blk.adaLN[-1].bias, 0)
        nn.init.constant_(self.final.ada[-1].weight, 0)
        nn.init.constant_(self.final.ada[-1].bias, 0)
        nn.init.constant_(self.final.proj.weight, 0)
        nn.init.constant_(self.final.proj.bias, 0)

    def forward(self, x, t, prompt, force_drop_mask=None):
        """
        x: (B, D)     # 建议外部已做过区间归一与 probit
        t: (B,)
        prompt: (B, 2, 512)
        force_drop_mask: (B,) 可选；1 表示无条件（CFG 用）
        """
        B, D = x.shape
        assert D == self.input_dim

        # [B,D] -> [B,D,1] -> [B,D,H]
        tokens = self.value_proj(x.unsqueeze(-1))
        tokens = tokens + self.pos_embed

        t_emb = self.t_embedder(t)                              # (B,H)
        p_emb = self.p_embedder(prompt, self.training, force_drop_mask)  # (B,H)
        c = t_emb + p_emb                                       # (B,H)

        for blk in self.blocks:
            tokens = blk(tokens, c)                             # (B,S,H)

        y = self.final(tokens, c)                               # (B,S,C_out)
        y = y.transpose(1, 2)                                   # (B,C_out,D)
        return y

    def forward_with_cfg(self, x, t, prompt, cfg_scale: float):
        """
        复制前半 batch 作为 cond/uncond 分支，通过置零 prompt 实现 CFG。
        """
        half = x[: len(x)//2]
        x_combined = torch.cat([half, half], dim=0)
        N = x_combined.size(0)

        force_drop = torch.zeros(N, device=x.device, dtype=torch.long)
        force_drop[N//2:] = 1

        prompt_half = prompt[: len(prompt)//2]
        prompt_combined = torch.cat([prompt_half, prompt_half], dim=0)

        out = self.forward(x_combined, t, prompt_combined, force_drop_mask=force_drop)  # (N,C_out,D)

        eps, rest = out[:, :1], out[:, 1:]  # 仅在 eps 通道上做 CFG；若想全部通道一起做，可直接 eps=out; rest=None
        cond_eps, uncond_eps = torch.split(eps, len(eps)//2, dim=0)
        guided = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        return torch.cat([guided, rest], dim=1)


# ---------- 快速配置 ----------
def DiT1D_S(**kwargs):
    return DiT1D(hidden_size=128, depth=8, num_heads=4, **kwargs)

def DiT1D_B(**kwargs):
    return DiT1D(hidden_size=160, depth=8, num_heads=5, **kwargs)

def DiT1D_L(**kwargs):
    return DiT1D(hidden_size=192, depth=8, num_heads=6, **kwargs)

def DiT1D_XL(**kwargs):
    return DiT1D(hidden_size=256, depth=8, num_heads=8, **kwargs)

if __name__ == "__main__":
    # 统计参数量的函数
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    model = DiT1D(
        input_dim=212,
        hidden_size=384,
        depth=6,
        num_heads=6,
        prompt_dim=512,
        prompt_dropout_prob=0.1,
        learn_sigma=True,
    )

    print(f"DiT-1D model parameters: {count_parameters(model)/1e6:.2f}M")

