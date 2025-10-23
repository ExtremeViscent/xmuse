from typing import Tuple, List, Optional

import torch
import torch.nn as nn

from data.dataloader import RandomParamDataset
from .tokenizer import ContinuousTokenizer, DiscreteTokenizer


class ContinuousReconHead(nn.Module):
    """
    把每个连续token隐表示 -> (mu, logvar) 供NLL
    输出 shape: (B, n_cont, 2) 其中[...,0]=mu, [...,1]=logvar
    """
    def __init__(self, dim: int):
        super().__init__()
        self.head = nn.Linear(dim, 2)

    def forward(self, h_cont: torch.Tensor):
        # h_cont: (B, n_cont, dim)
        return self.head(h_cont)  # (B, n_cont, 2)


class DiscreteMultiHead(nn.Module):
    """
    43个变量，每个变量一个线性层：Dim -> b_i
    forward返回: List[Tensor], len = n_disc，每个 logits_i 形状 (B, b_i)
    """
    def __init__(self, dim: int, cat_sizes: List[int], tie_with_tables: Optional[List[nn.Embedding]] = None):
        super().__init__()
        self.cat_sizes = cat_sizes
        self.n_disc = len(cat_sizes)
        self.heads = nn.ModuleList([nn.Linear(dim, b_i) for b_i in cat_sizes])

        # 可选：分类头与tokenizer的embedding表绑定权重
        if tie_with_tables is not None:
            assert len(tie_with_tables) == self.n_disc
            for i in range(self.n_disc):
                self.heads[i].weight = tie_with_tables[i].weight  # 共享权重（注意保持dim一致）

    def forward(self, h_disc: torch.Tensor):
        # h_disc: (B, n_disc, dim), 第 i 个位置对应第 i 个变量
        logits_list = []
        for i in range(self.n_disc):
            logits_i = self.heads[i](h_disc[:, i, :])   # (B, b_i)
            logits_list.append(logits_i)
        return logits_list  # List[(B, b_i)]


class TransformerVAE(nn.Module):
    def __init__(
        self,
        cont_tokens: int,             # = 184
        cat_sizes: List[int],         # 长度=43, 每个b_i
        dim: int = 512,
        heads: int = 8,
        depth_enc: int = 8,
        depth_dec: int = 8,
        tie_disc_weights: bool = True
    ):
        super().__init__()
        self.cont_tokens = cont_tokens
        self.cat_sizes   = cat_sizes
        self.disc_tokens = len(cat_sizes)
        self.seq_len     = cont_tokens + self.disc_tokens
        self.dim         = dim

        # tokenizers
        self.cont_tok = ContinuousTokenizer(n_tokens=self.cont_tokens, dim=dim)
        self.disc_tok = DiscreteTokenizer(cat_sizes=self.cat_sizes, dim=dim)

        # positional enc
        self.pos_enc_enc = nn.Parameter(torch.randn(1, self.seq_len, dim) * 0.02)
        self.pos_enc_dec = nn.Parameter(torch.randn(1, self.seq_len, dim) * 0.02)

        # encoder / decoder
        enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth_enc)

        self.to_mu     = nn.Linear(dim, dim)
        self.to_logvar = nn.Linear(dim, dim)

        dec_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=depth_dec)

        # heads
        self.cont_head = ContinuousReconHead(dim=dim)
        self.disc_head = DiscreteMultiHead(
            dim=dim,
            cat_sizes=self.cat_sizes,
            tie_with_tables=(self.disc_tok.tables if tie_disc_weights else None)
        )

    # VAE encode
    def encode(self, x_cont: torch.Tensor, x_cat: torch.Tensor):
        """
        x_cont: (B, cont_tokens) 连续参数（建议先做min-max到[-1,1]或其他归一化）
        x_cat : (B, disc_tokens)  每列是类别索引 [0..b_i-1]
        """
        x_c = self.cont_tok(x_cont)                         # (B, cont_tokens, dim)
        x_d = self.disc_tok(x_cat)                          # (B, disc_tokens, dim)
        x   = torch.cat([x_c, x_d], dim=1)                  # (B, seq_len, dim)

        h   = self.encoder(x + self.pos_enc_enc)            # (B, seq_len, dim)
        mu, logvar = self.to_mu(h), self.to_logvar(h)       # (B, seq_len, dim)
        eps = torch.randn_like(mu)
        z   = mu + (0.5 * logvar).exp() * eps               # reparam
        return z, mu, logvar

    # VAE decode -> 连续(μ,logσ²)，离散logits列表
    def decode(self, z: torch.Tensor):
        """
        输入:
          z: (B, seq_len, dim)
        输出:
          cont_params: (B, cont_tokens, 2)  # [...,0]=mu, [...,1]=logvar
          disc_logits_list: List[Tensor]    # 长度=disc_tokens, 每个 (B, b_i)
        """
        h = self.decoder(z + self.pos_enc_dec)              # (B, seq_len, dim)

        h_cont = h[:, :self.cont_tokens, :]                 # (B, cont, dim)
        h_disc = h[:, self.cont_tokens:, :]                 # (B, disc, dim)

        cont_params = self.cont_head(h_cont)                # (B, cont, 2)
        disc_logits_list = self.disc_head(h_disc)           # List[(B, b_i)]
        return cont_params, disc_logits_list

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor):
        z, mu, logvar = self.encode(x_cont, x_cat)
        cont_params, disc_logits_list = self.decode(z)
        return cont_params, disc_logits_list, mu, logvar
    

def get_vae_model(dataset: RandomParamDataset, name: str):
    # Only default model for now
    if name == "default":
        model = TransformerVAE(
            cont_tokens=dataset.num_cont_params,
            cat_sizes=dataset.cat_sizes,
            dim=512,
            heads=8,
            depth_enc=8,
            depth_dec=8,
            tie_disc_weights=True
        )
    else:
        raise ValueError(f"Unknown VAE model name: {name}")
    return model
