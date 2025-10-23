import torch
import torch.nn.functional as F
from typing import List

def gaussian_nll_from_params(cont_params: torch.Tensor, target: torch.Tensor, clamp_logvar: float = 10.0):
    """
    cont_params: (B, n_cont, 2)  [...,0]=mu, [...,1]=logvar
    target     : (B, n_cont)     目标连续值（与tokenizer一致的尺度）
    返回: 标量 loss（按token均值）
    """
    mu      = cont_params[..., 0]
    logvar  = cont_params[..., 1].clamp(min=-clamp_logvar, max=clamp_logvar)
    inv_var = torch.exp(-logvar)
    nll = 0.5 * ( (target - mu)**2 * inv_var + logvar )    # (B, n_cont)
    return nll.mean()

def multi_ce_loss(disc_logits_list: List[torch.Tensor], targets: torch.Tensor, class_weights=None):
    """
    disc_logits_list: 长度= n_disc 的列表; 第 i 个张量形状 (B, b_i)
    targets         : (B, n_disc) 每列是该变量的整型标签
    class_weights   : 可选 List[Tensor], 每个 (b_i,) 作为权重
    """
    losses = []
    for i, logits_i in enumerate(disc_logits_list):
        t_i = targets[:, i]
        if class_weights is not None and class_weights[i] is not None:
            losses.append(F.cross_entropy(logits_i, t_i, weight=class_weights[i]))
        else:
            losses.append(F.cross_entropy(logits_i, t_i))
    return torch.stack(losses).mean()

def kl_gaussian(mu: torch.Tensor, logvar: torch.Tensor):
    """
    标准VAE KL: N(mu, sigma^2) vs N(0,1)
    mu/logvar: (B, seq_len, dim)
    返回标量均值
    """
    kl = 0.5 * (mu.pow(2) + torch.exp(logvar) - logvar - 1.0)  # (B, seq_len, dim)
    return kl.mean()
