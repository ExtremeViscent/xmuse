import torch
import math
import json

from data.dm_dataloader import ParamMeta, ParamAudioDataset

# ---------- util ----------
def _as_like(vec_or_scalar, x):
    """
    把 [B] 或标量 reshape 成与 x 可广播的形状：
    [B, 1, 1, 1, ...] 维度数 = x.ndim - 1
    """
    if torch.is_tensor(vec_or_scalar) and vec_or_scalar.ndim == 0:
        vec_or_scalar = vec_or_scalar.unsqueeze(0)  # -> [1]
    dims = (1,) * (x.ndim - 1)
    return vec_or_scalar.view(-1, *dims)

# ---------- schedule: cosine 累乘 alpha ----------
def make_cosine_schedule(T):
    s = 0.008
    steps = torch.arange(T + 1, dtype=torch.float32)
    f = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** 2
    a = (f / f[0])  # alphas_cumprod
    return a

# ---------- 从网络输出统一还原 (x0_hat, eps_hat) ----------
def predict_x0_eps(x_t, t, net_out, alphas_cumprod, pred_type: str):
    """
    pred_type in {'eps','x0','v'}
    x_t: [B, D] 或 [B, C, H, W]
    t:   [B] 的 long 索引
    net_out: 与 x_t 同 shape（当 pred_type='v'/'eps'/'x0' 时分别输出 v / eps / x0）
    """
    a_t = alphas_cumprod[t]           # [B]
    sA  = _as_like(a_t.sqrt(), x_t)   # [B, 1, ...]
    sM  = _as_like((1.0 - a_t).sqrt(), x_t)

    if pred_type == 'eps':
        eps_hat = net_out
        x0_hat  = (x_t - sM * eps_hat) / (sA + 1e-12)
    elif pred_type == 'x0':
        x0_hat  = net_out
        eps_hat = (x_t - sA * x0_hat) / (sM + 1e-12)
    elif pred_type == 'v':
        # 线性解：  x0 = sA * x_t - sM * v ;  eps = sM * x_t + sA * v
        v       = net_out
        x0_hat  = sA * x_t - sM * v
        eps_hat = sM * x_t + sA * v
    else:
        raise ValueError("pred_type must be 'eps'|'x0'|'v'")
    return x0_hat, eps_hat

# ---------- Classifier-Free Guidance ----------
@torch.no_grad()
def cfg_forward(net, x, t, cond, guidance_scale=1.0, pred_type='eps'):
    """
    约定：net(x, t, cond, pred_type=...) -> 与 x 同 shape 的张量
    - cond=None 表示无条件分支
    - 不做任何奇怪的 [:,0,:] 切片，输出 shape 与模型保持一致
    """

    # 两次前向：无条件 + 有条件
    zero_cond = torch.zeros_like(cond)
    uncond = net(x, t, zero_cond)[:, 0, :]  # 仅取 eps/x0/v 通道
    conded = net(x, t, cond)[:, 0, :]  # 仅取 eps/x0/v 通道
    return uncond + guidance_scale * (conded - uncond)

# ---------- DDIM-ODE (η=0) 采样：确定式 ----------
@torch.no_grad()
def sample_ddim_ode(model, shape, steps=50, guidance_scale=3.5, cond=None,
                    pred_type='eps', device='cuda', dtype=torch.float32):
    """
    model(x, t, cond, pred_type=...) -> 与 x 同 shape 的预测（eps/x0/v）
    shape: e.g. [B, D] 或 [B, C, H, W]
    """
    B = shape[0]
    T_train = 1000
    a = make_cosine_schedule(T_train).to(device=device, dtype=dtype)  # [T+1]
    idx = torch.linspace(T_train, 0, steps + 1, dtype=torch.long, device=device)

    x = torch.randn(shape, device=device, dtype=dtype)
    for i in range(steps, 0, -1):
        t_now  = idx[i]         # 标量索引（同一批使用相同 t）
        t_prev = idx[i - 1]
        t_vec  = torch.full((B,), t_now.item(), device=device, dtype=torch.long)

        # 预测（带 CFG）
        net_out = cfg_forward(model, x, t_vec, cond, guidance_scale, pred_type=pred_type)

        # 还原 (\hat{x}_0, \hat{\epsilon})
        x0_hat, eps_hat = predict_x0_eps(x, t_vec, net_out, a, pred_type)

        # DDIM-ODE (η=0) 更新： x_{t-1} = sqrt(a_prev)*x0 + sqrt(1-a_prev)*eps
        a_prev = a[t_prev]                      # 标量
        sAprev = _as_like(a_prev.sqrt(), x)     # [B,1,...] 广播
        sMprev = _as_like((1.0 - a_prev).sqrt(), x)
        x = sAprev * x0_hat + sMprev * eps_hat

    return x

if __name__ == "__main__":
    # Load cache dataset and pick one random sample
    cache_path = "cache/param_audio_dataset.pt"
    dataset = torch.load(cache_path, weights_only=False)
    sample = dataset[500]

    from model.dit import DiT1D

    model = DiT1D(
        input_dim=212, hidden_size=256, depth=8, num_heads=8,
        prompt_dim=512, prompt_dropout_prob=0.1, learn_sigma=False
    ).to("cuda")

    checkpoint_path = "checkpoints/step_0020000_final.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint['ema'], strict=True)
    model.eval()

    shape = (1, 212)
    sampled = sample_ddim_ode(
        model,
        shape,
        steps=50,
        guidance_scale=1.0,
        cond=sample['features'].to("cuda").unsqueeze(0),
        pred_type='v',
        device="cuda",
        dtype=torch.float32
    )
    print("Sampled shape:", sampled.shape)

    from functools import partial
    decode_fn = partial(dataset.param_meta.param_decode, dataset.sorted_param_keys)
    print(sampled)
    print(sample['params'])
    exit(0)

    output_config = decode_fn(sampled.squeeze(0).cpu())
    baseline_config = decode_fn(sample['params'])

    print("Output config:", output_config)
    print("Original config:", baseline_config)

    # Save output and original params
    with open("params_res.json", "w") as f:
        json.dump({
            "output_params": output_config,
            "original_params": baseline_config
        }, f, indent=4)
