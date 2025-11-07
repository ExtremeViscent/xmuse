import numpy as np
import torch
from pedalboard import load_plugin
from mido import Message
import json

def _to_numpy_mono(x):
    x = np.asarray(x, dtype=np.float32)  # (channels, samples) or (samples,)
    if x.ndim == 2:                      # pedalboard 多为 (channels, n)
        x = x.mean(axis=0)               # 转单声道
    return x

def _rms(x, eps=1e-8):
    return float(np.sqrt(np.mean(np.square(x)) + eps))

def _loudness_match(x, y, target='first', eps=1e-8):
    """把 y 调整到与 x 相同的 RMS；target='first' 表示 y 对齐到 x 的响度"""
    rms_x, rms_y = _rms(x, eps), _rms(y, eps)
    if target == 'first' and rms_y > eps:
        y = y * (rms_x / rms_y)
    return x, y

def _log_stft_L2(x, y, n_fft=2048, hop=512, eps=1e-6):
    # 用 torch 做 STFT，返回 log 幅度谱的 L2
    tx = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # [1, T]
    ty = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    X = torch.stft(tx, n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft), return_complex=True).abs()+eps
    Y = torch.stft(ty, n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft), return_complex=True).abs()+eps
    X = torch.log(X)
    Y = torch.log(Y)
    return float(torch.mean((X - Y)**2))

def _cosine_sim(x, y, eps=1e-8):
    x = x.astype(np.float32); y = y.astype(np.float32)
    num = float(np.dot(x, y))
    den = float(np.linalg.norm(x) * np.linalg.norm(y) + eps)
    return num / den

def _apply_raw_values(inst, raw_param_dict):
    """把保存的 raw_value 回写到插件参数；忽略不存在的键"""
    params = inst.parameters  # dict[str, param_obj]
    hit, miss = 0, 0
    for k, v in raw_param_dict.items():
        if k in params:
            try:
                params[k].raw_value = v
                hit += 1
            except Exception:
                miss += 1
        else:
            miss += 1
    return hit, miss

def _render(inst, note=60, dur=2.0, sr=48000):
    midi = [Message("note_on", note=note), Message("note_off", note=note, time=2)]
    audio = inst(midi, duration=dur, sample_rate=sr)
    return _to_numpy_mono(audio)

def compare_audio(
    param_dict1: dict,
    param_dict2: dict,
    instrument=None,
    plugin_path: str = None,
    plugin_name: str = None,
    note: int = 60,
    duration: float = 2.0,
    sample_rate: int = 48000,
    to_mono: bool = True,
    loudness_match: bool = True,
):
    """
    param_dict{1,2}: 形如 {'param_name': raw_value, ...}
    - 你可以传入已有的 instrument；否则给 plugin_path 自动加载。
    - 返回: 两段音频与若干对比指标。
    """
    # 1) 准备插件实例
    close_after = False
    if instrument is None:
        assert plugin_path is not None, "未提供 instrument，也未提供 plugin_path。"
        instrument = load_plugin(plugin_path, plugin_name=plugin_name) if plugin_name else load_plugin(plugin_path)
        close_after = True

    # 2) 渲染音频1
    hits1, miss1 = _apply_raw_values(instrument, param_dict1)
    audio1 = _render(instrument, note=note, dur=duration, sr=sample_rate)

    # 3) 渲染音频2
    hits2, miss2 = _apply_raw_values(instrument, param_dict2)
    audio2 = _render(instrument, note=note, dur=duration, sr=sample_rate)

    # 4) 处理为单声道并响度对齐
    if to_mono:
        audio1 = _to_numpy_mono(audio1)
        audio2 = _to_numpy_mono(audio2)
    if loudness_match:
        audio1, audio2 = _loudness_match(audio1, audio2, target='first')

    # 5) 指标
    # 波形均方误差
    min_len = min(len(audio1), len(audio2))
    a1 = audio1[:min_len]; a2 = audio2[:min_len]
    mse = float(np.mean((a1 - a2) ** 2))
    cos = _cosine_sim(a1, a2)
    stft_l2 = _log_stft_L2(a1, a2)

    # 6) 清理
    if close_after:
        try:
            instrument.close()  # 某些宿主支持
        except Exception:
            pass

    return {
        "audio1": a1,            # np.float32
        "audio2": a2,            # np.float32
        "metrics": {
            "waveform_mse": mse,
            "waveform_cosine": cos,
            "log_stft_L2": stft_l2,
        },
        "applied_params": {
            "p1_hit": hits1, "p1_miss": miss1,
            "p2_hit": hits2, "p2_miss": miss2,
        },
        "meta": {
            "note": note, "duration": duration, "sample_rate": sample_rate,
            "to_mono": to_mono, "loudness_match": loudness_match,
        }
    }

if __name__ == "__main__":
    # 两组参数（raw_value 字典），可以从你前面 dump 出来的 param_values 里拿
    with open("params_res.json", "r") as f:
        params_data = json.load(f)
    p1 = params_data["original_params"]
    p2 = params_data["output_params"]

    res = compare_audio(
        p1, p2,
        plugin_path=r"C:\\Program Files\\Common Files\\vst3\\Serum2.vst3\\Contents\\x86_64-win\\Serum2.vst3",
        plugin_name="Serum 2",
        note=60, duration=2.0, sample_rate=48000,
        to_mono=True, loudness_match=True,
    )

    print(res["metrics"])
    # {'waveform_mse': ..., 'waveform_cosine': ..., 'log_stft_L2': ...}

    import soundfile as sf

    # 保存 audio1 和 audio2 到本地
    sf.write("audio1.wav", res["audio1"], samplerate=48000)
    sf.write("audio2.wav", res["audio2"], samplerate=48000)