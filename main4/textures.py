import torch

def checker_texture(M, center, diffuse, scale=2, intensity=0.5):
    uv = (M[..., ::2] - center[::2]) * scale
    # 生成棋盘图案
    checker = ((uv[..., 0] > 0.5) ^ (uv[..., 1] > 0.5)).unsqueeze(-1)
    texture_color = (0.5 + 0.5 * checker.float()) * intensity
    return texture_color * diffuse