import torch
import numpy as np
from config import device
from setting import O,TO
# 张量向量运算函数
def dot(a, b):
    return torch.sum(a * b, dim=-1, keepdim=True)

def norm(v):
    mag = torch.sqrt(torch.clamp_min(dot(v, v), 1e-10))
    return v / mag

def reflect(D, N):
    return D - 2 * dot(D, N) * N

def cross(a, b):
    # 确保输入向量的形状正确
    assert a.shape[-1] == 3, "输入向量必须是三维的"
    assert b.shape[-1] == 3, "输入向量必须是三维的"
    # 计算叉积
    cross_product = torch.stack([
        a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
        a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    ], dim=-1)
    return cross_product

# 光线生成
def generate_rays(w, h, fov=60):
    aspect_ratio = w / h
    S = (-1, 1 / aspect_ratio + 0.25, 1, -1 / aspect_ratio + 0.25)
    
    x = torch.linspace(S[0], S[2], w, device=device)
    y = torch.linspace(S[1], S[3], h, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    
    Q = torch.stack([X, Y, torch.zeros_like(X)], dim=-1)
    Os = O.expand_as(Q)
    
    D = Q - Os
    return norm(D)

def raytrace(O, D, scene, bounce=0):
    distances = [obj.intersect(O, D) for obj in scene]
    min_dist = torch.stack(distances).min(dim=0).values

    color = torch.zeros(O.shape[:-1] + (3,), device=device)
    
    for i, obj in enumerate(scene):
        mask = (distances[i] == min_dist) & (min_dist < torch.inf)
        mask = mask.squeeze(-1)  # 确保 mask 的形状为 [768, 1024]

        if not mask.any():
            continue

        color[mask] = obj.shade(O[mask], D[mask], min_dist[mask], scene, bounce)

    return color

