import torch
from objs import Sphere, Plane,Cylinder
from func import generate_rays, dot, norm, reflect
from config import device
from setting import O,light_sources

import matplotlib.pyplot as plt
import time
import numbers
from matplotlib.animation import FuncAnimation
import cv2
from functools import reduce
import numpy as np
def raytrace(O, D, scene, bounce=0):
    distances = [obj.intersect(O, D) for obj in scene]
    min_dist = torch.stack(distances).min(dim=0).values

    color = torch.zeros(O.shape[:-1] + (3,), device=device)
    
    for i, obj in enumerate(scene):
        mask = (distances[i] == min_dist) & (min_dist < torch.inf)
        mask = mask.squeeze(-1)  # 确保 mask 的形状为 [h, w]

        if not mask.any():
            continue

        color[mask] = obj.shade(O[mask], D[mask], min_dist[mask], scene, bounce)

    return color

def render(scene, w, h, depth, im):
    def _render(frame):
        D = generate_rays(w, h)
        Os = O.expand_as(D)
        
        start = time.time()
        image = raytrace(Os, D, scene)
        print(f"Rendering time: {time.time() - start:.4f}s")
        im.set_data(image.cpu().numpy())

    return _render