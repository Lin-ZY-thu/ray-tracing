import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from config import device
from render import render
from scene import create_scene
from setting import w,h,depth,light_sources

import time
import cv2
from functools import reduce
import numpy as np

# 配置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 渲染参数


# 创建场景
scene = create_scene()

# 创建一个图和轴
fig, ax = plt.subplots()
im = ax.imshow(np.zeros((h, w, 3), dtype=np.uint8), interpolation='nearest')

# 主渲染函数
_render = render(scene, w, h, depth, im)

# 可视化结果
ani = FuncAnimation(fig, _render, interval=0)
plt.axis('off')
plt.show()