from config import device
import torch

O = torch.tensor([0, 0.2, -1], device=device)
TO = torch.tensor([0, 0, 1], device=device)
ambient=0.4
depth =1
w, h = 1000, 1000
light_sources = [
    {'position': torch.tensor([2.0, 2.0, -1.0], device=device), 'intensity': 0.5},
    {'position': torch.tensor([-2.0, 2.0, -1.0], device=device), 'intensity': 0.3},
    {'position': torch.tensor([2.0, 2.0, 1.0], device=device), 'intensity': 0.2}
]