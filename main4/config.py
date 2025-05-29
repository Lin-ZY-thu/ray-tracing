import torch

# 配置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')