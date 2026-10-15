# %%
"""
本文件实现有顺序依赖的序列控制生成目标图的功能
"""

import matplotlib.pyplot as plt
import numpy as np

# %%
# ===== 参数设置 =====
L = 8  # 序列长度
H, W = 32, 32  # 图像高度、宽度
r = 3  # 高斯斑大小
sigma = 2.0  # 高斯方差
tau = 0.05  # 噪声水平

# rng = np.random.default_rng(seed=42)
rng = np.random.default_rng()
s = rng.uniform(-1.0, 1.0, L)

p = np.zeros((L + 1, 2))
t = np.zeros(L + 1)

p[0] = np.array([H / 2.0, W / 2.0])
t[0] = 0

for k in range(1, L + 1):
    t[k] = t[k - 1] + 0.51 * np.pi * s[k - 1]
    dx = np.cos(t[k])
    dy = np.sin(t[k])
    p[k] = p[k - 1] + r * np.array([dy, dx])

u, v = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
G = np.zeros((H, W))
for k in range(1, L + 1):
    py, px = p[k]
    G += np.exp(-((u - py) ** 2 + (v - px) ** 2) / (2 * sigma**2))

Gmax, Gmin = G.max(), G.min()
x = 2 * (G - Gmin) / (Gmax - Gmin) - 1
x += rng.normal(0, tau, (H, W))

plt.figure(figsize=(6, 6))
plt.imshow(x, origin="upper")
plt.show()
