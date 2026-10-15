import numpy as np


def sequence_to_image(s, rng, H=32, W=32, r=3, sigma=2.0, tau=0.05):
    """
    由有顺序依赖的序列控制生成目标图
    """
    L = len(s)

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
    return x.astype(np.float32)
