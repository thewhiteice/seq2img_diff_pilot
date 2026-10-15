import numpy as np

from .fomula import sequence_to_image


def sample_input(rng, L=8):
    return rng.uniform(-1.0, 1.0, L)


def generate_sample(rng, L=8):
    s = sample_input(rng, L)
    x = sequence_to_image(s, rng)
    return s, x


def generate_dataset(n, seed=0, L=8):
    rng = np.random.default_rng(seed=seed)
    S = np.zeros((n, L), np.float32)
    X = np.zeros((n, 32, 32), np.float32)
    for i in range(n):
        S[i], X[i] = generate_sample(rng, L)
    return S, X
