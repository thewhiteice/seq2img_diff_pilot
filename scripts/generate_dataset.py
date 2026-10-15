import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sec2img.data.synthetic import generate_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="shard_000.npz")

    args = parser.parse_args()
    S, X = generate_dataset(args.n, args.seed)

    DATA_DIR = Path(__file__).parent.parent / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(DATA_DIR/args.out, input=S, image=X)


if __name__ == "__main__":
    main()
