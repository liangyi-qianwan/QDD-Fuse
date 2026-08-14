from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def build_split(n: int, seq_len: int, rng: np.random.Generator) -> dict:
    text = rng.normal(size=(n, seq_len, 16)).astype(np.float32)
    audio = rng.normal(size=(n, seq_len, 8)).astype(np.float32)
    vision = rng.normal(size=(n, seq_len, 10)).astype(np.float32)
    signal = 0.5 * text[:, :, 0].mean(axis=1) + 0.25 * audio[:, :, 0].mean(axis=1) + 0.25 * vision[:, :, 0].mean(axis=1)
    labels = np.clip(signal * 3.0, -3.0, 3.0).astype(np.float32)
    return {
        "text": text,
        "audio": audio,
        "vision": vision,
        "regression_labels": labels,
        "id": np.asarray([f"toy_{i}" for i in range(n)]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny MMSA-style pickle for smoke tests.")
    parser.add_argument("--output", default="data/toy_mosi.pkl")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    data = {
        "train": build_split(32, 12, rng),
        "valid": build_split(12, 12, rng),
        "test": build_split(12, 12, rng),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(data, f)
    print(output)


if __name__ == "__main__":
    main()
