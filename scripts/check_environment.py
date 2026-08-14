from __future__ import annotations

import importlib
import sys


PACKAGES = ["torch", "numpy", "scipy", "sklearn", "tqdm", "huggingface_hub", "gdown", "mmsdk", "h5py"]


def main() -> None:
    print("python", sys.version.replace("\n", " "))
    for package in PACKAGES:
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "unknown")
        print(f"{package} {version}")
    import torch

    print("torch_cuda", torch.cuda.is_available())
    print("torch_cuda_version", torch.version.cuda)


if __name__ == "__main__":
    main()
