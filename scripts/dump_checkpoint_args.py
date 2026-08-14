from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump selected metadata from a PyTorch checkpoint.")
    parser.add_argument("checkpoint")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    payload = {
        "checkpoint": args.checkpoint,
        "epoch": ckpt.get("epoch"),
        "input_dims": ckpt.get("input_dims"),
        "selection_metric": ckpt.get("selection_metric"),
        "selection_score": ckpt.get("selection_score"),
        "valid": ckpt.get("valid"),
        "valid_losses": ckpt.get("valid_losses"),
        "args": ckpt.get("args", {}),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
