from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusenet.data import collate_mmsa, make_datasets
from fusenet.metrics import regression_report
from fusenet.model import FUSENet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved FUSE-Net checkpoint.")
    parser.add_argument("--dataset", choices=["mosi", "mosei", "simsv2"], required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: FUSENet, loader: DataLoader, device: torch.device, dataset: str) -> dict[str, float]:
    model.eval()
    preds = []
    truths = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        preds.append(out["pred"].detach().cpu().numpy())
        truths.append(batch["label"].detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(truths, axis=0)
    return regression_report(pred, true, dataset, -3.0, 3.0)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    train_set, valid_set, test_set = make_datasets(args.data_root, args.dataset, args.data_file)
    input_dims = train_set.feature_dims()
    print(json.dumps(
        {
            "dataset": args.dataset,
            "input_dims": input_dims,
            "split_sizes": {
                "train": len(train_set),
                "valid": len(valid_set),
                "test": len(test_set),
            },
        },
        ensure_ascii=False,
        indent=2,
    ))

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_mmsa,
        pin_memory=device.type == "cuda",
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    model = FUSENet(
        input_dims=input_dims,
        hidden_dim=int(saved_args.get("hidden_dim", 128)),
        dropout=float(saved_args.get("dropout", 0.1)),
        disabled_modalities=saved_args.get("disabled_modalities", []),
        fusion_mode=saved_args.get("fusion_mode", "text_guided_attention"),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    metrics = evaluate(model, test_loader, device, args.dataset)
    output = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "metrics": metrics,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
