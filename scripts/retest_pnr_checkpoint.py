from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusenet.data import make_collate_mmsa, make_datasets
from fusenet.losses import FUSENetCriterion, LossWeights
from fusenet.model import FUSENet
from train import default_raw_media_root, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retest a saved PNR-TriFuse checkpoint with saved model args.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", default=None, help="Override the data root stored in the checkpoint.")
    parser.add_argument(
        "--text-root",
        default="checkpoints/text_encoders",
        help="Directory containing packaged text encoder config/tokenizer folders.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--limit-eval-batches", type=int, default=None)
    return parser.parse_args()


def resolve_project_path(value: str | None) -> str | None:
    if not value:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def resolve_text_backbone(value: str | None, text_root: str | None) -> str | None:
    if not value:
        return value

    root = Path(text_root or "checkpoints/text_encoders")
    if not root.is_absolute():
        root = PROJECT_ROOT / root

    path = Path(value)
    names = []
    if path.name == "encoder":
        names.append(path.parent.name)
    names.append(path.name)

    for name in names:
        candidate = root / name / "encoder"
        if candidate.exists():
            return str(candidate)
        candidate = root / name
        if (candidate / "config.json").exists():
            return str(candidate)

    return resolve_project_path(value)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = dict(checkpoint.get("args", {}))
    saved_args["text_backbone"] = resolve_text_backbone(saved_args.get("text_backbone"), args.text_root)

    dataset = saved_args.get("dataset", "mosi")
    data_root = resolve_project_path(args.data_root or saved_args.get("data_root", "data"))
    data_file = resolve_project_path(saved_args.get("data_file"))
    disabled_modalities = saved_args.get("disabled_modalities", [])

    load_raw_audio = saved_args.get("audio_backbone") is not None and "audio" not in disabled_modalities
    load_raw_vision = saved_args.get("vision_backbone") is not None and "vision" not in disabled_modalities
    raw_media_root = saved_args.get("raw_media_root")
    if raw_media_root is None:
        raw_media_root = str(default_raw_media_root(data_root, dataset))
    raw_media_root = resolve_project_path(raw_media_root)
    media_cache_root = saved_args.get("media_cache_root")
    if media_cache_root is None:
        media_cache_root = str(Path(saved_args.get("output_dir", "outputs")) / "_media_cache" / dataset)
    media_cache_root = resolve_project_path(media_cache_root)

    _, _, test_set = make_datasets(
        data_root,
        dataset,
        data_file,
        raw_media_root=raw_media_root,
        media_cache_root=media_cache_root,
        load_raw_audio=load_raw_audio,
        load_raw_vision=load_raw_vision,
        audio_sample_rate=int(saved_args.get("audio_sample_rate", 16000)),
        audio_max_seconds=float(saved_args.get("audio_max_seconds", 8.0)),
        vision_num_frames=int(saved_args.get("vision_num_frames", 16)),
        vision_frame_size=int(saved_args.get("vision_frame_size", 224)),
    )

    tokenizer = None
    if saved_args.get("text_backbone"):
        tokenizer = AutoTokenizer.from_pretrained(
            saved_args["text_backbone"],
            cache_dir=resolve_project_path(saved_args.get("text_cache_dir")),
            use_fast=True,
        )
    collate_fn = make_collate_mmsa(tokenizer, int(saved_args.get("text_max_length", 50)))
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size or int(saved_args.get("batch_size", 64)),
        shuffle=False,
        num_workers=args.num_workers if args.num_workers is not None else int(saved_args.get("num_workers", 0)),
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = FUSENet(
        input_dims=checkpoint["input_dims"],
        hidden_dim=int(saved_args.get("hidden_dim", 128)),
        dropout=float(saved_args.get("dropout", 0.1)),
        disabled_modalities=disabled_modalities,
        text_model_name_or_path=saved_args.get("text_backbone"),
        text_cache_dir=resolve_project_path(saved_args.get("text_cache_dir")),
        freeze_text_encoder=bool(saved_args.get("freeze_text_backbone", False)),
        text_gradient_checkpointing=bool(saved_args.get("text_gradient_checkpointing", False)),
        audio_model_name_or_path=saved_args.get("audio_backbone"),
        audio_cache_dir=resolve_project_path(saved_args.get("audio_cache_dir")),
        freeze_audio_encoder=bool(saved_args.get("freeze_audio_backbone", False)),
        audio_gradient_checkpointing=bool(saved_args.get("audio_gradient_checkpointing", False)),
        vision_model_name_or_path=saved_args.get("vision_backbone"),
        vision_cache_dir=resolve_project_path(saved_args.get("vision_cache_dir")),
        freeze_vision_encoder=bool(saved_args.get("freeze_vision_backbone", False)),
        vision_gradient_checkpointing=bool(saved_args.get("vision_gradient_checkpointing", False)),
        vision_frame_chunk_size=saved_args.get("vision_frame_chunk_size"),
        fusion_mode=saved_args.get("fusion_mode", "text_guided_attention"),
        model_variant=saved_args.get("model_variant", "minimal"),
        disable_common=bool(saved_args.get("disable_common", False)),
        disable_pairwise=bool(saved_args.get("disable_pairwise", False)),
        disable_private=bool(saved_args.get("disable_private", False)),
        disable_noise=bool(saved_args.get("disable_noise", False)),
        disable_reconstruction=bool(saved_args.get("disable_reconstruction", False)),
        disable_cross_factor_attention=bool(saved_args.get("disable_cross_factor_attention", False)),
        disable_dynamic_fusion=bool(saved_args.get("disable_dynamic_fusion", False)),
        disable_self_guided_fusion=bool(saved_args.get("disable_self_guided_fusion", False)),
        disable_text_guided_fusion=bool(saved_args.get("disable_text_guided_fusion", False)),
        disable_type_level_fusion=bool(saved_args.get("disable_type_level_fusion", False)),
        pnr_fusion_combine=saved_args.get("pnr_fusion_combine", "mean_token_type"),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)

    weights = LossWeights(
        task=float(saved_args.get("task_weight", 1.0)),
        common=float(saved_args.get("common_weight", 0.03)),
        pair=float(saved_args.get("pair_weight", 0.03)),
        orth=float(saved_args.get("orth_weight", 0.01)),
        hsic=float(saved_args.get("hsic_weight", 0.005)),
        info=float(saved_args.get("info_weight", 0.03)),
        noise=float(saved_args.get("noise_weight", 0.005)),
        reconstruction=float(saved_args.get("reconstruction_weight", 0.05)),
        kl=float(saved_args.get("kl_weight", 0.0005)),
    )
    criterion = FUSENetCriterion(
        weights,
        orth_mode=saved_args.get("orth_mode", "selective"),
        loss_weighting=saved_args.get("loss_weighting", "uncertainty"),
    ).to(device)

    metrics, losses = evaluate(
        model,
        criterion,
        test_loader,
        device,
        dataset,
        float(saved_args.get("label_min", -1.0 if dataset == "simsv2" else -3.0)),
        float(saved_args.get("label_max", 1.0 if dataset == "simsv2" else 3.0)),
        bool(saved_args.get("fp16", False)),
        args.limit_eval_batches,
    )

    labels = np.asarray(test_set.labels).reshape(-1)
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset": dataset,
        "model_variant": saved_args.get("model_variant", "minimal"),
        "pnr_fusion_combine": saved_args.get("pnr_fusion_combine", "mean_token_type"),
        "batch_size": args.batch_size or int(saved_args.get("batch_size", 64)),
        "support": {
            "test": int(labels.size),
            "zero_labels": int((labels == 0).sum()),
            "no0": int((labels != 0).sum()),
        },
        "test": metrics,
        "test_losses": losses,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
