from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def _patch_transformers_dtensor_import() -> None:
    try:
        import sys
        import types
        from torch.distributed import tensor as torch_dist_tensor

        for name in ["DTensor", "DeviceMesh", "Shard", "Replicate", "Partial"]:
            if not hasattr(torch_dist_tensor, name):
                setattr(torch_dist_tensor, name, type(name, (), {}))
        module_name = "torch.distributed.tensor._utils"
        if module_name not in sys.modules:
            utils_module = types.ModuleType(module_name)

            def compute_local_shape_and_global_offset(shape, device_mesh, placements):
                return tuple(shape), tuple(0 for _ in shape)

            utils_module.compute_local_shape_and_global_offset = compute_local_shape_and_global_offset
            sys.modules[module_name] = utils_module
        placement_module_name = "torch.distributed.tensor.placement_types"
        if placement_module_name not in sys.modules:
            placement_module = types.ModuleType(placement_module_name)

            class Shard:
                def __init__(self, dim=0):
                    self.dim = dim

                def is_shard(self):
                    return True

                @staticmethod
                def local_shard_size_and_offset(size, world_size, rank):
                    shard_size = (size + world_size - 1) // world_size
                    offset = min(rank * shard_size, size)
                    return max(0, min(shard_size, size - offset)), offset

            placement_module.Shard = Shard
            sys.modules[placement_module_name] = placement_module
    except Exception:
        pass


_patch_transformers_dtensor_import()

from transformers import AutoTokenizer

from fusenet.data import make_collate_mmsa, make_datasets
from fusenet.losses import FUSENetCriterion, LossWeights
from fusenet.metrics import regression_report
from fusenet.model import FUSENet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FUSE-Net on MMSA-style multimodal sentiment data.")
    parser.add_argument("--dataset", choices=["mosi", "mosei", "simsv2"], required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--text-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--fusion-mode",
        choices=["text_guided_attention"],
        default="text_guided_attention",
    )
    parser.add_argument("--initial-freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--initial-freeze-mapping-attention-epochs", type=int, default=0)
    parser.add_argument("--unfreeze-last-n-layers", type=int, default=0)
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional checkpoint whose compatible model weights initialize this run.",
    )
    parser.add_argument("--disabled-modalities", nargs="*", default=[], choices=["text", "audio", "vision"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=808041,
        help="Seed used only for DataLoader shuffle/worker order. Defaults to 808041.",
    )
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--raw-media-root", default=None)
    parser.add_argument("--media-cache-root", default=None)
    parser.add_argument("--text-backbone", default=None)
    parser.add_argument("--text-cache-dir", default=None)
    parser.add_argument("--text-max-length", type=int, default=50)
    parser.add_argument("--freeze-text-backbone", action="store_true")
    parser.add_argument("--text-gradient-checkpointing", action="store_true")
    parser.add_argument("--audio-backbone", default=None)
    parser.add_argument("--audio-cache-dir", default=None)
    parser.add_argument("--audio-lr", type=float, default=None)
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--audio-max-seconds", type=float, default=8.0)
    parser.add_argument("--freeze-audio-backbone", action="store_true")
    parser.add_argument("--audio-gradient-checkpointing", action="store_true")
    parser.add_argument("--vision-backbone", default=None)
    parser.add_argument("--vision-cache-dir", default=None)
    parser.add_argument("--vision-lr", type=float, default=None)
    parser.add_argument("--vision-num-frames", type=int, default=16)
    parser.add_argument("--vision-frame-size", type=int, default=224)
    parser.add_argument("--vision-frame-chunk-size", type=int, default=None)
    parser.add_argument("--freeze-vision-backbone", action="store_true")
    parser.add_argument("--vision-gradient-checkpointing", action="store_true")
    parser.add_argument("--label-min", type=float, default=-3.0)
    parser.add_argument("--label-max", type=float, default=3.0)
    parser.add_argument("--task-weight", type=float, default=1.0)
    parser.add_argument("--model-variant", choices=["minimal", "pnr_trifuse"], default="minimal")
    parser.add_argument("--disable-common", action="store_true")
    parser.add_argument("--disable-pairwise", action="store_true")
    parser.add_argument("--disable-private", action="store_true")
    parser.add_argument("--disable-noise", action="store_true")
    parser.add_argument("--disable-reconstruction", action="store_true")
    parser.add_argument("--disable-cross-factor-attention", action="store_true")
    parser.add_argument("--disable-dynamic-fusion", action="store_true")
    parser.add_argument("--disable-self-guided-fusion", action="store_true")
    parser.add_argument("--disable-text-guided-fusion", action="store_true")
    parser.add_argument("--disable-type-level-fusion", action="store_true")
    parser.add_argument(
        "--pnr-fusion-combine",
        choices=["mean_token_type", "tri_gate"],
        default="mean_token_type",
        help="How PNR-TriFuse combines self-guided, text-guided, and type-level fusion results.",
    )
    parser.add_argument("--orth-mode", choices=["full", "selective"], default="selective")
    parser.add_argument("--loss-weighting", choices=["static", "uncertainty"], default="uncertainty")
    parser.add_argument("--common-weight", type=float, default=0.03)
    parser.add_argument("--pair-weight", type=float, default=0.03)
    parser.add_argument("--orth-weight", type=float, default=0.01)
    parser.add_argument("--hsic-weight", type=float, default=0.005)
    parser.add_argument("--info-weight", type=float, default=0.03)
    parser.add_argument("--noise-weight", type=float, default=0.005)
    parser.add_argument("--reconstruction-weight", type=float, default=0.05)
    parser.add_argument("--kl-weight", type=float, default=0.0005)
    parser.add_argument("--regularization-warmup-epochs", type=int, default=3)
    parser.add_argument("--selection-metric", default="mae", help="Validation metric used to choose best checkpoint.")
    parser.add_argument(
        "--extra-selection-metrics",
        nargs="*",
        default=[],
        help="Additional validation metrics/losses to track and evaluate on the test set when improved.",
    )
    parser.add_argument(
        "--no-save-checkpoint",
        action="store_true",
        help="Skip writing best.pt; record test metrics immediately whenever validation selection improves.",
    )
    parser.add_argument(
        "--save-checkpoint-with-no-save",
        action="store_true",
        help="When --no-save-checkpoint is set, still write best.pt after recording test metrics.",
    )
    parser.add_argument(
        "--eval-test-on-improvement",
        action="store_true",
        help="Evaluate test set whenever validation selection improves, even when checkpoints are saved.",
    )
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-eval-batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@contextmanager
def preserve_rng_state(device: torch.device):
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices, enabled=True):
        yield


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def iterate_batches(loader: DataLoader, limit: Optional[int]) -> Iterable[Tuple[int, Dict[str, torch.Tensor]]]:
    for idx, batch in enumerate(loader):
        if limit is not None and idx >= limit:
            break
        yield idx, batch


def metric_is_better(metric_name: str, current: float, best: float) -> bool:
    metric_name = metric_name.lower()
    if metric_name in {"mae", "loss", "total", "task"} or metric_name.startswith("loss_"):
        return current < best
    return current > best


def initial_best_score(metric_name: str) -> float:
    metric_name = metric_name.lower()
    if metric_name in {"mae", "loss", "total", "task"} or metric_name.startswith("loss_"):
        return float("inf")
    return -float("inf")


def selection_scores(valid_metrics: Dict[str, float], valid_losses: Dict[str, float]) -> Dict[str, float]:
    scores = dict(valid_metrics)
    scores.update(valid_losses)
    scores.update({f"loss_{key}": value for key, value in valid_losses.items()})
    if "total" in valid_losses:
        scores["loss"] = valid_losses["total"]
    return scores


def default_raw_media_root(data_root: str, dataset: str) -> Path:
    name = "SIMSv2" if dataset.lower() == "simsv2" else dataset.upper()
    return Path(data_root) / name / "Raw"


def build_optimizer(
    model: FUSENet,
    args: argparse.Namespace,
    criterion: Optional[FUSENetCriterion] = None,
) -> torch.optim.Optimizer:
    groups: Dict[str, list] = {"other": [], "text": [], "audio": [], "vision": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("text_encoder."):
            groups["text"].append(param)
        elif name.startswith("audio_encoder."):
            groups["audio"].append(param)
        elif name.startswith("vision_encoder."):
            groups["vision"].append(param)
        else:
            groups["other"].append(param)

    param_groups = []
    lr_by_group = {
        "other": args.lr,
        "text": args.text_lr if args.text_lr is not None else args.lr,
        "audio": args.audio_lr if args.audio_lr is not None else args.lr,
        "vision": args.vision_lr if args.vision_lr is not None else args.lr,
    }
    for group_name in ["other", "text", "audio", "vision"]:
        if groups[group_name]:
            param_groups.append({"params": groups[group_name], "lr": lr_by_group[group_name], "name": group_name})
    if criterion is not None:
        criterion_params = [param for param in criterion.parameters() if param.requires_grad]
        if criterion_params:
            param_groups.append({"params": criterion_params, "lr": args.lr, "name": "criterion"})
    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


def train_epoch(
    model: FUSENet,
    criterion: FUSENetCriterion,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    grad_accum: int,
    limit: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    running: Dict[str, float] = {}
    steps = 0
    optimizer.zero_grad(set_to_none=True)
    total_batches = limit or len(loader)
    for step, (_, batch) in enumerate(
        tqdm(iterate_batches(loader, limit), total=total_batches, desc="train", leave=False),
        start=1,
    ):
        batch = to_device(batch, device)
        labels = batch["label"]
        with torch.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
            outputs = model(batch)
            losses = criterion(outputs, labels)
            backward_loss = losses["total"] / max(1, grad_accum)
        scaler.scale(backward_loss).backward()
        if step % max(1, grad_accum) == 0 or step == total_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        steps += 1
        for key, value in losses.items():
            running[key] = running.get(key, 0.0) + float(value.detach().cpu())
    return {k: v / max(1, steps) for k, v in running.items()}


@torch.no_grad()
def evaluate(
    model: FUSENet,
    criterion: FUSENetCriterion,
    loader: DataLoader,
    device: torch.device,
    dataset: str,
    label_min: float,
    label_max: float,
    use_amp: bool,
    limit: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    all_pred = []
    all_true = []
    running: Dict[str, float] = {}
    steps = 0
    for _, batch in tqdm(iterate_batches(loader, limit), total=limit or len(loader), desc="eval", leave=False):
        batch = to_device(batch, device)
        labels = batch["label"]
        with torch.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
            outputs = model(batch)
            losses = criterion(outputs, labels)
        all_pred.append(outputs["pred"].detach().cpu().numpy())
        all_true.append(labels.detach().cpu().numpy())
        steps += 1
        for key, value in losses.items():
            running[key] = running.get(key, 0.0) + float(value.detach().cpu())
    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)
    metrics = regression_report(pred, true, dataset, label_min, label_max)
    losses = {k: v / max(1, steps) for k, v in running.items()}
    return metrics, losses


def evaluate_preserving_rng(
    model: FUSENet,
    criterion: FUSENetCriterion,
    loader: DataLoader,
    device: torch.device,
    dataset: str,
    label_min: float,
    label_max: float,
    use_amp: bool,
    limit: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    with preserve_rng_state(device):
        return evaluate(model, criterion, loader, device, dataset, label_min, label_max, use_amp, limit)


def main() -> None:
    args = parse_args()
    if args.dataset == "simsv2" and args.label_min == -3.0 and args.label_max == 3.0:
        args.label_min, args.label_max = -1.0, 1.0
    set_seed(args.seed)
    device = torch.device(args.device)

    load_raw_audio = args.audio_backbone is not None and "audio" not in args.disabled_modalities
    load_raw_vision = args.vision_backbone is not None and "vision" not in args.disabled_modalities
    raw_media_root = Path(args.raw_media_root) if args.raw_media_root else default_raw_media_root(args.data_root, args.dataset)
    media_cache_root = Path(args.media_cache_root) if args.media_cache_root else Path(args.output_dir) / "_media_cache" / args.dataset
    if load_raw_audio or load_raw_vision:
        print(f"Using raw media root: {raw_media_root}")
        print(f"Using media cache root: {media_cache_root}")

    train_set, valid_set, test_set = make_datasets(
        args.data_root,
        args.dataset,
        args.data_file,
        raw_media_root=raw_media_root,
        media_cache_root=media_cache_root,
        load_raw_audio=load_raw_audio,
        load_raw_vision=load_raw_vision,
        audio_sample_rate=args.audio_sample_rate,
        audio_max_seconds=args.audio_max_seconds,
        vision_num_frames=args.vision_num_frames,
        vision_frame_size=args.vision_frame_size,
    )
    input_dims = train_set.feature_dims()
    print(f"Using input dims: {input_dims}")
    print(f"Splits: train={len(train_set)} valid={len(valid_set)} test={len(test_set)}")
    tokenizer = None
    if args.text_backbone:
        tokenizer = AutoTokenizer.from_pretrained(args.text_backbone, cache_dir=args.text_cache_dir, use_fast=True)
        print(f"Using end-to-end text backbone: {args.text_backbone}")
    if args.audio_backbone:
        print(f"Using end-to-end audio backbone: {args.audio_backbone}")
    if args.vision_backbone:
        print(f"Using end-to-end vision backbone: {args.vision_backbone}")
    collate_fn = make_collate_mmsa(tokenizer, args.text_max_length)
    shuffle_seed = args.shuffle_seed if args.shuffle_seed is not None else args.seed
    loader_generators = {
        "train": make_generator(shuffle_seed + 1001),
        "valid": make_generator(shuffle_seed + 1002),
        "test": make_generator(shuffle_seed + 1003),
    }
    print(
        json.dumps(
            {
                "event": "reproducibility",
                "seed": args.seed,
                "shuffle_seed": shuffle_seed,
                "train_loader_seed": shuffle_seed + 1001,
                "valid_loader_seed": shuffle_seed + 1002,
                "test_loader_seed": shuffle_seed + 1003,
                "deterministic_algorithms": True,
            },
            sort_keys=True,
        )
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=loader_generators["train"],
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=loader_generators["valid"],
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=loader_generators["test"],
    )

    model = FUSENet(
        input_dims=input_dims,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        disabled_modalities=args.disabled_modalities,
        text_model_name_or_path=args.text_backbone,
        text_cache_dir=args.text_cache_dir,
        freeze_text_encoder=args.freeze_text_backbone,
        text_gradient_checkpointing=args.text_gradient_checkpointing,
        audio_model_name_or_path=args.audio_backbone,
        audio_cache_dir=args.audio_cache_dir,
        freeze_audio_encoder=args.freeze_audio_backbone,
        audio_gradient_checkpointing=args.audio_gradient_checkpointing,
        vision_model_name_or_path=args.vision_backbone,
        vision_cache_dir=args.vision_cache_dir,
        freeze_vision_encoder=args.freeze_vision_backbone,
        vision_gradient_checkpointing=args.vision_gradient_checkpointing,
        vision_frame_chunk_size=args.vision_frame_chunk_size,
        fusion_mode=args.fusion_mode,
        model_variant=args.model_variant,
        disable_common=args.disable_common,
        disable_pairwise=args.disable_pairwise,
        disable_private=args.disable_private,
        disable_noise=args.disable_noise,
        disable_reconstruction=args.disable_reconstruction,
        disable_cross_factor_attention=args.disable_cross_factor_attention,
        disable_dynamic_fusion=args.disable_dynamic_fusion,
        disable_self_guided_fusion=args.disable_self_guided_fusion,
        disable_text_guided_fusion=args.disable_text_guided_fusion,
        disable_type_level_fusion=args.disable_type_level_fusion,
        pnr_fusion_combine=args.pnr_fusion_combine,
    ).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        incompatible = model.load_state_dict(state_dict, strict=False)
        print(
            json.dumps(
                {
                    "event": "load_init_checkpoint",
                    "path": args.init_checkpoint,
                    "source_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
                    "missing_keys": list(incompatible.missing_keys),
                    "unexpected_keys": list(incompatible.unexpected_keys),
                },
                sort_keys=True,
            )
        )
    if args.initial_freeze_backbone_epochs > 0:
        model.set_backbone_trainability(False)
        print(
            json.dumps(
                {
                    "event": "freeze_backbones",
                    "epochs": args.initial_freeze_backbone_epochs,
                },
                sort_keys=True,
            )
        )
    if args.initial_freeze_mapping_attention_epochs > 0:
        model.set_mapping_attention_trainability(False)
        print(
            json.dumps(
                {
                    "event": "freeze_mapping_attention",
                    "epochs": args.initial_freeze_mapping_attention_epochs,
                },
                sort_keys=True,
            )
        )
    weights = LossWeights(
        task=args.task_weight,
        common=args.common_weight,
        pair=args.pair_weight,
        orth=args.orth_weight,
        hsic=args.hsic_weight,
        info=args.info_weight,
        noise=args.noise_weight,
        reconstruction=args.reconstruction_weight,
        kl=args.kl_weight,
    )
    criterion = FUSENetCriterion(weights, orth_mode=args.orth_mode, loss_weighting=args.loss_weighting).to(device)
    optimizer = build_optimizer(model, args, criterion)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    out_dir = Path(args.output_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_metric = args.selection_metric
    best_score = initial_best_score(selection_metric)
    best_epoch = 0
    best_valid_metrics = None
    best_valid_losses = None
    best_test_metrics = None
    best_test_losses = None
    extra_selection_metrics = [metric for metric in args.extra_selection_metrics if metric != selection_metric]
    extra_best_scores = {metric: initial_best_score(metric) for metric in extra_selection_metrics}
    extra_best_records = {}
    evaluate_test_on_improvement = args.no_save_checkpoint or args.eval_test_on_improvement
    stale = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        rebuild_optimizer = False
        if args.initial_freeze_backbone_epochs > 0 and epoch == args.initial_freeze_backbone_epochs + 1:
            model.set_backbone_trainability(True, args.unfreeze_last_n_layers)
            rebuild_optimizer = True
            print(
                json.dumps(
                    {
                        "event": "unfreeze_backbones",
                        "epoch": epoch,
                        "last_n_layers": args.unfreeze_last_n_layers,
                    },
                    sort_keys=True,
                )
            )
        if (
            args.initial_freeze_mapping_attention_epochs > 0
            and epoch == args.initial_freeze_mapping_attention_epochs + 1
        ):
            model.set_mapping_attention_trainability(True)
            rebuild_optimizer = True
            print(
                json.dumps(
                    {
                        "event": "unfreeze_mapping_attention",
                        "epoch": epoch,
                    },
                    sort_keys=True,
                )
            )
        if rebuild_optimizer:
            optimizer = build_optimizer(model, args, criterion)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        if args.regularization_warmup_epochs > 0:
            regularization_scale = min(1.0, epoch / float(args.regularization_warmup_epochs))
        else:
            regularization_scale = 1.0
        criterion.set_regularization_scale(regularization_scale)
        train_losses = train_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            device,
            scaler,
            args.fp16,
            args.grad_accum,
            args.limit_train_batches,
        )
        valid_metrics, valid_losses = evaluate_preserving_rng(
            model,
            criterion,
            valid_loader,
            device,
            args.dataset,
            args.label_min,
            args.label_max,
            args.fp16,
            args.limit_eval_batches,
        )
        scheduler.step(valid_metrics["mae"])
        row = {
            "epoch": epoch,
            "train": train_losses,
            "valid": valid_metrics,
            "valid_losses": valid_losses,
            "regularization_scale": regularization_scale,
            "lr": optimizer.param_groups[0]["lr"],
            "group_lrs": {group.get("name", f"group_{i}"): group["lr"] for i, group in enumerate(optimizer.param_groups)},
        }
        history.append(row)
        print(json.dumps(row, indent=2, sort_keys=True))

        scores = selection_scores(valid_metrics, valid_losses)
        if selection_metric not in scores:
            raise KeyError(
                f"Selection metric {selection_metric!r} is not available. "
                f"Valid score keys: {sorted(scores)}"
            )
        for metric in extra_selection_metrics:
            if metric not in scores:
                raise KeyError(
                    f"Extra selection metric {metric!r} is not available. "
                    f"Valid score keys: {sorted(scores)}"
                )
        test_metrics_for_epoch = None
        test_losses_for_epoch = None
        current_score = scores[selection_metric]
        if metric_is_better(selection_metric, current_score, best_score):
            best_score = current_score
            best_epoch = epoch
            best_valid_metrics = dict(valid_metrics)
            best_valid_losses = dict(valid_losses)
            stale = 0
            if not args.no_save_checkpoint:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "input_dims": input_dims,
                        "args": vars(args),
                        "epoch": epoch,
                        "valid": valid_metrics,
                        "valid_losses": valid_losses,
                        "selection_metric": selection_metric,
                        "selection_score": best_score,
                    },
                    out_dir / "best.pt",
                )
            if evaluate_test_on_improvement:
                if test_metrics_for_epoch is None or test_losses_for_epoch is None:
                    test_metrics_for_epoch, test_losses_for_epoch = evaluate_preserving_rng(
                        model,
                        criterion,
                        test_loader,
                        device,
                        args.dataset,
                        args.label_min,
                        args.label_max,
                        args.fp16,
                        args.limit_eval_batches,
                    )
                best_test_metrics = test_metrics_for_epoch
                best_test_losses = test_losses_for_epoch
                print(
                    json.dumps(
                        {
                            "best_epoch": best_epoch,
                            "selection_metric": selection_metric,
                            "selection_score": best_score,
                            "test_at_best": best_test_metrics,
                            "test_losses_at_best": best_test_losses,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            if args.no_save_checkpoint and args.save_checkpoint_with_no_save:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "input_dims": input_dims,
                        "args": vars(args),
                        "epoch": epoch,
                        "valid": valid_metrics,
                        "valid_losses": valid_losses,
                        "selection_metric": selection_metric,
                        "selection_score": best_score,
                    },
                    out_dir / "best.pt",
                )
        else:
            stale += 1
        for metric in extra_selection_metrics:
            current_extra_score = scores[metric]
            if metric_is_better(metric, current_extra_score, extra_best_scores[metric]):
                extra_best_scores[metric] = current_extra_score
                if evaluate_test_on_improvement:
                    if test_metrics_for_epoch is None or test_losses_for_epoch is None:
                        test_metrics_for_epoch, test_losses_for_epoch = evaluate_preserving_rng(
                            model,
                            criterion,
                            test_loader,
                            device,
                            args.dataset,
                            args.label_min,
                            args.label_max,
                            args.fp16,
                            args.limit_eval_batches,
                        )
                    extra_best_records[metric] = {
                        "best_epoch": epoch,
                        "selection_metric": metric,
                        "selection_score": current_extra_score,
                        "valid": dict(valid_metrics),
                        "valid_losses": dict(valid_losses),
                        "test": test_metrics_for_epoch,
                        "test_losses": test_losses_for_epoch,
                    }
                    print(
                        json.dumps(
                            {
                                "extra_best_epoch": epoch,
                                "extra_selection_metric": metric,
                                "extra_selection_score": current_extra_score,
                                "test_at_extra_best": test_metrics_for_epoch,
                                "test_losses_at_extra_best": test_losses_for_epoch,
                            },
                            indent=2,
                            sort_keys=True,
                        )
                    )
        if stale >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}.")
            break

    if args.no_save_checkpoint:
        if best_test_metrics is None or best_test_losses is None:
            best_test_metrics, best_test_losses = evaluate_preserving_rng(
                model,
                criterion,
                test_loader,
                device,
                args.dataset,
                args.label_min,
                args.label_max,
                args.fp16,
                args.limit_eval_batches,
            )
        test_metrics = best_test_metrics
        test_losses = best_test_losses
        checkpoint_valid = best_valid_metrics or {}
        checkpoint_valid_losses = best_valid_losses or {}
    else:
        checkpoint = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        test_metrics, test_losses = evaluate_preserving_rng(
            model,
            criterion,
            test_loader,
            device,
            args.dataset,
            args.label_min,
            args.label_max,
            args.fp16,
            args.limit_eval_batches,
        )
        checkpoint_valid = checkpoint.get("valid", {})
        checkpoint_valid_losses = checkpoint.get("valid_losses", {})
    final = {
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "best_valid_score": best_score,
        "best_valid_mae": checkpoint_valid.get("mae"),
        "best_valid_loss": checkpoint_valid_losses.get("total"),
        "checkpoint_saved": (not args.no_save_checkpoint) or args.save_checkpoint_with_no_save,
        "test": test_metrics,
        "test_losses": test_losses,
        "extra_best": extra_best_records,
    }
    print(json.dumps(final, indent=2, sort_keys=True))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "test_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
