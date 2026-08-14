from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    task: float = 1.0
    common: float = 0.03
    pair: float = 0.03
    orth: float = 0.01
    hsic: float = 0.005
    info: float = 0.03
    noise: float = 0.005
    reconstruction: float = 0.05
    kl: float = 0.0005


REGULARIZER_NAMES = (
    "common",
    "pair",
    "orth",
    "hsic",
    "info",
    "noise",
    "reconstruction",
    "kl",
)


def _mean_or_zero(values: Iterable[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    values = list(values)
    if not values:
        return reference.new_tensor(0.0)
    return torch.stack(values).mean()


def _cosine_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(first, second, dim=-1)).mean()


def _center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0, keepdim=True)


def _cross_covariance_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.size(0) <= 1:
        return first.new_tensor(0.0)
    first = _center(first)
    second = _center(second)
    first = first / (first.std(dim=0, keepdim=True) + 1e-6)
    second = second / (second.std(dim=0, keepdim=True) + 1e-6)
    cov = first.transpose(0, 1).matmul(second) / max(1, first.size(0) - 1)
    return cov.pow(2).mean()


def _label_dependency_loss(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim == 1:
        label_features = labels.reshape(-1, 1).to(features.dtype)
    else:
        label_features = labels.to(features.dtype)
    return _cross_covariance_loss(features, label_features)


def _factor_list(modality_factors: Dict[str, object]) -> List[torch.Tensor]:
    factors = []
    if "common" in modality_factors:
        factors.append(modality_factors["common"])
    factors.extend(modality_factors["pairwise"].values())
    if "private" in modality_factors:
        factors.append(modality_factors["private"])
    if "noise" in modality_factors:
        factors.append(modality_factors["noise"])
    return factors


def _orthogonal_pair_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = F.normalize(first, dim=-1)
    second = F.normalize(second, dim=-1)
    return (first * second).sum(dim=-1).pow(2).mean()


def _full_orthogonality_loss(factors: Dict[str, Dict[str, object]], reference: torch.Tensor) -> torch.Tensor:
    losses = []
    for modality_factors in factors.values():
        branches = _factor_list(modality_factors)
        if len(branches) <= 1:
            continue
        normalized = [F.normalize(branch, dim=-1) for branch in branches]
        stacked = torch.stack(normalized, dim=1)
        gram = torch.bmm(stacked, stacked.transpose(1, 2))
        eye = torch.eye(gram.size(1), device=gram.device, dtype=gram.dtype).unsqueeze(0)
        losses.append((gram * (1.0 - eye)).pow(2).mean())
    return _mean_or_zero(losses, reference)


def _selective_orthogonality_loss(factors: Dict[str, Dict[str, object]], reference: torch.Tensor) -> torch.Tensor:
    losses = []
    for modality_factors in factors.values():
        shared_parts = []
        if "common" in modality_factors:
            shared_parts.append(modality_factors["common"])
        shared_parts.extend(modality_factors["pairwise"].values())
        private = modality_factors.get("private")
        if private is not None:
            for shared in shared_parts:
                losses.append(_orthogonal_pair_loss(shared, private))
        if "noise" in modality_factors:
            noise = modality_factors["noise"]
            sentiment_factors = list(shared_parts)
            if private is not None:
                sentiment_factors.append(private)
            for sentiment_factor in sentiment_factors:
                losses.append(_orthogonal_pair_loss(sentiment_factor, noise))
    return _mean_or_zero(losses, reference)


def _orthogonality_loss(
    factors: Dict[str, Dict[str, object]],
    reference: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "full":
        return _full_orthogonality_loss(factors, reference)
    if mode == "selective":
        return _selective_orthogonality_loss(factors, reference)
    raise ValueError("orthogonality mode must be either 'full' or 'selective'.")


def _hsic_like_loss(factors: Dict[str, Dict[str, object]], reference: torch.Tensor) -> torch.Tensor:
    losses = []
    for modality_factors in factors.values():
        shared_parts = []
        if "common" in modality_factors:
            shared_parts.append(modality_factors["common"])
        shared_parts.extend(modality_factors["pairwise"].values())
        private = modality_factors.get("private")
        if private is not None:
            for shared in shared_parts:
                losses.append(_cross_covariance_loss(shared, private))
        if "noise" in modality_factors:
            noise = modality_factors["noise"]
            sentiment_factors = list(shared_parts)
            if private is not None:
                sentiment_factors.append(private)
            for sentiment_factor in sentiment_factors:
                losses.append(_cross_covariance_loss(sentiment_factor, noise))
    return _mean_or_zero(losses, reference)


class FUSENetCriterion(nn.Module):
    """Regression or classification objective with optional PNR-TriFuse regularization."""

    def __init__(
        self,
        weights: LossWeights,
        orth_mode: str = "full",
        loss_weighting: str = "static",
        task_type: str = "regression",
        num_classes: Optional[int] = None,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if orth_mode not in {"full", "selective"}:
            raise ValueError("orth_mode must be either 'full' or 'selective'.")
        if loss_weighting not in {"static", "uncertainty"}:
            raise ValueError("loss_weighting must be either 'static' or 'uncertainty'.")
        if task_type not in {"regression", "classification"}:
            raise ValueError("task_type must be either 'regression' or 'classification'.")
        self.weights = weights
        self.orth_mode = orth_mode
        self.loss_weighting = loss_weighting
        self.task_type = task_type
        self.num_classes = num_classes
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)
        self.regularization_scale = 1.0
        self.regularizer_log_vars = nn.ParameterDict()
        if self.loss_weighting == "uncertainty":
            self.regularizer_log_vars = nn.ParameterDict(
                {name: nn.Parameter(torch.zeros(())) for name in REGULARIZER_NAMES}
            )

    def set_regularization_scale(self, scale: float) -> None:
        self.regularization_scale = float(max(0.0, min(1.0, scale)))

    def _task_loss(self, outputs: Dict[str, object], labels: torch.Tensor) -> torch.Tensor:
        if self.task_type == "classification":
            return F.cross_entropy(outputs["pred"], labels.to(torch.long), weight=self.class_weights)
        return F.mse_loss(outputs["pred"], labels)

    def _common_loss(self, factors: Dict[str, Dict[str, object]], reference: torch.Tensor) -> torch.Tensor:
        common = [modality_factors["common"] for modality_factors in factors.values() if "common" in modality_factors]
        losses = []
        for i in range(len(common)):
            for j in range(i + 1, len(common)):
                losses.append(_cosine_distance(common[i], common[j]))
        return _mean_or_zero(losses, reference)

    def _pair_loss(self, factors: Dict[str, Dict[str, object]], reference: torch.Tensor) -> torch.Tensor:
        pair_to_branches: Dict[str, List[torch.Tensor]] = {}
        for modality_factors in factors.values():
            for name, branch in modality_factors["pairwise"].items():
                pair_to_branches.setdefault(name, []).append(branch)
        losses = []
        for branches in pair_to_branches.values():
            if len(branches) == 2:
                losses.append(_cosine_distance(branches[0], branches[1]))
        return _mean_or_zero(losses, reference)

    def _info_loss(self, outputs: Dict[str, object], labels: torch.Tensor) -> torch.Tensor:
        aux_preds = outputs.get("aux_preds", {})
        if self.task_type == "classification":
            losses = [
                F.cross_entropy(pred, labels.to(torch.long), weight=self.class_weights)
                for pred in aux_preds.values()
            ]
        else:
            losses = [F.mse_loss(pred, labels) for pred in aux_preds.values()]
        return _mean_or_zero(losses, labels)

    def _noise_loss(self, factors: Dict[str, Dict[str, object]], labels: torch.Tensor) -> torch.Tensor:
        dependency_labels = labels
        if self.task_type == "classification":
            num_classes = int(self.num_classes or (labels.max().item() + 1))
            dependency_labels = F.one_hot(labels.to(torch.long), num_classes=num_classes).to(torch.float32)
        losses = []
        for modality_factors in factors.values():
            if "noise" in modality_factors:
                losses.append(_label_dependency_loss(modality_factors["noise"], dependency_labels))
        noise_tokens = None
        if isinstance(factors, dict):
            noise_stack = [modality_factors["noise"] for modality_factors in factors.values() if "noise" in modality_factors]
            if noise_stack:
                noise_tokens = torch.stack(noise_stack, dim=1).mean(dim=1)
        if noise_tokens is not None:
            losses.append(_label_dependency_loss(noise_tokens, dependency_labels))
        return _mean_or_zero(losses, labels)

    def _reconstruction_losses(self, outputs: Dict[str, object], reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        reconstructions = outputs.get("reconstructions", {})
        targets = outputs.get("reconstruction_targets", {})
        kl_losses = outputs.get("kl_losses", {})
        recon_terms = [
            F.mse_loss(reconstructions[modality], targets[modality])
            for modality in reconstructions
            if modality in targets
        ]
        kl_terms = list(kl_losses.values())
        return {
            "reconstruction": _mean_or_zero(recon_terms, reference),
            "kl": _mean_or_zero(kl_terms, reference),
        }

    def _regularizer_total(self, loss_parts: Dict[str, torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
        terms = []
        for name in REGULARIZER_NAMES:
            loss_value = loss_parts[name]
            base_weight = getattr(self.weights, name)
            if base_weight <= 0.0:
                continue
            weighted_loss = reference.new_tensor(float(base_weight)) * loss_value
            if self.loss_weighting == "uncertainty":
                log_var = torch.clamp(self.regularizer_log_vars[name], min=-5.0, max=5.0)
                terms.append(torch.exp(-log_var) * weighted_loss + F.softplus(log_var))
            else:
                terms.append(weighted_loss)
        return torch.stack(terms).sum() if terms else reference.new_tensor(0.0)

    def forward(self, outputs: Dict[str, object], labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        task = self._task_loss(outputs, labels)
        losses: Dict[str, torch.Tensor] = {"task": task}
        total = self.weights.task * task

        factors = outputs.get("factors")
        if factors:
            common = self._common_loss(factors, task)
            pair = self._pair_loss(factors, task)
            orth = _orthogonality_loss(factors, task, self.orth_mode)
            hsic = _hsic_like_loss(factors, task)
            info = self._info_loss(outputs, labels)
            noise = self._noise_loss(factors, labels)
            recon_parts = self._reconstruction_losses(outputs, task)

            losses.update(
                {
                    "common": common,
                    "pair": pair,
                    "orth": orth,
                    "hsic": hsic,
                    "info": info,
                    "noise": noise,
                    "reconstruction": recon_parts["reconstruction"],
                    "kl": recon_parts["kl"],
                }
            )
            scale = task.new_tensor(self.regularization_scale)
            total = total + scale * self._regularizer_total(losses, task)
            if self.loss_weighting == "uncertainty":
                for name in REGULARIZER_NAMES:
                    losses[f"dynamic_weight_{name}"] = torch.exp(
                        -torch.clamp(self.regularizer_log_vars[name], min=-5.0, max=5.0)
                    ).detach()

        losses["total"] = total
        return losses
