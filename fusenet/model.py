from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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

from pathlib import Path

from transformers import AutoConfig, AutoModel


MODALITIES = ("text", "audio", "vision")
PAIR_MODALITIES = (("text", "audio"), ("text", "vision"), ("audio", "vision"))
VIDEO_BACKBONE_TYPES = {"timesformer", "videomae", "vivit"}


def pair_name(first: str, second: str) -> str:
    return f"{first}_{second}"


def config_hidden_size(config) -> int:
    for name in ("hidden_size", "d_model", "encoder_embed_dim", "projection_dim"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError(f"Could not infer hidden size from config {config.__class__.__name__}.")


def load_auto_model(model_name_or_path: str, cache_dir: Optional[str] = None) -> nn.Module:
    local_path = Path(model_name_or_path)
    if local_path.is_dir():
        weight_names = {
            "model.safetensors",
            "pytorch_model.bin",
            "tf_model.h5",
            "flax_model.msgpack",
        }
        if not any((local_path / name).exists() for name in weight_names):
            config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
            return AutoModel.from_config(config)
    try:
        return AutoModel.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    except AttributeError as exc:
        message = str(exc)
        safetensor_metadata_failure = "metadata" in message or ("NoneType" in message and "get" in message)
        if not safetensor_metadata_failure:
            raise
        return AutoModel.from_pretrained(model_name_or_path, cache_dir=cache_dir, use_safetensors=False)


def lengths_to_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    lengths = lengths.clamp(min=1, max=max_length).to(torch.long)
    steps = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    return steps < lengths.unsqueeze(1)


def masked_mean(sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    lengths = lengths.clamp(min=1, max=sequence.size(1)).to(torch.long)
    mask = lengths_to_mask(lengths.to(sequence.device), sequence.size(1)).unsqueeze(-1)
    masked = sequence * mask.to(sequence.dtype)
    return masked.sum(dim=1) / lengths.to(sequence.dtype).unsqueeze(-1)


class TemporalFeatureExtractor(nn.Module):
    """Feature extractor for precomputed MMSA modality sequences."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.rnn = nn.GRU(
            output_dim,
            output_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = lengths.clamp(min=1, max=sequence.size(1)).to(torch.long)
        x = self.input_proj(self.input_norm(sequence))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=sequence.size(1))
        return self.output_proj(output), lengths.to(sequence.device)


class FeatureMapping(nn.Module):
    """Maps an extracted modality sequence into the shared hidden size."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TextGuidedAttentionPooler(nn.Module):
    """Pools a modality sequence with scores conditioned on the pooled text vector."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        text_query: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query = text_query.unsqueeze(1).expand(-1, sequence.size(1), -1)
        scores = self.score(torch.cat([sequence, query, sequence * query], dim=-1)).squeeze(-1)
        if lengths is not None:
            mask = lengths_to_mask(lengths.to(sequence.device), sequence.size(1))
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        pooled = (sequence * weights.unsqueeze(-1).to(sequence.dtype)).sum(dim=1)
        return pooled, weights


class SelfAttentionPooler(nn.Module):
    """Pools a modality sequence without cross-modal conditioning."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(sequence).squeeze(-1)
        if lengths is not None:
            mask = lengths_to_mask(lengths.to(sequence.device), sequence.size(1))
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        pooled = (sequence * weights.unsqueeze(-1).to(sequence.dtype)).sum(dim=1)
        return pooled, weights


class ClassifierHead(nn.Module):
    """Predicts the scalar sentiment score from fused modality features."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.net(fused).squeeze(-1)


class FactorBranch(nn.Module):
    """Projects a pooled modality vector into one disentangled factor."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature)


class ModalityFactorizer(nn.Module):
    """Factorizes one modality into common, pairwise, private, and optional noise factors."""

    def __init__(
        self,
        hidden_dim: int,
        pair_names: Iterable[str],
        dropout: float = 0.1,
        enable_common: bool = True,
        enable_private: bool = True,
        enable_noise: bool = True,
    ):
        super().__init__()
        self.common = FactorBranch(hidden_dim, dropout) if enable_common else None
        self.private = FactorBranch(hidden_dim, dropout) if enable_private else None
        self.pairwise = nn.ModuleDict({name: FactorBranch(hidden_dim, dropout) for name in pair_names})
        self.noise = FactorBranch(hidden_dim, dropout) if enable_noise else None

    def forward(self, feature: torch.Tensor) -> Dict[str, object]:
        out: Dict[str, object] = {
            "pairwise": {name: branch(feature) for name, branch in self.pairwise.items()},
        }
        if self.common is not None:
            out["common"] = self.common(feature)
        if self.private is not None:
            out["private"] = self.private(feature)
        if self.noise is not None:
            out["noise"] = self.noise(feature)
        return out


class VariationalReconstructor(nn.Module):
    """Information-bottleneck reconstruction channel for one modality."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden_dim, hidden_dim)
        self.logvar = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, factors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(factors)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden).clamp(min=-10.0, max=10.0)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        recon = self.decoder(z)
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        return recon, kl


class PNRFusionModule(nn.Module):
    """Pairwise-aware dynamic fusion over disentangled factors."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        use_cross_factor_attention: bool = True,
        use_dynamic_fusion: bool = True,
        use_self_guided_fusion: bool = True,
        use_text_guided_fusion: bool = True,
        use_type_level_fusion: bool = True,
        fusion_combine_mode: str = "mean_token_type",
    ):
        super().__init__()
        if fusion_combine_mode not in {"mean_token_type", "tri_gate"}:
            raise ValueError("fusion_combine_mode must be 'mean_token_type' or 'tri_gate'.")
        self.use_cross_factor_attention = use_cross_factor_attention
        self.use_dynamic_fusion = use_dynamic_fusion
        self.use_self_guided_fusion = use_self_guided_fusion
        self.use_text_guided_fusion = use_text_guided_fusion
        self.use_type_level_fusion = use_type_level_fusion
        self.fusion_combine_mode = fusion_combine_mode
        self.type_embeddings = nn.ParameterDict(
            {
                "common": nn.Parameter(torch.empty(hidden_dim)),
                "pair": nn.Parameter(torch.empty(hidden_dim)),
                "private": nn.Parameter(torch.empty(hidden_dim)),
            }
        )
        for embedding in self.type_embeddings.values():
            nn.init.normal_(embedding, mean=0.0, std=0.02)

        heads = 4 if hidden_dim % 4 == 0 else 1
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.token_score = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.guidance_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.type_score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        if fusion_combine_mode == "tri_gate":
            self.tri_fusion_gate = nn.Sequential(
                nn.LayerNorm(hidden_dim * 3),
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim * 3),
            )
        else:
            self.tri_fusion_gate = None

    def _add_type_embeddings(self, tokens: torch.Tensor, token_types: List[str]) -> torch.Tensor:
        embeddings = torch.stack([self.type_embeddings[token_type] for token_type in token_types], dim=0)
        return tokens + embeddings.unsqueeze(0).to(tokens.dtype)

    def _group_contexts(
        self,
        tokens: torch.Tensor,
        token_types: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        contexts = []
        mask = []
        for group in ("common", "pair", "private"):
            indices = [idx for idx, token_type in enumerate(token_types) if token_type == group]
            if indices:
                group_tokens = tokens[:, indices]
                contexts.append(group_tokens.mean(dim=1))
                mask.append(True)
            else:
                contexts.append(torch.zeros_like(tokens[:, 0]))
                mask.append(False)
        return torch.stack(contexts, dim=1), torch.tensor(mask, device=tokens.device, dtype=torch.bool)

    def forward(
        self,
        tokens: torch.Tensor,
        token_types: List[str],
        noise_tokens: Optional[torch.Tensor] = None,
        text_query: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        tokens = self._add_type_embeddings(tokens, token_types)
        attention_matrix = None
        if self.use_cross_factor_attention:
            attended, attention_matrix = self.cross_attention(tokens, tokens, tokens, need_weights=True)
            tokens = self.cross_norm(tokens + attended)
            tokens = tokens + self.cross_ffn(tokens)

        token_weights = torch.full(
            (tokens.size(0), tokens.size(1)),
            1.0 / max(1, tokens.size(1)),
            device=tokens.device,
            dtype=tokens.dtype,
        )
        guided_weights = None
        gate = None
        tri_gate_weights = None
        type_weights = None

        if self.use_dynamic_fusion:
            global_context = tokens.mean(dim=1)
            self_context = None
            guided_context = None

            if self.use_self_guided_fusion:
                self_query = global_context.unsqueeze(1).expand(-1, tokens.size(1), -1)
                self_logits = self.token_score(torch.cat([tokens, self_query, tokens * self_query], dim=-1)).squeeze(-1)
                self_weights = torch.softmax(self_logits, dim=-1)
                self_context = (tokens * self_weights.unsqueeze(-1)).sum(dim=1)
            else:
                self_weights = None

            if self.use_text_guided_fusion and text_query is not None:
                guided_query = text_query.unsqueeze(1).expand(-1, tokens.size(1), -1)
                guided_logits = self.token_score(
                    torch.cat([tokens, guided_query, tokens * guided_query], dim=-1)
                ).squeeze(-1)
                guided_weights = torch.softmax(guided_logits, dim=-1)
                guided_context = (tokens * guided_weights.unsqueeze(-1)).sum(dim=1)

            if self.use_self_guided_fusion and guided_context is not None:
                gate = self.guidance_gate(torch.cat([guided_context, self_context], dim=-1))
                token_context = gate * guided_context + (1.0 - gate) * self_context
                token_weights = guided_weights
            elif guided_context is not None:
                token_context = guided_context
                token_weights = guided_weights
            elif self_context is not None:
                token_context = self_context
                token_weights = self_weights
            else:
                token_context = global_context

            if self.use_type_level_fusion:
                group_contexts, group_mask = self._group_contexts(tokens, token_types)
                type_query = text_query if text_query is not None else global_context
                type_logits = self.type_score(type_query)
                type_logits = type_logits.masked_fill(~group_mask.unsqueeze(0), torch.finfo(type_logits.dtype).min)
                type_weights = torch.softmax(type_logits, dim=-1)
                type_context = (group_contexts * type_weights.unsqueeze(-1)).sum(dim=1)
                if (
                    self.fusion_combine_mode == "tri_gate"
                    and self_context is not None
                    and guided_context is not None
                ):
                    tri_contexts = torch.stack([self_context, guided_context, type_context], dim=1)
                    tri_logits = self.tri_fusion_gate(
                        torch.cat([self_context, guided_context, type_context], dim=-1)
                    )
                    tri_gate_weights = torch.softmax(tri_logits.view(tokens.size(0), 3, -1), dim=1)
                    fused = (tri_contexts * tri_gate_weights).sum(dim=1)
                else:
                    fused = 0.5 * (token_context + type_context)
            else:
                fused = token_context
        else:
            fused = tokens.mean(dim=1)

        diagnostics: Dict[str, object] = {
            "token_types": token_types,
            "token_weights": token_weights,
            "guided_token_weights": guided_weights if self.use_dynamic_fusion else None,
            "guidance_gate": gate if self.use_dynamic_fusion else None,
            "tri_gate_weights": tri_gate_weights if self.use_dynamic_fusion else None,
            "type_weights": type_weights,
            "noise_gate": None,
            "cross_attention": attention_matrix,
        }
        return fused, diagnostics


class FUSENet(nn.Module):
    """Minimal text-guided multimodal sentiment model."""

    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int = 128,
        dropout: float = 0.1,
        disabled_modalities: Iterable[str] = (),
        text_model_name_or_path: Optional[str] = None,
        text_cache_dir: Optional[str] = None,
        freeze_text_encoder: bool = False,
        text_gradient_checkpointing: bool = False,
        audio_model_name_or_path: Optional[str] = None,
        audio_cache_dir: Optional[str] = None,
        freeze_audio_encoder: bool = False,
        audio_gradient_checkpointing: bool = False,
        vision_model_name_or_path: Optional[str] = None,
        vision_cache_dir: Optional[str] = None,
        freeze_vision_encoder: bool = False,
        vision_gradient_checkpointing: bool = False,
        vision_frame_chunk_size: Optional[int] = None,
        fusion_mode: str = "text_guided_attention",
        model_variant: str = "minimal",
        disable_common: bool = False,
        disable_pairwise: bool = False,
        disable_private: bool = False,
        disable_noise: bool = False,
        disable_reconstruction: bool = False,
        disable_cross_factor_attention: bool = False,
        disable_dynamic_fusion: bool = False,
        disable_self_guided_fusion: bool = False,
        disable_text_guided_fusion: bool = False,
        disable_type_level_fusion: bool = False,
        pnr_fusion_combine: str = "mean_token_type",
    ):
        super().__init__()
        if fusion_mode != "text_guided_attention":
            raise ValueError("The minimal model only supports fusion_mode='text_guided_attention'.")
        if model_variant not in {"minimal", "pnr_trifuse"}:
            raise ValueError("model_variant must be either 'minimal' or 'pnr_trifuse'.")
        if pnr_fusion_combine not in {"mean_token_type", "tri_gate"}:
            raise ValueError("pnr_fusion_combine must be 'mean_token_type' or 'tri_gate'.")

        self.hidden_dim = hidden_dim
        self.fusion_mode = fusion_mode
        self.model_variant = model_variant
        self.pnr_fusion_combine = pnr_fusion_combine
        self.disable_common = disable_common
        self.disable_pairwise = disable_pairwise
        self.disable_private = disable_private
        self.disable_noise = disable_noise
        self.disable_reconstruction = disable_reconstruction
        self.disable_cross_factor_attention = disable_cross_factor_attention
        self.disable_dynamic_fusion = disable_dynamic_fusion
        self.disable_self_guided_fusion = disable_self_guided_fusion
        self.disable_text_guided_fusion = disable_text_guided_fusion
        self.disable_type_level_fusion = disable_type_level_fusion
        self.disabled_modalities = tuple(disabled_modalities)
        self.active_modalities = tuple(m for m in MODALITIES if m not in self.disabled_modalities)
        if not self.active_modalities:
            raise ValueError("At least one modality must remain active.")
        if self.model_variant == "minimal" and "text" not in self.active_modalities:
            raise ValueError("The minimal text-guided model requires the text modality.")

        self.uses_text_backbone = text_model_name_or_path is not None
        self.uses_audio_backbone = audio_model_name_or_path is not None and "audio" in self.active_modalities
        self.uses_vision_backbone = vision_model_name_or_path is not None and "vision" in self.active_modalities
        self.freeze_text_encoder = freeze_text_encoder
        self.freeze_audio_encoder = freeze_audio_encoder
        self.freeze_vision_encoder = freeze_vision_encoder
        self.vision_frame_chunk_size = vision_frame_chunk_size

        self.feature_dims: Dict[str, int] = {}

        self.text_encoder: Optional[nn.Module] = None
        if self.uses_text_backbone:
            self.text_encoder = load_auto_model(text_model_name_or_path, cache_dir=text_cache_dir)
            if text_gradient_checkpointing and hasattr(self.text_encoder, "gradient_checkpointing_enable"):
                self.text_encoder.gradient_checkpointing_enable()
            if freeze_text_encoder:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
            self.feature_dims["text"] = config_hidden_size(self.text_encoder.config)
        else:
            self.feature_dims["text"] = hidden_dim

        self.audio_encoder: Optional[nn.Module] = None
        if self.uses_audio_backbone:
            self.audio_encoder = load_auto_model(audio_model_name_or_path, cache_dir=audio_cache_dir)
            if audio_gradient_checkpointing and hasattr(self.audio_encoder, "gradient_checkpointing_enable"):
                self.audio_encoder.gradient_checkpointing_enable()
            if freeze_audio_encoder:
                for param in self.audio_encoder.parameters():
                    param.requires_grad = False
            self.feature_dims["audio"] = config_hidden_size(self.audio_encoder.config)
        elif "audio" in self.active_modalities:
            self.feature_dims["audio"] = hidden_dim

        self.vision_encoder: Optional[nn.Module] = None
        self.vision_is_video_backbone = False
        if self.uses_vision_backbone:
            self.vision_encoder = load_auto_model(vision_model_name_or_path, cache_dir=vision_cache_dir)
            if vision_gradient_checkpointing and hasattr(self.vision_encoder, "gradient_checkpointing_enable"):
                self.vision_encoder.gradient_checkpointing_enable()
            if freeze_vision_encoder:
                for param in self.vision_encoder.parameters():
                    param.requires_grad = False
            model_type = str(getattr(self.vision_encoder.config, "model_type", "")).lower()
            self.vision_is_video_backbone = model_type in VIDEO_BACKBONE_TYPES
            self.feature_dims["vision"] = config_hidden_size(self.vision_encoder.config)
        elif "vision" in self.active_modalities:
            self.feature_dims["vision"] = hidden_dim

        self.feature_extractors = nn.ModuleDict()
        for modality in self.active_modalities:
            if modality == "text" and self.uses_text_backbone:
                continue
            if modality == "audio" and self.uses_audio_backbone:
                continue
            if modality == "vision" and self.uses_vision_backbone:
                continue
            self.feature_extractors[modality] = TemporalFeatureExtractor(input_dims[modality], hidden_dim, dropout)

        self.mappers = nn.ModuleDict(
            {modality: FeatureMapping(self.feature_dims[modality], hidden_dim, dropout) for modality in self.active_modalities}
        )
        self.attention_poolers = nn.ModuleDict(
            {
                modality: TextGuidedAttentionPooler(hidden_dim, dropout)
                for modality in self.active_modalities
                if modality != "text"
            }
        )
        if self.model_variant == "minimal":
            self.classifier = ClassifierHead(hidden_dim * len(self.active_modalities), hidden_dim, dropout)
        else:
            self.intrinsic_poolers = nn.ModuleDict(
                {
                    modality: SelfAttentionPooler(hidden_dim, dropout)
                    for modality in self.active_modalities
                    if modality != "text"
                }
            )
            self.active_pairs = tuple(
                (first, second)
                for first, second in PAIR_MODALITIES
                if first in self.active_modalities and second in self.active_modalities
            )
            pair_names_by_modality: Dict[str, List[str]] = {modality: [] for modality in self.active_modalities}
            if not self.disable_pairwise:
                for first, second in self.active_pairs:
                    name = pair_name(first, second)
                    pair_names_by_modality[first].append(name)
                    pair_names_by_modality[second].append(name)
            self.pair_names_by_modality = pair_names_by_modality
            self.factorizers = nn.ModuleDict(
                {
                    modality: ModalityFactorizer(
                        hidden_dim,
                        pair_names_by_modality[modality],
                        dropout,
                        enable_common=not self.disable_common,
                        enable_private=not self.disable_private,
                        enable_noise=not self.disable_noise,
                    )
                    for modality in self.active_modalities
                }
            )
            if not self.disable_reconstruction:
                recon_input_dims = {
                    modality: hidden_dim
                    * (
                        (0 if self.disable_common else 1)
                        + len(pair_names_by_modality[modality])
                        + (0 if self.disable_private else 1)
                        + (0 if self.disable_noise else 1)
                    )
                    for modality in self.active_modalities
                }
                self.reconstructors = nn.ModuleDict(
                    {
                        modality: VariationalReconstructor(recon_input_dims[modality], hidden_dim, dropout)
                        for modality in self.active_modalities
                    }
                )
            else:
                self.reconstructors = nn.ModuleDict()
            self.pnr_fusion = PNRFusionModule(
                hidden_dim,
                dropout,
                use_cross_factor_attention=not self.disable_cross_factor_attention,
                use_dynamic_fusion=not self.disable_dynamic_fusion,
                use_self_guided_fusion=not self.disable_self_guided_fusion,
                use_text_guided_fusion=not self.disable_text_guided_fusion,
                use_type_level_fusion=not self.disable_type_level_fusion,
                fusion_combine_mode=self.pnr_fusion_combine,
            )
            self.aux_heads = nn.ModuleDict(
                {
                    "common": ClassifierHead(hidden_dim, hidden_dim, dropout),
                    "pair": ClassifierHead(hidden_dim, hidden_dim, dropout),
                    "private": ClassifierHead(hidden_dim, hidden_dim, dropout),
                }
            )
            self.text_factor_query = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            self.classifier = ClassifierHead(hidden_dim, hidden_dim, dropout)

    def _run_encoder(self, encoder: nn.Module, frozen: bool, **kwargs):
        if frozen:
            encoder.eval()
            with torch.no_grad():
                return encoder(**kwargs)
        return encoder(**kwargs)

    def _set_module_trainable(self, module: Optional[nn.Module], trainable: bool) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = trainable

    def _set_last_layers_trainable(self, module: Optional[nn.Module], last_n_layers: int) -> None:
        if module is None:
            return
        self._set_module_trainable(module, False)
        if last_n_layers <= 0:
            self._set_module_trainable(module, True)
            return
        layers = None
        encoder = getattr(module, "encoder", None)
        if encoder is not None:
            if hasattr(encoder, "layer"):
                layers = encoder.layer
            elif hasattr(encoder, "layers"):
                layers = encoder.layers
        if layers is None:
            self._set_module_trainable(module, True)
            return
        for layer in list(layers)[-last_n_layers:]:
            self._set_module_trainable(layer, True)
        for attr in ("pooler", "layernorm", "layer_norm", "post_layernorm", "final_layer_norm"):
            self._set_module_trainable(getattr(module, attr, None), True)

    def set_backbone_trainability(self, trainable: bool, last_n_layers: int = 0) -> None:
        for module in (self.text_encoder, self.audio_encoder, self.vision_encoder):
            if trainable and last_n_layers > 0:
                self._set_last_layers_trainable(module, last_n_layers)
            else:
                self._set_module_trainable(module, trainable)
        self.freeze_text_encoder = not trainable
        self.freeze_audio_encoder = not trainable
        self.freeze_vision_encoder = not trainable

    def set_mapping_attention_trainability(self, trainable: bool) -> None:
        for module in (self.mappers, self.attention_poolers):
            self._set_module_trainable(module, trainable)

    def _single_step_sequence(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.ones(feature.size(0), device=feature.device, dtype=torch.long)
        return feature.unsqueeze(1), lengths

    def _extract_text(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.uses_text_backbone:
            return self.feature_extractors["text"](batch["text"], batch["text_lengths"])
        kwargs = {
            "input_ids": batch["text_input_ids"],
            "attention_mask": batch["text_attention_mask"],
        }
        if "text_token_type_ids" in batch:
            kwargs["token_type_ids"] = batch["text_token_type_ids"]
        outputs = self._run_encoder(self.text_encoder, self.freeze_text_encoder, **kwargs)
        lengths = batch["text_attention_mask"].sum(dim=1).clamp(min=1)
        return outputs.last_hidden_state, lengths

    def _audio_output_lengths(self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            return torch.full((hidden.size(0),), hidden.size(1), device=hidden.device, dtype=torch.long)
        input_lengths = attention_mask.sum(dim=1).to(torch.long)
        if hasattr(self.audio_encoder, "_get_feat_extract_output_lengths"):
            lengths = self.audio_encoder._get_feat_extract_output_lengths(input_lengths)
        else:
            scale = hidden.size(1) / max(float(attention_mask.size(1)), 1.0)
            lengths = torch.ceil(input_lengths.to(torch.float32) * scale).to(torch.long)
        return lengths.to(hidden.device).clamp(min=1, max=hidden.size(1))

    def _extract_audio(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.uses_audio_backbone:
            return self.feature_extractors["audio"](batch["audio"], batch["audio_lengths"])
        kwargs = {"input_values": batch["audio_input_values"]}
        attention_mask = batch.get("audio_attention_mask")
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        outputs = self._run_encoder(self.audio_encoder, self.freeze_audio_encoder, **kwargs)
        hidden = outputs.last_hidden_state
        return hidden, self._audio_output_lengths(hidden, attention_mask)

    def _vision_sequence_from_frames(self, pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_frames, channels, height, width = pixels.shape
        flat_pixels = pixels.reshape(batch_size * num_frames, channels, height, width)
        chunk_size = self.vision_frame_chunk_size or flat_pixels.size(0)
        chunks = []
        for start in range(0, flat_pixels.size(0), chunk_size):
            chunk_pixels = flat_pixels[start : start + chunk_size]
            outputs = self._run_encoder(
                self.vision_encoder,
                self.freeze_vision_encoder,
                pixel_values=chunk_pixels,
            )
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                chunks.append(outputs.pooler_output)
            else:
                hidden = outputs.last_hidden_state
                chunks.append(hidden[:, 0] if hidden.size(1) > 1 else hidden.mean(dim=1))
        frame_embeddings = torch.cat(chunks, dim=0).reshape(batch_size, num_frames, -1)
        lengths = torch.full((batch_size,), num_frames, device=pixels.device, dtype=torch.long)
        return frame_embeddings, lengths

    def _extract_vision(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.uses_vision_backbone:
            return self.feature_extractors["vision"](batch["vision"], batch["vision_lengths"])
        pixels = batch["vision_pixel_values"]
        if pixels.dim() == 5 and self.vision_is_video_backbone:
            outputs = self._run_encoder(
                self.vision_encoder,
                self.freeze_vision_encoder,
                pixel_values=pixels,
            )
            hidden = outputs.last_hidden_state
            lengths = torch.full((hidden.size(0),), hidden.size(1), device=hidden.device, dtype=torch.long)
            return hidden, lengths
        if pixels.dim() == 5:
            return self._vision_sequence_from_frames(pixels)
        if pixels.dim() == 4:
            outputs = self._run_encoder(
                self.vision_encoder,
                self.freeze_vision_encoder,
                pixel_values=pixels,
            )
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return self._single_step_sequence(outputs.pooler_output)
            return self._single_step_sequence(outputs.last_hidden_state[:, 0])
        raise ValueError(f"Expected vision_pixel_values to be 4D or 5D, got shape {tuple(pixels.shape)}.")

    def _extract_sequences(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        sequences: Dict[str, torch.Tensor] = {}
        lengths_by_modality: Dict[str, torch.Tensor] = {}
        for modality in self.active_modalities:
            if modality == "text":
                sequence, lengths = self._extract_text(batch)
            elif modality == "audio":
                sequence, lengths = self._extract_audio(batch)
            elif modality == "vision":
                sequence, lengths = self._extract_vision(batch)
            else:
                raise ValueError(f"Unknown modality {modality!r}.")
            sequences[modality] = sequence
            lengths_by_modality[modality] = lengths
        return sequences, lengths_by_modality

    def _concat_modality_factors(self, factors: Dict[str, object], modality: str) -> torch.Tensor:
        parts = []
        if "common" in factors[modality]:
            parts.append(factors[modality]["common"])
        parts.extend(factors[modality]["pairwise"].values())
        if "private" in factors[modality]:
            parts.append(factors[modality]["private"])
        if "noise" in factors[modality]:
            parts.append(factors[modality]["noise"])
        return torch.cat(parts, dim=-1)

    def _forward_pnr(
        self,
        features: Dict[str, torch.Tensor],
        attention_weights: Dict[str, torch.Tensor],
    ) -> Dict[str, object]:
        factors: Dict[str, Dict[str, object]] = {
            modality: self.factorizers[modality](features[modality]) for modality in self.active_modalities
        }

        tokens = []
        token_types = []
        common_token = None
        common_branches = [
            factors[modality]["common"] for modality in self.active_modalities if "common" in factors[modality]
        ]
        if common_branches:
            common_token = torch.stack(common_branches, dim=1).mean(dim=1)
            tokens.append(common_token)
            token_types.append("common")

        pair_tokens: Dict[str, torch.Tensor] = {}
        if not self.disable_pairwise:
            for first, second in self.active_pairs:
                name = pair_name(first, second)
                if name in factors[first]["pairwise"] and name in factors[second]["pairwise"]:
                    pair_branches = [factors[first]["pairwise"][name], factors[second]["pairwise"][name]]
                    pair_token = 0.5 * (pair_branches[0] + pair_branches[1])
                    pair_tokens[name] = pair_token
                    tokens.append(pair_token)
                    token_types.append("pair")

        private_tokens = []
        for modality in self.active_modalities:
            if "private" in factors[modality]:
                private_tokens.append(factors[modality]["private"])
                tokens.append(factors[modality]["private"])
                token_types.append("private")

        sentiment_tokens = torch.stack(tokens, dim=1)
        noise_tokens = None
        if not self.disable_noise:
            noise_tokens = torch.stack([factors[modality]["noise"] for modality in self.active_modalities], dim=1)
        text_query = None
        if "text" in factors:
            text_factors = factors["text"]
            query_parts = []
            if "common" in text_factors:
                query_parts.append(text_factors["common"])
            else:
                query_parts.append(torch.zeros_like(features["text"]))
            if "private" in text_factors:
                query_parts.append(text_factors["private"])
            else:
                query_parts.append(torch.zeros_like(features["text"]))
            text_query = self.text_factor_query(torch.cat(query_parts, dim=-1))
        fused, fusion_diagnostics = self.pnr_fusion(sentiment_tokens, token_types, noise_tokens, text_query)
        pred = self.classifier(fused)

        reconstructions: Dict[str, torch.Tensor] = {}
        reconstruction_targets: Dict[str, torch.Tensor] = {}
        kl_losses: Dict[str, torch.Tensor] = {}
        if not self.disable_reconstruction:
            for modality in self.active_modalities:
                concat = self._concat_modality_factors(factors, modality)
                recon, kl = self.reconstructors[modality](concat)
                reconstructions[modality] = recon
                reconstruction_targets[modality] = features[modality].detach()
                kl_losses[modality] = kl

        aux_preds: Dict[str, torch.Tensor] = {}
        if common_token is not None:
            aux_preds["common"] = self.aux_heads["common"](common_token)
        if private_tokens:
            private_info = torch.stack(private_tokens, dim=1).mean(dim=1)
            aux_preds["private"] = self.aux_heads["private"](private_info)
        if pair_tokens:
            ordered_pair_tokens = list(pair_tokens.values())
            pair_info = torch.stack(ordered_pair_tokens, dim=1).mean(dim=1)
            aux_preds["pair"] = self.aux_heads["pair"](pair_info)

        return {
            "pred": pred,
            "features": features,
            "factors": factors,
            "pair_tokens": pair_tokens,
            "sentiment_tokens": sentiment_tokens,
            "noise_tokens": noise_tokens,
            "reconstructions": reconstructions,
            "reconstruction_targets": reconstruction_targets,
            "kl_losses": kl_losses,
            "aux_preds": aux_preds,
            "fusion": {
                "attention_weights": attention_weights,
                "fused": fused,
                "pnr": fusion_diagnostics,
            },
            "active_modalities": self.active_modalities,
            "model_variant": self.model_variant,
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, object]:
        sequences, lengths_by_modality = self._extract_sequences(batch)
        mapped_sequences = {modality: self.mappers[modality](sequence) for modality, sequence in sequences.items()}

        text_feature = None
        features: Dict[str, torch.Tensor] = {}
        if "text" in self.active_modalities:
            text_feature = masked_mean(mapped_sequences["text"], lengths_by_modality["text"])
            features["text"] = text_feature
        attention_weights: Dict[str, torch.Tensor] = {}

        if self.model_variant == "pnr_trifuse":
            for modality in self.active_modalities:
                if modality == "text":
                    continue
                pooled, weights = self.intrinsic_poolers[modality](
                    mapped_sequences[modality],
                    lengths_by_modality[modality],
                )
                features[modality] = pooled
                attention_weights[modality] = weights
            return self._forward_pnr(features, attention_weights)

        for modality in self.active_modalities:
            if modality == "text":
                continue
            if text_feature is None:
                raise ValueError("The minimal text-guided model requires the text modality.")
            pooled, weights = self.attention_poolers[modality](
                mapped_sequences[modality],
                text_feature,
                lengths_by_modality[modality],
            )
            features[modality] = pooled
            attention_weights[modality] = weights

        fused = torch.cat([features[modality] for modality in self.active_modalities], dim=-1)
        pred = self.classifier(fused)
        return {
            "pred": pred,
            "features": features,
            "fusion": {
                "attention_weights": attention_weights,
                "fused": fused,
            },
            "active_modalities": self.active_modalities,
        }
