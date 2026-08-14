from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .media import MediaResolver, load_audio_waveform, load_video_frames


DATASET_FILES = {
    "mosi": [
        "MOSI/Processed/aligned_50.pkl",
        "CMU-MOSI/Processed/aligned_50.pkl",
        "aligned_50.pkl",
    ],
    "mosei": [
        "MOSEI/Processed/aligned_50.pkl",
        "MOSEI/Processed/aligned_50-003.pkl",
        "MOSEI/Processed/unaligned_50-002.pkl",
        "CMU-MOSEI/Processed/aligned_50.pkl",
        "aligned_50.pkl",
        "aligned_50-003.pkl",
    ],
    "simsv2": [
        "SIMSv2/Processed/unaligned-004.pkl",
        "SIMSv2/Processed/train_mix.pkl",
        "SIMSv2/Processed/SimsLargeV6.pkl",
        "SIMSv2/Processed/CHSims_aligned2.pkl",
        "SIMSv2/Processed/unaligned-002.pkl",
        "SIMSv2/Processed/sims_unaligned.pkl",
        "CH-SIMS-v2/Processed/sims_unaligned.pkl",
        "sims_unaligned.pkl",
    ],
}

SPLIT_ALIASES = {
    "train": ["train"],
    "valid": ["valid", "val", "dev"],
    "test": ["test"],
}

MODALITY_KEYS = {
    "text": ["text", "language", "bert", "text_bert", "textual"],
    "audio": ["audio", "acoustic", "A"],
    "vision": ["vision", "visual", "V"],
}

LABEL_KEYS = ["regression_labels", "labels", "label", "sentiment", "Y"]


def resolve_dataset_file(data_root: str | Path, dataset: str, explicit_file: Optional[str] = None) -> Path:
    root = Path(data_root)
    if explicit_file:
        path = Path(explicit_file)
        if path.is_absolute() or path.exists():
            return path
        return root / path

    dataset = dataset.lower()
    if dataset not in DATASET_FILES:
        raise ValueError(f"Unknown dataset {dataset!r}. Expected one of {sorted(DATASET_FILES)}")

    for rel in DATASET_FILES[dataset]:
        path = root / rel
        if path.exists():
            return path
    candidates = "\n".join(str(root / rel) for rel in DATASET_FILES[dataset])
    raise FileNotFoundError(f"Could not find processed file for {dataset}. Checked:\n{candidates}")


def load_pickle(path: str | Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _find_key(container: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    lower = {str(k).lower(): k for k in container.keys()}
    for key in keys:
        if key in container:
            return key
        if key.lower() in lower:
            return lower[key.lower()]
    return None


def _split_data(all_data: Dict[str, Any], split: str) -> Dict[str, Any]:
    for alias in SPLIT_ALIASES[split]:
        if alias in all_data:
            return all_data[alias]
    raise KeyError(f"Split {split!r} not found. Available top-level keys: {list(all_data.keys())}")


def _as_sequence_array(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = np.asarray(value.tolist(), dtype=np.float32)
    arr = np.nan_to_num(arr.astype(np.float32), copy=False)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def _infer_length(seq: np.ndarray) -> int:
    if seq.ndim == 1:
        return 1
    nonzero = np.abs(seq).sum(axis=-1) > 1e-8
    length = int(nonzero.sum())
    return length if length > 0 else int(seq.shape[0])


def _get_length(split_data: Dict[str, Any], base_key: str, idx: int, seq: np.ndarray) -> int:
    for suffix in ["_lengths", "_lens", "_len", " lengths"]:
        key = f"{base_key}{suffix}"
        if key in split_data:
            return int(np.asarray(split_data[key])[idx])
    return _infer_length(seq)


def _extract_labels(split_data: Dict[str, Any]) -> np.ndarray:
    key = _find_key(split_data, LABEL_KEYS)
    if key is None:
        raise KeyError(f"Could not find regression labels. Available keys: {list(split_data.keys())}")
    labels = np.asarray(split_data[key])
    if labels.ndim > 1:
        labels = labels.reshape(labels.shape[0], -1)[:, 0]
    return labels.astype(np.float32)


class MMSAPickleDataset(Dataset):
    """Dataset wrapper for MMSA-style processed pickle files."""

    def __init__(
        self,
        data_file: str | Path,
        split: str,
        dataset: str,
        raw_media_root: str | Path | None = None,
        media_cache_root: str | Path | None = None,
        load_raw_audio: bool = False,
        load_raw_vision: bool = False,
        audio_sample_rate: int = 16000,
        audio_max_seconds: float = 8.0,
        vision_num_frames: int = 16,
        vision_frame_size: int = 224,
    ):
        self.data_file = Path(data_file)
        self.split = split
        self.dataset = dataset.lower()
        self.load_raw_audio = load_raw_audio
        self.load_raw_vision = load_raw_vision
        self.audio_sample_rate = audio_sample_rate
        self.audio_max_seconds = audio_max_seconds
        self.vision_num_frames = vision_num_frames
        self.vision_frame_size = vision_frame_size
        self.media_resolver = None
        if self.load_raw_audio or self.load_raw_vision:
            if raw_media_root is None or media_cache_root is None:
                raise ValueError("raw_media_root and media_cache_root are required for raw audio/video loading.")
            self.media_resolver = MediaResolver(self.dataset, raw_media_root, media_cache_root)
        self.raw = load_pickle(self.data_file)
        self.split_data = _split_data(self.raw, split)
        self.labels = _extract_labels(self.split_data)
        self.modality_key_map = {}
        for modality, keys in MODALITY_KEYS.items():
            key = _find_key(self.split_data, keys)
            if key is None:
                raise KeyError(
                    f"Could not find {modality} feature in {self.data_file}. "
                    f"Available split keys: {list(self.split_data.keys())}"
                )
            self.modality_key_map[modality] = key

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {"label": float(self.labels[idx])}
        if "raw_text" in self.split_data:
            raw_text = np.asarray(self.split_data["raw_text"]).reshape(-1)
            item["raw_text"] = str(raw_text[idx])
        for modality, key in self.modality_key_map.items():
            values = self.split_data[key]
            seq = _as_sequence_array(values[idx])
            length = min(_get_length(self.split_data, key, idx, seq), seq.shape[0])
            item[modality] = seq
            item[f"{modality}_length"] = max(1, length)
        if "id" in self.split_data:
            item["id"] = self.split_data["id"][idx]
        if self.load_raw_audio or self.load_raw_vision:
            if self.media_resolver is None or "id" not in item:
                raise KeyError("Sample ids are required for raw audio/video loading.")
            media_path = self.media_resolver.resolve(str(item["id"]))
            if media_path is None:
                raise FileNotFoundError(f"Could not resolve raw media for sample id {item['id']!r}.")
            item["media_path"] = str(media_path)
            if self.load_raw_audio:
                waveform, audio_mask = load_audio_waveform(
                    media_path,
                    sample_rate=self.audio_sample_rate,
                    max_seconds=self.audio_max_seconds,
                )
                item["audio_waveform"] = waveform
                item["audio_attention_mask"] = audio_mask
            if self.load_raw_vision:
                item["vision_pixels"] = load_video_frames(
                    media_path,
                    num_frames=self.vision_num_frames,
                    size=self.vision_frame_size,
                )
        return item

    def feature_dims(self) -> Dict[str, int]:
        dims = {}
        for modality, key in self.modality_key_map.items():
            values = self.split_data[key]
            dims[modality] = int(_as_sequence_array(values[0]).shape[-1])
        return dims


def collate_mmsa(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, torch.Tensor] = {}
    for modality in ["text", "audio", "vision"]:
        seqs = [torch.as_tensor(item[modality], dtype=torch.float32) for item in batch]
        lengths = torch.as_tensor([item[f"{modality}_length"] for item in batch], dtype=torch.long)
        padded = pad_sequence(seqs, batch_first=True)
        max_len = padded.size(1)
        mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
        out[modality] = padded
        out[f"{modality}_lengths"] = lengths
        out[f"{modality}_mask"] = mask
    out["label"] = torch.as_tensor([item["label"] for item in batch], dtype=torch.float32)
    if "raw_text" in batch[0]:
        out["raw_text"] = [str(item["raw_text"]) for item in batch]
    if "audio_waveform" in batch[0]:
        waveforms = [item["audio_waveform"].to(torch.float32) for item in batch]
        masks = [item["audio_attention_mask"].to(torch.long) for item in batch]
        out["audio_input_values"] = pad_sequence(waveforms, batch_first=True)
        out["audio_attention_mask"] = pad_sequence(masks, batch_first=True)
    if "vision_pixels" in batch[0]:
        out["vision_pixel_values"] = torch.stack([item["vision_pixels"].to(torch.float32) for item in batch], dim=0)
    return out


def make_collate_mmsa(tokenizer=None, text_max_length: int = 50) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = collate_mmsa(batch)
        if tokenizer is not None:
            if "raw_text" not in out:
                raise KeyError("raw_text is required when a text tokenizer is configured.")
            encoded = tokenizer(
                out["raw_text"],
                padding="max_length",
                truncation=True,
                max_length=text_max_length,
                return_tensors="pt",
            )
            for key, value in encoded.items():
                out[f"text_{key}"] = value
        return out

    return collate


def make_datasets(
    data_root: str | Path,
    dataset: str,
    data_file: Optional[str] = None,
    raw_media_root: str | Path | None = None,
    media_cache_root: str | Path | None = None,
    load_raw_audio: bool = False,
    load_raw_vision: bool = False,
    audio_sample_rate: int = 16000,
    audio_max_seconds: float = 8.0,
    vision_num_frames: int = 16,
    vision_frame_size: int = 224,
) -> Tuple[MMSAPickleDataset, MMSAPickleDataset, MMSAPickleDataset]:
    path = resolve_dataset_file(data_root, dataset, data_file)
    return (
        MMSAPickleDataset(
            path,
            "train",
            dataset,
            raw_media_root,
            media_cache_root,
            load_raw_audio,
            load_raw_vision,
            audio_sample_rate,
            audio_max_seconds,
            vision_num_frames,
            vision_frame_size,
        ),
        MMSAPickleDataset(
            path,
            "valid",
            dataset,
            raw_media_root,
            media_cache_root,
            load_raw_audio,
            load_raw_vision,
            audio_sample_rate,
            audio_max_seconds,
            vision_num_frames,
            vision_frame_size,
        ),
        MMSAPickleDataset(
            path,
            "test",
            dataset,
            raw_media_root,
            media_cache_root,
            load_raw_audio,
            load_raw_vision,
            audio_sample_rate,
            audio_max_seconds,
            vision_num_frames,
            vision_frame_size,
        ),
    )
