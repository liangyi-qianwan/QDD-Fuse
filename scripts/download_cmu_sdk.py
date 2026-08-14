from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


def _video_id(segment_id: str) -> str:
    text = str(segment_id)
    if "[" in text:
        return text.split("[", 1)[0]
    return text.split("_", 1)[0]


def _safe_features(dataset, sequence_name: str, segment_id: str) -> np.ndarray:
    data = dataset.computational_sequences[sequence_name].data
    if segment_id not in data:
        return np.zeros((1, 1), dtype=np.float32)
    feat = np.asarray(data[segment_id]["features"], dtype=np.float32)
    feat = np.nan_to_num(feat, copy=False)
    if feat.ndim == 1:
        feat = feat.reshape(1, -1)
    return feat


def _fold_sets(standard_folds) -> Tuple[set, set, set]:
    train = set(getattr(standard_folds, "standard_train_fold"))
    valid = set(getattr(standard_folds, "standard_valid_fold"))
    test = set(getattr(standard_folds, "standard_test_fold"))
    return train, valid, test


def collapse_average(intervals, features):
    if features is None or len(features) == 0:
        return np.zeros((1,), dtype=np.float32)
    return np.asarray(features, dtype=np.float32).mean(axis=0)


def _assign_split(segment_id: str, folds: Tuple[set, set, set]) -> str:
    vid = _video_id(segment_id)
    train, valid, test = folds
    if vid in train:
        return "train"
    if vid in valid:
        return "valid"
    if vid in test:
        return "test"
    return "train"


def _as_object_array(values: Iterable[np.ndarray]) -> np.ndarray:
    return np.asarray(list(values), dtype=object)


def convert_dataset(
    dataset,
    label_sequence: str,
    text_sequence: str,
    audio_sequence: str,
    vision_sequence: str,
    folds,
) -> Dict[str, dict]:
    split_items = {"train": [], "valid": [], "test": []}
    labels = dataset.computational_sequences[label_sequence].data
    for segment_id in sorted(labels.keys()):
        label_feat = np.asarray(labels[segment_id]["features"], dtype=np.float32).reshape(-1)
        if label_feat.size == 0:
            continue
        item = {
            "id": segment_id,
            "text": _safe_features(dataset, text_sequence, segment_id),
            "audio": _safe_features(dataset, audio_sequence, segment_id),
            "vision": _safe_features(dataset, vision_sequence, segment_id),
            "regression_labels": float(label_feat[0]),
        }
        split_items[_assign_split(segment_id, folds)].append(item)

    output = {}
    for split, items in split_items.items():
        output[split] = {
            "id": np.asarray([item["id"] for item in items]),
            "text": _as_object_array(item["text"] for item in items),
            "audio": _as_object_array(item["audio"] for item in items),
            "vision": _as_object_array(item["vision"] for item in items),
            "regression_labels": np.asarray([item["regression_labels"] for item in items], dtype=np.float32),
        }
    return output


def download_and_convert(dataset_name: str, root: Path, work_dir: Path) -> Path:
    from mmsdk import mmdatasdk

    if dataset_name == "mosi":
        spec = {
            "highlevel": mmdatasdk.cmu_mosi.highlevel,
            "labels": mmdatasdk.cmu_mosi.labels,
            "label_sequence": "Opinion Segment Labels",
            "text": "glove_vectors",
            "audio": "COVAREP",
            "vision": "FACET 4.1",
            "folds": mmdatasdk.cmu_mosi.standard_folds,
            "output": root / "MOSI/Processed/aligned_50.pkl",
        }
    elif dataset_name == "mosei":
        spec = {
            "highlevel": mmdatasdk.cmu_mosei.highlevel,
            "labels": mmdatasdk.cmu_mosei.labels,
            "label_sequence": "Sentiment Labels",
            "text": "glove_vectors",
            "audio": "COVAREP",
            "vision": "FACET 4.2",
            "folds": mmdatasdk.cmu_mosei.standard_folds,
            "output": root / "MOSEI/Processed/aligned_50.pkl",
        }
    else:
        raise ValueError("CMU SDK route only supports mosi and mosei.")

    cache_dir = work_dir / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cmu-sdk] downloading high-level sequences into {cache_dir}")
    dataset = mmdatasdk.mmdataset(spec["highlevel"], str(cache_dir))
    print(f"[cmu-sdk] word-aligning to {spec['text']}")
    dataset.align(spec["text"], collapse_functions=[collapse_average])
    dataset.add_computational_sequences(spec["labels"], str(cache_dir))
    print(f"[cmu-sdk] aligning to {spec['label_sequence']}")
    dataset.align(spec["label_sequence"])
    data = convert_dataset(
        dataset,
        spec["label_sequence"],
        spec["text"],
        spec["audio"],
        spec["vision"],
        _fold_sets(spec["folds"]),
    )
    output = spec["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(data, f)
    print(f"[cmu-sdk] wrote {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MOSI/MOSEI through CMU Multimodal SDK and convert to training pickle.")
    parser.add_argument("--datasets", nargs="+", default=["mosi", "mosei"], choices=["mosi", "mosei"])
    parser.add_argument("--root", default="data")
    parser.add_argument("--work-dir", default="data/cmu_sdk_raw")
    args = parser.parse_args()
    root = Path(args.root)
    work_dir = Path(args.work_dir)
    for dataset_name in args.datasets:
        download_and_convert(dataset_name, root, work_dir)


if __name__ == "__main__":
    main()
