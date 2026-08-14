from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import torch


def parse_sample_id(sample_id: str) -> Optional[Tuple[str, str]]:
    if "$_$" not in sample_id:
        return None
    video_id, clip_id = sample_id.split("$_$", 1)
    return video_id, clip_id


class MediaResolver:
    """Resolve MMSA sample ids to raw mp4 files, extracting zip entries lazily."""

    def __init__(self, dataset: str, raw_media_root: str | Path, cache_root: str | Path):
        self.dataset = dataset.lower()
        self.raw_media_root = Path(raw_media_root)
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._zip_index: Optional[Dict[str, Tuple[Path, str]]] = None

    def resolve(self, sample_id: str) -> Optional[Path]:
        parsed = parse_sample_id(sample_id)
        if parsed is None:
            return None
        video_id, clip_id = parsed
        for name in self._candidate_names(clip_id):
            for root in self._candidate_roots(video_id):
                candidate = root / name
                if candidate.exists() and not name.startswith("._"):
                    return candidate
            cached = self.cache_root / video_id / name
            if cached.exists():
                return cached

        index = self._build_zip_index()
        for name in self._candidate_names(clip_id):
            key = f"{video_id}/{name}"
            if key not in index:
                continue
            zip_path, entry = index[key]
            out_path = self.cache_root / video_id / name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not out_path.exists():
                with zipfile.ZipFile(zip_path) as zf, zf.open(entry) as src, out_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            return out_path
        return None

    def _candidate_roots(self, video_id: str) -> Iterable[Path]:
        yield self.raw_media_root / video_id
        yield self.raw_media_root / "Raw" / video_id
        yield self.raw_media_root / "extracted" / video_id
        yield self.raw_media_root.parent / "Raw" / video_id
        yield self.raw_media_root.parent / "extracted" / video_id

    def _candidate_names(self, clip_id: str):
        yield f"{clip_id}.mp4"
        try:
            number = int(clip_id)
        except ValueError:
            return
        yield f"{number}.mp4"
        yield f"{number:04d}.mp4"
        yield f"{number:05d}.mp4"

    def _archive_dirs(self) -> Iterable[Path]:
        seen = set()
        for path in [
            self.raw_media_root / "archives",
            self.raw_media_root.parent / "archives",
            self.raw_media_root,
            self.raw_media_root.parent,
        ]:
            if path in seen:
                continue
            seen.add(path)
            if path.exists():
                yield path

    def _build_zip_index(self) -> Dict[str, Tuple[Path, str]]:
        if self._zip_index is not None:
            return self._zip_index
        index: Dict[str, Tuple[Path, str]] = {}
        for archive_dir in self._archive_dirs():
            for zip_path in sorted(archive_dir.glob("*.zip")):
                with zipfile.ZipFile(zip_path) as zf:
                    for entry in zf.namelist():
                        if not entry.lower().endswith(".mp4"):
                            continue
                        path = Path(entry)
                        if path.name.startswith("._") or len(path.parts) < 3:
                            continue
                        if path.parts[-3] != "Raw":
                            continue
                        key = f"{path.parts[-2]}/{path.name}"
                        index.setdefault(key, (zip_path, entry))
        self._zip_index = index
        return index


def load_audio_waveform(path: str | Path, sample_rate: int = 16000, max_seconds: float = 8.0) -> Tuple[torch.Tensor, torch.Tensor]:
    max_samples = int(sample_rate * max_seconds)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-t",
        str(max_seconds),
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    waveform = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    waveform = np.nan_to_num(waveform, copy=False)
    if waveform.size == 0:
        waveform = np.zeros(1, dtype=np.float32)
    waveform = waveform[:max_samples]
    if waveform.size > 1:
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-7)
    mask = np.ones(waveform.shape[0], dtype=np.int64)
    return torch.from_numpy(waveform), torch.from_numpy(mask)


def load_video_frames(path: str | Path, num_frames: int = 16, size: int = 224) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        cap.release()
        return torch.zeros(num_frames, 3, size, size, dtype=torch.float32)

    indices = np.linspace(0, max(0, frame_count - 1), num=num_frames).astype(np.int64)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((size, size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    cap.release()
    arr = np.stack(frames, axis=0)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (0, 3, 1, 2))
    return torch.from_numpy(arr.astype(np.float32))
