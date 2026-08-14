from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional


TARGETS: Dict[str, Dict[str, object]] = {
    "mosi": {
        "output": "MOSI/Processed/aligned_50.pkl",
        "hf_repo": "tamb2203579/CMU-MOSI",
        "hf_files": ["Processed/aligned_50.pkl", "MOSI/Processed/aligned_50.pkl", "aligned_50.pkl"],
        "gdrive_suffixes": ["MOSI/Processed/aligned_50.pkl", "CMU-MOSI/Processed/aligned_50.pkl", "aligned_50.pkl"],
    },
    "mosei": {
        "output": "MOSEI/Processed/aligned_50.pkl",
        "hf_repo": "tamb2203579/CMU-MOSEI",
        "hf_files": ["Processed/aligned_50.pkl", "MOSEI/Processed/aligned_50.pkl", "aligned_50.pkl"],
        "gdrive_suffixes": ["MOSEI/Processed/aligned_50.pkl", "CMU-MOSEI/Processed/aligned_50.pkl", "aligned_50.pkl"],
    },
    "simsv2": {
        "output": "SIMSv2/Processed/sims_unaligned.pkl",
        "hf_repo": None,
        "hf_files": [],
        "gdrive_suffixes": ["SIMSv2/Processed/sims_unaligned.pkl", "CH-SIMS-v2/Processed/sims_unaligned.pkl", "sims_unaligned.pkl"],
    },
}

MMSA_GDRIVE_FOLDER = "https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk"


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_from_hf(target: Dict[str, object], out_path: Path) -> bool:
    repo = target.get("hf_repo")
    if not repo:
        return False
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        print(f"[hf] huggingface_hub unavailable: {exc}")
        return False

    endpoint = os.environ.get("HF_ENDPOINT")
    for filename in target.get("hf_files", []):
        try:
            print(f"[hf] trying {repo}:{filename}")
            downloaded = hf_hub_download(
                repo_id=str(repo),
                filename=str(filename),
                repo_type="dataset",
                endpoint=endpoint,
                local_dir=str(out_path.parent),
                local_dir_use_symlinks=False,
            )
            src = Path(downloaded)
            if src.resolve() != out_path.resolve():
                ensure_parent(out_path)
                shutil.copy2(src, out_path)
            return out_path.exists()
        except Exception as exc:
            print(f"[hf] failed {repo}:{filename}: {exc}")
    return False


def gdown_json(folder_url: str) -> Optional[list]:
    cmd = [sys.executable, "-m", "gdown", "--folder", "--json", folder_url]
    try:
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
    except Exception as exc:
        print(f"[gdrive] failed to list folder: {exc}")
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        print(f"[gdrive] could not parse JSON: {exc}")
        print(proc.stdout[:1000])
        return None


def download_from_gdrive(target: Dict[str, object], out_path: Path, folder_url: str) -> bool:
    listing = gdown_json(folder_url)
    if not listing:
        return False
    suffixes = [str(s).replace("\\", "/") for s in target.get("gdrive_suffixes", [])]
    match = None
    for entry in listing:
        path = str(entry.get("path", "")).replace("\\", "/")
        if any(path.endswith(suffix) for suffix in suffixes):
            match = entry
            break
    if not match:
        print(f"[gdrive] target not found in folder listing. Wanted suffixes: {suffixes}")
        return False
    url = match.get("url")
    if not url:
        print("[gdrive] matched entry has no URL")
        return False
    ensure_parent(out_path)
    cmd = [sys.executable, "-m", "gdown", "--continue", "-O", str(out_path), str(url)]
    try:
        print(f"[gdrive] downloading {match.get('path')} -> {out_path}")
        subprocess.run(cmd, check=True)
        return out_path.exists()
    except Exception as exc:
        print(f"[gdrive] download failed: {exc}")
        return False


def check_target(name: str, root: Path) -> bool:
    target = TARGETS[name]
    out_path = root / str(target["output"])
    if not out_path.exists():
        print(f"[missing] {name}: {out_path}")
        return False
    size = out_path.stat().st_size
    digest = sha256sum(out_path)
    print(f"[ok] {name}: {out_path} size={size} sha256={digest}")
    return True


def download_target(name: str, root: Path, sources: Iterable[str], gdrive_folder: str) -> bool:
    target = TARGETS[name]
    out_path = root / str(target["output"])
    if out_path.exists():
        return check_target(name, root)
    ensure_parent(out_path)
    for source in sources:
        if source == "hf" and download_from_hf(target, out_path):
            return check_target(name, root)
        if source == "gdrive" and download_from_gdrive(target, out_path, gdrive_folder):
            return check_target(name, root)
    print(f"[failed] {name}: could not download automatically.")
    print(f"Place the file manually at: {out_path}")
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MMSA processed features for FUSE-Net.")
    parser.add_argument("--datasets", nargs="+", default=["mosi", "mosei", "simsv2"], choices=sorted(TARGETS))
    parser.add_argument("--root", default="data")
    parser.add_argument("--sources", nargs="+", default=["hf", "gdrive"], choices=["hf", "gdrive"])
    parser.add_argument("--gdrive-folder", default=MMSA_GDRIVE_FOLDER)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    ok = True
    for name in args.datasets:
        if args.check_only:
            ok = check_target(name, root) and ok
        else:
            ok = download_target(name, root, args.sources, args.gdrive_folder) and ok
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
