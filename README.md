# QDD-Fuse

This repository is the reproducible package for the QDD-Fuse / PNR-TriFuse multimodal sentiment experiments.
It contains the minimal source code, configuration files, final checkpoints, packaged text-encoder metadata, and saved result logs needed to retest the reported checkpoints.

## Contents

- `fusenet/`: model, loss, data, media, and metrics modules.
- `train.py`: training entry point used for the original runs.
- `scripts/retest_pnr_checkpoint.py`: checkpoint retest entry point.
- `scripts/run_retest_all.sh`: one-command retest for MOSI, MOSEI, and SIMSv2.
- `configs/`: dataset configuration files.
- `checkpoints/{mosi,mosei,simsv2}/best.pt`: final QDD-Fuse checkpoints in the full release archive.
- `checkpoints/text_encoders/`: tokenizer/config files for the fine-tuned text encoders. The large encoder weight files are intentionally omitted because the checkpoint state dict already contains the required encoder weights.
- `results/`: saved metrics, logs, hyperparameter summary, and original retest JSON files.

The dataset files are not included. On `emo`, the verified data root is:

```bash
/hy-tmp/CVPR2026/data
```

Expected processed files:

```text
MOSI/Processed/aligned_50.pkl
MOSEI/Processed/aligned_50.pkl
SIMSv2/Processed/sims_unaligned.pkl
```

## Environment

The verified Python executable on `emo` is:

```bash
/hy-tmp/CVPR2026/.venvs/CVPR2026/bin/python
```

Install requirements in a compatible environment with:

```bash
python -m pip install -r requirements.txt
```

## Retest

Run all three checkpoint retests:

```bash
DATA_ROOT=/hy-tmp/CVPR2026/data PYTHON=/hy-tmp/CVPR2026/.venvs/CVPR2026/bin/python bash scripts/run_retest_all.sh
```

Run one dataset manually:

```bash
/hy-tmp/CVPR2026/.venvs/CVPR2026/bin/python scripts/retest_pnr_checkpoint.py \
  --checkpoint checkpoints/mosi/best.pt \
  --output-json results/retest/mosi_retest.json \
  --data-root /hy-tmp/CVPR2026/data
```

## Expected Metrics

The full retest should match the saved results up to minor floating-point noise:

| Dataset | MAE | Corr | Main Acc/F1 |
|---|---:|---:|---|
| MOSI | 0.671797 | 0.819266 | Acc2 No0 86.890 / F1 No0 86.863 |
| MOSEI | 0.521774 | 0.782800 | Acc2 No0 86.764 / F1 No0 86.727 |
| SIMSv2 | 0.291396 | 0.722231 | Acc2 80.271 / F1 80.309 |

See `results/best_hparams_summary.json` and `results/original_retest/` for the saved verification records.

## GitHub Release Archive

The `best.pt` files are larger than GitHub's normal 100 MB file limit. In the GitHub repository source tree, they are not committed directly. Download the `QDD-Fuse.tar.part-*` files from the release assets, then reconstruct the full package:

```bash
cat QDD-Fuse.tar.part-* > QDD-Fuse.tar
sha256sum -c QDD-Fuse.tar.parts.sha256
tar -xf QDD-Fuse.tar
```

The extracted `QDD-Fuse/` directory contains the checkpoints and can run the retest commands above.
