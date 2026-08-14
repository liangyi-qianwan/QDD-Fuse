# Dataset Download Status

Date: 2026-07-19

Remote host: `emo`

Target files:

- `data/MOSI/Processed/aligned_50.pkl`
- `data/MOSEI/Processed/aligned_50.pkl`
- `data/SIMSv2/Processed/sims_unaligned.pkl`

Attempts from `emo`:

- Hugging Face API: timed out connecting to `huggingface.co:443`.
- Hugging Face mirror: timed out connecting to `hf-mirror.com:443`.
- Google Drive MMSA folder: timed out connecting to `drive.google.com:443`.
- CMU Multimodal SDK data server: timed out connecting to `immortal.multicomp.cs.cmu.edu`.
- Local workspace scan: no matching `aligned_50.pkl` or `sims_unaligned.pkl` files were found under `D:\python_code`.

Reachable services:

- PyPI mirror at `https://mirrors.aliyun.com/pypi/simple`
- GitHub repository API for lightweight search queries

Current result:

- Public dataset hosts were inaccessible from `emo`, but the local CH-SIMS v2 files were later moved from `C:\Users\10498\Downloads` to the remote workspace.
- SIMSv2 is now available through:
  `data/SIMSv2/Processed/sims_unaligned.pkl -> unaligned-004.pkl`
- The moved raw archives are stored in:
  `data/SIMSv2/Raw/archives/`
- MOSI/MOSEI are still not available unless downloaded later.
- Re-run checks with:

```bash
python scripts/download_datasets.py --check-only --datasets mosi mosei simsv2 --root data
```

Additional attempt:

- User-provided Google Drive folder:
  `https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk`
- Command:
  `python -m gdown --folder --no-cookies -O data/gdrive_raw <folder-url>`
- Result:
  failed before listing folder contents with `ConnectTimeoutError` to `drive.google.com:443`.
- Local non-sandbox HEAD check also failed to connect to `drive.google.com:443`.

Local transfer completion:

- `unaligned-004.pkl` copied to `data/SIMSv2/Processed/unaligned-004.pkl`.
- `unaligned-002.pkl` copied to `data/SIMSv2/Processed/unaligned-002.pkl`.
- `sims_unaligned.pkl` was created as a symlink to `unaligned-004.pkl`.
- 12 raw archive files `CH-SIMS v2-20260718T181151Z-1-005.zip` through `...-016.zip` copied to `data/SIMSv2/Raw/archives/`.
- Local `Downloads` copies of these `pkl` and `zip` files were deleted after remote size verification.
