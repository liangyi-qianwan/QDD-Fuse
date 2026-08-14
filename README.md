# QDD-Fuse

Repository: https://github.com/liangyi-qianwan/QDD-Fuse

Release package: https://github.com/liangyi-qianwan/QDD-Fuse/releases/tag/qdd-fuse-repro-20260814

QDD-Fuse is a reproducible multimodal sentiment analysis package for MOSI, MOSEI, and SIMSv2. The source tree contains the model code, retest scripts, configuration files, metric records, and lightweight text-encoder config/tokenizer files. The large `best.pt` checkpoint files are distributed through the GitHub Release archive instead of being committed directly to the repository.

## Repository Layout

- `fusenet/`: model, loss, data, media, and metrics modules.
- `train.py`: training entry point.
- `scripts/retest_pnr_checkpoint.py`: retest one saved checkpoint.
- `scripts/run_retest_all.sh`: retest MOSI, MOSEI, and SIMSv2.
- `configs/`: dataset configuration files.
- `checkpoints/`: checkpoint placeholders and text-encoder metadata.
- `results/`: saved metrics, logs, hyperparameter summary, and verification records.

The GitHub source tree does not contain the large files below:

```text
checkpoints/mosi/best.pt
checkpoints/mosei/best.pt
checkpoints/simsv2/best.pt
```

They appear after extracting the Release archive.

## Dataset Layout

Datasets are not included in this repository. Prepare MMSA-style processed files locally and pass their parent directory as `DATA_ROOT`.

Expected layout:

```text
DATA_ROOT/
  MOSI/
    Processed/
      aligned_50.pkl
  MOSEI/
    Processed/
      aligned_50.pkl
  SIMSv2/
    Processed/
      sims_unaligned.pkl
```

## Install

Create or activate a Python environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

Use a CUDA-enabled PyTorch build if you want GPU retesting.

## Get Checkpoints

Download all Release assets from:

https://github.com/liangyi-qianwan/QDD-Fuse/releases/tag/qdd-fuse-repro-20260814

Required assets:

- `QDD-Fuse_repro_20260814.tar.part-aa`
- `QDD-Fuse_repro_20260814.tar.part-ac`
- `QDD-Fuse_repro_20260814.tar.parts.sha256`
- `QDD-Fuse_repro_20260814.tar.part-ab.chunk-*`
- `QDD-Fuse_repro_20260814.tar.part-ab.chunks.sha256`

Reconstruct and extract the full package:

```bash
cat QDD-Fuse_repro_20260814.tar.part-ab.chunk-* > QDD-Fuse_repro_20260814.tar.part-ab
sha256sum -c QDD-Fuse_repro_20260814.tar.part-ab.chunks.sha256
sha256sum -c QDD-Fuse_repro_20260814.tar.parts.sha256
cat QDD-Fuse_repro_20260814.tar.part-aa \
  QDD-Fuse_repro_20260814.tar.part-ab \
  QDD-Fuse_repro_20260814.tar.part-ac > QDD-Fuse.tar
tar -xf QDD-Fuse.tar
```

Run retesting from the extracted `QDD-Fuse/` directory, or copy the extracted `checkpoints/{dataset}/best.pt` files into a cloned repository with the same paths.

## Retest

Run all checkpoint retests:

```bash
cd QDD-Fuse
DATA_ROOT=/absolute/path/to/DATA_ROOT PYTHON=python bash scripts/run_retest_all.sh
```

Run one dataset:

```bash
python scripts/retest_pnr_checkpoint.py \
  --checkpoint checkpoints/mosi/best.pt \
  --output-json results/retest/mosi_retest.json \
  --data-root /absolute/path/to/DATA_ROOT
```

## Expected Metrics

The full retest should match the saved results up to minor floating-point noise:

| Dataset | MAE | Corr | Main metrics |
|---|---:|---:|---|
| MOSI | 0.671797 | 0.819266 | Acc2 No0 86.890 / F1 No0 86.863 |
| MOSEI | 0.521774 | 0.782800 | Acc2 No0 86.764 / F1 No0 86.727 |
| SIMSv2 | 0.291396 | 0.722231 | Acc2 80.271 / F1 80.309 / Acc3 73.888 / Acc5 57.544 |

Saved verification files are under `results/`.

## 中文使用说明

### 1. 克隆代码库

```bash
git clone https://github.com/liangyi-qianwan/QDD-Fuse.git
cd QDD-Fuse
```

普通 GitHub 源码树里不会直接包含三份 `.pt` checkpoint，因为单个文件超过 GitHub 普通文件大小限制。真实 checkpoint 在 Release 分卷包里。

### 2. 准备数据集

本仓库不上传数据集。请在本地准备 MMSA 风格的处理后数据，并记住其上一级目录路径，后续用 `DATA_ROOT` 指向它。

目录结构应为：

```text
DATA_ROOT/
  MOSI/Processed/aligned_50.pkl
  MOSEI/Processed/aligned_50.pkl
  SIMSv2/Processed/sims_unaligned.pkl
```

### 3. 下载 Release 分卷

进入 Release 页面下载所有分卷文件：

https://github.com/liangyi-qianwan/QDD-Fuse/releases/tag/qdd-fuse-repro-20260814

需要下载：

```text
QDD-Fuse_repro_20260814.tar.part-aa
QDD-Fuse_repro_20260814.tar.part-ac
QDD-Fuse_repro_20260814.tar.parts.sha256
QDD-Fuse_repro_20260814.tar.part-ab.chunk-*
QDD-Fuse_repro_20260814.tar.part-ab.chunks.sha256
```

### 4. 还原完整复现包

在这些文件所在目录执行：

```bash
cat QDD-Fuse_repro_20260814.tar.part-ab.chunk-* > QDD-Fuse_repro_20260814.tar.part-ab
sha256sum -c QDD-Fuse_repro_20260814.tar.part-ab.chunks.sha256
sha256sum -c QDD-Fuse_repro_20260814.tar.parts.sha256
cat QDD-Fuse_repro_20260814.tar.part-aa \
  QDD-Fuse_repro_20260814.tar.part-ab \
  QDD-Fuse_repro_20260814.tar.part-ac > QDD-Fuse.tar
tar -xf QDD-Fuse.tar
```

解压后会得到完整的 `QDD-Fuse/` 目录，其中包含：

```text
checkpoints/mosi/best.pt
checkpoints/mosei/best.pt
checkpoints/simsv2/best.pt
```

### 5. 安装环境

进入解压后的 `QDD-Fuse/` 目录或克隆仓库目录：

```bash
python -m pip install -r requirements.txt
```

如果使用 GPU，请确保 PyTorch 与本机 CUDA 环境匹配。

### 6. 复测全部 checkpoint

```bash
DATA_ROOT=/absolute/path/to/DATA_ROOT PYTHON=python bash scripts/run_retest_all.sh
```

### 7. 单独复测一个数据集

```bash
python scripts/retest_pnr_checkpoint.py \
  --checkpoint checkpoints/mosi/best.pt \
  --output-json results/retest/mosi_retest.json \
  --data-root /absolute/path/to/DATA_ROOT
```

### 8. 结果核对

复测输出应与 `results/` 下保存的指标基本一致。主要指标如下：

| 数据集 | MAE | Corr | 主要分类指标 |
|---|---:|---:|---|
| MOSI | 0.671797 | 0.819266 | Acc2 No0 86.890 / F1 No0 86.863 |
| MOSEI | 0.521774 | 0.782800 | Acc2 No0 86.764 / F1 No0 86.727 |
| SIMSv2 | 0.291396 | 0.722231 | Acc2 80.271 / F1 80.309 / Acc3 73.888 / Acc5 57.544 |
