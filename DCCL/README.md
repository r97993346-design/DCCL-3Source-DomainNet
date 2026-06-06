# Connecting Domains and Contrasting Samples: A Ladder for Domain Generalization (DCCL)

[![KDD 2025](https://img.shields.io/badge/KDD-2025-blue)](https://kdd2025.kdd.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-latest-red)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.6+-green)](https://python.org/)

Official implementation of **"Connecting Domains and Contrasting Samples: A Ladder for Domain Generalization"** (KDD 2025).

## 📋 Overview

DCCL (Domain-Aware Contrastive Cross-domain Learning) is a novel approach for domain generalization that combines multiple complementary loss components to learn robust representations across different domains. The algorithm integrates:

- **Cross-entropy loss** for standard classification
- **Contrastive loss** between aggressively augmented views for invariant representation learning
- **Layer-wise contrastive loss** for contrastive feature alignment with pre-trained models
- **Generative alignment regularization** to generative align features with pre-trained knowledge

## 🏗️ Code Structure

```
data/
DCCL/
├── train_all.py                    # Main training script
├── config.yaml                     # Configuration file
├── domainbed/
│   ├── algorithms/
│   │   └── algorithms.py           # 🔥 Core DCCL algorithm implementation
│   ├── lib/
│   │   └── cl_hparams.py           # 🔥 Core hyperparameter settings
│   ├── datasets/                   # Dataset loaders
│   ├── networks.py                 # Network architectures
│   ├── trainer.py                  # Training loop
│   └── ...
└── ...
```

### Key Files:
- **`DCCL/domainbed/algorithms/algorithms.py`**: Contains the main DCCL algorithm with detailed comments explaining each loss component
- **`DCCL/domainbed/lib/cl_hparams.py`**: Core hyperparameter configurations for different datasets

## 🚀 Algorithm Flow

The DCCL algorithm follows this training procedure:

1. **Data Preparation**: Load original and augmented image pairs
2. **Feature Extraction**: Extract features using trainable and frozen pre-trained networks
3. **Multi-Loss Computation**:
   - Classification loss (always active)
   - Contrastive loss (controlled by `--l`)
   - Domain alignment loss (controlled by `--l_d`) 
   - Layer-wise contrastive loss (controlled by `--l_layer`)
4. **Optimization**: Multi-component loss backpropagation with different learning rates

## ⚙️ Core Hyperparameters

The main tuning parameters are located in `DCCL/domainbed/lib/cl_hparams.py`:

### Essential Parameters (main tuning focus):
- `--l`: Weight for contrastive loss (default: 1.0)
- `--l_d`: Weight for domain alignment loss (default: 0.05) 
- `--l_layer`: Weight for layer-wise contrastive loss (default: 1.0)
- `--t`: Temperature for contrastive loss (default: 0.1)
- `--t_pre`: Temperature for pre-trained feature loss (default: 0.2)
- `--n_layer`: Number of layers in projection head (default: 1)

## 🛠️ Installation

### Environment Requirements

```
Python: 3.6+
PyTorch: latest
Torchvision: 0.10.0
CUDA: 10.2
CUDNN: 7605
NumPy: 1.19.5
PIL: 7.2.0
```

### Setup
You can
```bash
git clone <this-repo>
cd DCCL/
pip install -r requirements.txt
```

## 📁 Data Preparation

Each dataset can be easily accessed from official sources. For example, the VLCS dataset can be found on the official [repo](https://github.com/belaalb/G2DM#download-vlcs).

To download all datasets automatically:

```bash
python download.py --data_dir data
```

## 🏃‍♂️ Running Experiments

### Basic Usage

Navigate to the DCCL directory and run:

We default set ```--algorithm DCCL```

```bash
cd DCCL/
python train_all.py DCCL_OH_0 --dataset OfficeHome --deterministic --trial_seed 0 --checkpoint_freq 100 --data_dir ../data
```

### Multiple Seeds

```bash
python train_all.py DCCL_OH_0 --dataset OfficeHome --deterministic --trial_seed 0 --checkpoint_freq 100 --data_dir ../data
python train_all.py DCCL_OH_1 --dataset OfficeHome --deterministic --trial_seed 1 --checkpoint_freq 100 --data_dir ../data
python train_all.py DCCL_OH_2 --dataset OfficeHome --deterministic --trial_seed 2 --checkpoint_freq 100 --data_dir ../data
```

### Different Configurations

**Different Backbone Models:**
```bash
# CLIP ViT-B/16
python train_all.py DCCL_OH_vit --dataset OfficeHome --deterministic --trial_seed 2 --checkpoint_freq 100 --data_dir ../data --model clip_vit-b16

# RegNet
python train_all.py DCCL_OH_reg --dataset OfficeHome --deterministic --trial_seed 2 --checkpoint_freq 100 --data_dir ../data --model regnet
```

**Limited Labeled Data:**
```bash
# 10% labeled data
python train_all.py DCCL_OH_res50_0.1 --dataset OfficeHome --deterministic --trial_seed 2 --checkpoint_freq 100 --data_dir ../data --label_ratio 0.1
```

**Different Datasets:**
```bash
# PACS
python train_all.py DCCL_PACS_0 --dataset PACS --deterministic --trial_seed 0 --checkpoint_freq 100 --data_dir ../data

# VLCS  
python train_all.py DCCL_VLCS_0 --dataset VLCS --deterministic --trial_seed 0 --checkpoint_freq 100 --data_dir ../data

# TerraIncognita
python train_all.py DCCL_TI_0 --dataset TerraIncognita --deterministic --trial_seed 0 --checkpoint_freq 100 --data_dir ../data
```

## 📊 Results

The training outputs will be saved in `DCCL/train_output/[DATASET]/[EXPERIMENT_NAME]/` containing:
- Training logs
- Evaluation results

## 🙏 Acknowledgments

This codebase builds heavily upon the excellent [SWAD](https://github.com/khanrc/swad) framework. We gratefully acknowledge their foundational work in domain generalization research.

## 📖 Citation

If you find this work helpful, please kindly cite:

```bibtex
@inproceedings{wei2025connecting,
  title={Connecting domains and contrasting samples: A ladder for domain generalization},
  author={Wei, Tianxin and Chen, Yifan and He, Xinrui and Bao, Wenxuan and He, Jingrui},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1},
  pages={1563--1574},
  year={2025}
}
```
## RISE-guided DCCL on normal DomainNet 3-source -> 1-target splits

The RISE-guided extension is intended to run on the normal DomainNet directory
layout used by DCCL, not on an extra reduced/sub10 image set. The dataset root
should contain the full DomainNet environments under `domain_net/`:

```text
<data_dir>/domain_net/
├── clip/
├── info/
├── paint/
├── quick/
├── real/
└── sketch/
```

When `--source_envs` and `--target_env` are provided, `train_all.py` filters the
normal DomainNet environments to exactly those three source domains plus the one
target domain. No additional training images are generated. CLIP is used only as
a frozen training-time teacher when `--use_rise` is enabled; evaluation still
uses only the DCCL student.

Domain ids are ordered as follows:

| id | domain |
|---:|--------|
| 0 | clip |
| 1 | info |
| 2 | paint |
| 3 | quick |
| 4 | real |
| 5 | sketch |

### Baseline DCCL, normal 3-source -> 1-target

```bash
cd DCCL/DCCL
python train_all.py domainnet_024_to_5_dccl \
  --dataset DomainNet \
  --data_dir /path/to/data \
  --algorithm DCCL \
  --source_envs 0 2 4 \
  --target_env 5
```

<<<<<<< ours
=======
### DCCL + RISE text prototype alignment (AD), normal 3-source -> 1-target

This is the closest first-stage setting to `DCCL + AD` in the RISE-style
progression: it enables text prototype alignment only, without KD.

```bash
cd DCCL/DCCL
python train_all.py domainnet_024_to_5_rise_proto \
  --dataset DomainNet \
  --data_dir /path/to/data \
  --algorithm DCCL \
  --source_envs 0 2 4 \
  --target_env 5 \
  --use_rise \
  --use_rise_proto \
  --rise_clip_model_name ViT-B/32 \
  --rise_proto_weight 0.1 \
  --rise_prompt_mode multi \
  --rise_projection_dim 512
```

>>>>>>> theirs
### DCCL + RISE-KD, normal 3-source -> 1-target

```bash
cd DCCL/DCCL
python train_all.py domainnet_024_to_5_rise_kd \
  --dataset DomainNet \
  --data_dir /path/to/data \
  --algorithm DCCL \
  --source_envs 0 2 4 \
  --target_env 5 \
  --use_rise \
  --use_rise_kd \
  --rise_clip_model_name ViT-B/32 \
  --rise_kd_weight 0.5 \
  --rise_kd_temperature 2.0 \
  --rise_prompt_mode multi
```

### DCCL + RISE-KD + text prototype alignment, normal 3-source -> 1-target

```bash
cd DCCL/DCCL
python train_all.py domainnet_024_to_5_rise_kd_proto \
  --dataset DomainNet \
  --data_dir /path/to/data \
  --algorithm DCCL \
  --source_envs 0 2 4 \
  --target_env 5 \
  --use_rise \
  --use_rise_kd \
  --use_rise_proto \
  --rise_clip_model_name ViT-B/32 \
  --rise_kd_weight 0.5 \
  --rise_proto_weight 0.1 \
  --rise_kd_temperature 2.0 \
  --rise_prompt_mode multi \
  --rise_projection_dim 512
```

### Sweep normal DomainNet 3-source -> 1-target combinations

To run all normal DomainNet 3-source -> 1-target combinations with the built-in
sweep path, omit `--source_envs` and `--target_env`:

```bash
cd DCCL/DCCL
python train_all.py domainnet_all_3source_rise_kd_proto \
  --dataset DomainNet \
  --data_dir /path/to/data \
  --algorithm DCCL \
  --use_rise \
  --use_rise_kd \
  --use_rise_proto
```

### Scripted normal DomainNet RISE runs

Use `DCCL/DCCL/scripts/run_domainnet_rise_3source.py` when you want an actual
launcher for the normal full DomainNet 3-source -> 1-target protocol instead of
manually editing data paths. The script validates that `<data_dir>/domain_net`
contains all six normal DomainNet environment folders, enforces exactly three
<<<<<<< ours
source domains for fixed-combo runs, and emits/runs baseline, RISE-KD, and
RISE-KD+Proto jobs.
=======
source domains for fixed-combo runs, and emits/runs baseline, RISE-Proto/AD,
RISE-KD, and RISE-KD+Proto jobs.
>>>>>>> theirs

```bash
cd DCCL/DCCL
python scripts/run_domainnet_rise_3source.py \
  --data_dir /path/to/data \
  --source_envs 0 2 4 \
  --target_env 5 \
  --variant all \
  --gpu 0
```

Add `--dry_run` to print the generated normal DomainNet training commands
without launching training.

<<<<<<< ours
=======
For the staged RISE-style order shown in the discussion, start with
`--variant rise_proto` (`DCCL + AD`, implemented as CLIP text prototype
alignment). `--variant rise_kd` is KD-only and is mainly useful as an ablation;
`--variant rise_kd_proto` combines AD/prototype alignment with KD. RD is not
implemented in this first-stage code.

### RISE 80-template prompt ensemble

Use `--rise_prompt_mode rise80` to build CLIP text prototypes with the CLIP
recommended 80 ImageNet-style prompt templates used by RISE. Existing prompt
modes (`simple`, `multi`, and `domain_invariant`) and the default `multi` mode
remain unchanged. The prototype construction logic is the same: encode all
prompts for a class with the frozen CLIP text encoder, L2-normalize each prompt
embedding, average the embeddings, and L2-normalize the class prototype.

```bash
python scripts/run_domainnet_rise_3source.py \
  --data_dir /path/to/data \
  --source_envs 0 2 4 \
  --target_env 5 \
  --variant rise_proto \
  --rise_prompt_mode rise80 \
  --rise_clip_model_name ViT-B/32 \
  --rise_clip_download_root /path/to/clip_cache
```

>>>>>>> theirs
### Offline CLIP weights

The repository's `CLIP/` directory contains the OpenAI CLIP Python package, but
it does not contain the ViT checkpoint weights. On servers without internet,
prepare the CLIP `.pt` file before enabling RISE. You can either place the
weight file in a local CLIP cache directory and pass that directory:

```bash
python scripts/run_domainnet_rise_3source.py \
  --data_dir /path/to/data \
  --source_envs 0 2 4 \
  --target_env 5 \
  --variant rise_kd \
  --rise_clip_model_name ViT-B/32 \
  --rise_clip_download_root /path/to/clip_cache
```

For `ViT-B/32`, the cache directory should contain `ViT-B-32.pt`. Alternatively,
pass the checkpoint file directly as the model argument:

```bash
python scripts/run_domainnet_rise_3source.py \
  --data_dir /path/to/data \
  --source_envs 0 2 4 \
  --target_env 5 \
  --variant rise_kd \
  --rise_clip_model_name /path/to/ViT-B-32.pt
```
