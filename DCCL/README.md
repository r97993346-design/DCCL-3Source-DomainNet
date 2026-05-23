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

## 🧠 CSR Prompt Embeddings (Offline)

For CSR-DCCL phase-1, causal/spurious text embeddings are built **offline** from prompt JSON files.
No online LLM/VLM API is used in this workflow.

### 1) Prepare prompt JSON

Use a file like `DCCL/assets/prompts/example_causal_spurious_prompts.json`:
- `class_names`: class order used for exported embeddings.
- `prompts[class_name].causal_prompts`: non-empty list.
- `prompts[class_name].spurious_prompts`: non-empty list.

> Keep the prompt JSON under version control for reproducibility.

### 2) Build embeddings with CLIP text encoder

```bash
cd DCCL/
PYTHONPATH=../CLIP python DCCL/tools/build_causal_spurious_embeddings.py \
  --prompt_json DCCL/assets/prompts/example_causal_spurious_prompts.json \
  --output_dir DCCL/assets/semantic_embeddings/example \
  --clip_model ViT-B-32 \
  --pretrained openai \
  --device cuda
```

The script saves:
- `causal_embeddings.pt`
- `spurious_embeddings.pt`

Each file contains:
- `embeddings`: `torch.Tensor[num_classes, embedding_dim]`
- `class_names`: `List[str]` (same order as JSON `class_names`)
- `encoder_name`: `str`
- `prompt_type`: `"causal"` or `"spurious"`
- `source_json`: `str`

### 3) Train with generated embeddings

```bash
cd DCCL/
python train_all.py exp_csr --dataset DomainNet --algorithm DCCL --data_dir ../data \
  --use_causal_reliability \
  --causal_embedding_path DCCL/assets/semantic_embeddings/example/causal_embeddings.pt \
  --spurious_embedding_path DCCL/assets/semantic_embeddings/example/spurious_embeddings.pt
```

### 4) Prepare official DomainNet prompt draft (345 classes)

First prepare class order file at `DCCL/assets/prompts/domainnet_class_order.json`:
```json
{"class_names": ["class_0", "...", "class_344"]}
```

Then build the draft prompt template (causal empty, shared spurious library) and per-25-class batches:

```bash
python DCCL/tools/prepare_domainnet_prompt_template.py \
  --class_order_json DCCL/assets/prompts/domainnet_class_order.json \
  --output_json DCCL/assets/prompts/domainnet_causal_spurious_prompts_draft.json
```

Validate prompt quality/consistency:

```bash
python DCCL/tools/validate_domainnet_prompts.py \
  --class_order_json DCCL/assets/prompts/domainnet_class_order.json \
  --prompt_json DCCL/assets/prompts/domainnet_causal_spurious_prompts_draft.json
```

Use strict mode after manually filling causal prompts:

```bash
python DCCL/tools/validate_domainnet_prompts.py \
  --class_order_json DCCL/assets/prompts/domainnet_class_order.json \
  --prompt_json DCCL/assets/prompts/domainnet_causal_spurious_prompts_draft.json \
  --strict_causal
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
