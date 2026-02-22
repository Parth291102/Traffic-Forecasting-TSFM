# OpenCity: Open Spatio-Temporal Foundation Models for Traffic Prediction

<img src='opencity.png' />

A pytorch implementation for the paper: [OpenCity: Open Spatio-Temporal Foundation Models for Traffic Prediction](https://arxiv.org/abs/2408.10269)<br />  

[Zhonghang Li](https://scholar.google.com/citations?user=__9uvQkAAAAJ), [Long Xia](https://scholar.google.com/citations?user=NRwerBAAAAAJ), [Lei Shi](https://harryshil.github.io/), [Yong Xu](https://scholar.google.com/citations?user=1hx5iwEAAAAJ), [Dawei Yin](https://www.yindawei.com/), [Chao Huang](https://sites.google.com/view/chaoh)* (*Correspondence)<br />  

**[Data Intelligence Lab](https://sites.google.com/view/chaoh/home)@[University of Hong Kong](https://www.hku.hk/)**, [South China University of Technology](https://www.scut.edu.cn/en/), Baidu Inc  
<!--
-----

<a href='https://OpenCity-ST.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://github.com/HKUDS/OpenCity'><img src='https://img.shields.io/badge/Demo-Page-purple'></a> 
<#><img src='https://img.shields.io/badge/Paper-PDF-orange'></a> 
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=4BIbQt-EIAM)
 • 🌐 <a href="https://zhuanlan.zhihu.com/p/684785925" target="_blank">Chinese Blog</a>
-->
This repository hosts the code, data, and model weights of **OpenCity**.

-----
## 🎉 News 
- [x] [2024.08.21] Release the full paper.
- [x] [2024.08.20] Add video.
- [x] [2024.08.15] 🚀🚀 Release the code, model weights and datasets of OpenCity.
- [x] [2024.08.15] Release baselines codes.


🎯🎯📢📢 We upload the **models** and **data** used in our OpenCity on 🤗 **Huggingface**. We highly recommend referring to the table below for further details: 

| 🤗 Huggingface Address                                        | 🎯 Description                                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [https://huggingface.co/hkuds/OpenCity-Plus](https://huggingface.co/hkuds/OpenCity-Plus/tree/main) | It's the model weights of our OpenCity-Plus. |
| [https://huggingface.co/datasets/hkuds/OpenCity-dataset/tree/main](https://huggingface.co/datasets/hkuds/OpenCity-dataset/tree/main) | We released the datasets used in OpenCity. |

## 👉 TODO 
...


-----------

## Introduction

<p style="text-align: justify">
In this work, we aim to unlock new possibilities for building versatile, resilient and adaptive spatio-temporal foundation models for traffic prediction. 
To achieve this goal, we introduce a novel foundation model, named OpenCity, that can effectively capture and normalize the underlying spatio-temporal patterns from diverse data characteristics, facilitating zero-shot generalization across diverse urban environments. 
OpenCity integrates the Transformer architecture with graph neural networks to model the complex spatio-temporal dependencies in traffic data. 
By pre-training OpenCity on large-scale, heterogeneous traffic datasets, we enable the model to learn rich, generalizable representations that can be seamlessly applied to a wide range of traffic forecasting scenarios. 
Experimental results demonstrate that OpenCity exhibits exceptional zero-shot predictive performance in various traffic prediction tasks.
</p>

![The detailed framework of the proposed OpenCity.](https://github.com/OpenCity-ST/OpenCity-ST.github.io/blob/main/images/framework.png)

## Main Results
**Outstanding Zero-shot Prediction Performance.** OpenCity achieves significant zero-shot learning breakthroughs, outperforming most baselines even without fine-tuning. This highlights the approach's robustness and effectiveness at learning complex spatio-temporal patterns in large-scale traffic data, extracting universal insights applicable across downstream tasks.
 
![Zero-shot vs. Full-shot.](https://github.com/OpenCity-ST/OpenCity-ST.github.io/blob/main/images/zero-shot.png)



### Demo Video
https://github.com/user-attachments/assets/39265dc5-0126-483b-951e-518c6cb210e0

-----------
<span id='Usage'/>

## Getting Started

<span id='all_catelogue'/>

### Table of Contents:
* <a href='#Code Structure'>1. Code Structure</a>
* <a href='#Environment'>2. Environment </a>
* <a href='#Training OpenCity'>3. Training OpenCity </a>
  * <a href='#Preparing Pre-trained Data'>3.1. Preparing Pre-trained Data </a>
  * <a href='#Pre-training'>3.2. Pre-training </a>
* <a href='#Evaluating'>4. Evaluating </a>
  * <a href='#Zero-shot'>4.1. Zero-shot Evaluation (`test` mode)</a>
  * <a href='#Fast-Adaptation'>4.2. Fast Adaptation (`eval` mode)</a>
  * <a href='#Supervised'>4.3. Supervised Training from Scratch (`ori` mode)</a>
  * <a href='#ICT'>4.4. In-Context Traffic Forecasting (`ict` mode)</a>
****


<span id='Code Structure'/>

### 1. Code Structure <a href='#all_catelogue'>[Back to Top]</a>

```
├── conf/
│   ├── AGCRN/
│   │   └── AGCRN.conf
│   ├── ASTGCN/
│   │   └── ASTGCN.conf
│   ├── general_conf/
│   │   ├── global_baselines.conf
│   │   └── pretrain.conf
│   ├── GWN/
│   │   └── GWN.conf
│   ├── MSDR/
│   │   └── MSDR.conf
│   ├── MTGNN/
│   │   └── MTGNN.conf
│   ├── OpenCity/
│   │   └── OpenCity.conf
│   ├── PDFormer/
│   │   └── PDFormer.conf
│   ├── STGCN/
│   │   └── STGCN.conf
│   ├── STSGCN/
│   │   └── STSGCN.conf
│   ├── STWA/
│   │   └── STWA.conf
│   └── TGCN/
│       └── TGCN.conf
├── data/
│   ├── generate_ca_data.py
│   └── README.md
├── lib/
│   ├── data_process.py
│   ├── logger.py
│   ├── metrics.py
│   ├── Params_predictor.py
│   ├── Params_pretrain.py
│   ├── predifineGraph.py
│   └── TrainInits.py
├── model/
│   ├── AGCRN/
│   │   ├── AGCN.py
│   │   ├── AGCRN.py
│   │   ├── AGCRNCell.py
│   │   └── args.py
│   ├── ASTGCN/
│   │   ├── args.py
│   │   └── ASTGCN.py
│   ├── GWN/
│   │   ├── args.py
│   │   └── GWN.py
│   ├── MSDR/
│   │   ├── args.py
│   │   ├── gmsdr_cell.py
│   │   └── gmsdr_model.py
│   ├── MTGNN/
│   │   ├── args.py
│   │   └── MTGNN.py
│   ├── OpenCity/
│   │   ├── args.py
│   │   └── OpenCity.py
│   ├── PDFormer/
│   │   ├── args.py
│   │   └── PDFormer.py
│   ├── ST_WA/
│   │   ├── args.py
│   │   ├── attention.py
│   │   └── ST_WA.py
│   ├── STGCN/
│   │   ├── args.py
│   │   └── stgcn.py
│   ├── STSGCN/
│   │   ├── args.py
│   │   └── STSGCN.py
│   └── TGCN/
│       ├── args.py
│       └── TGCN.py
│   ├── Model.py
│   ├── BasicTrainer.py
│   ├── Run.py
└── model_weights/
    ├── OpenCity/
    └── README.md
```


<span id='Environment'/>

### 2.Environment <a href='#all_catelogue'>[Back to Top]</a>

We use [uv](https://docs.astral.sh/uv/) to manage the Python environment and dependencies. All dependencies (including PyTorch with CUDA 12.4) are declared in `pyproject.toml`.

#### Quick Start

```shell
# Step 1: Clone the repo and enter the project directory
git clone https://github.com/HKUDS/OpenCity.git
cd OpenCity
# ⚠️  All subsequent commands in this guide must be run from this directory.
#     Verify before proceeding:
pwd   # should end with /OpenCity

# Step 2: Pull LFS-tracked files (model weights, adjacency matrices, etc.)
#   The repo uses Git LFS for large binary files (.npy, .pth, .npz).
#   Without this step those files contain only LFS pointer text, not real data.
git lfs pull

# Step 3: Install uv into the project's own bin/ directory
#   UV_INSTALL_DIR pins uv to the repo root so it is self-contained
#   and independent from any system-wide or user-wide uv installation.
curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$(pwd)/bin" sh

# Step 4: Add the local bin/ to PATH for the current shell session
export PATH="$(pwd)/bin:$PATH"

# To persist across sessions you can add the absolute path to your shell profile:
# echo 'export PATH="/absolute/path/to/OpenCity/bin:$PATH"' >> ~/.bashrc   # bash
# echo 'export PATH="/absolute/path/to/OpenCity/bin:$PATH"' >> ~/.zshrc    # zsh

# Step 5: Install all dependencies (Python 3.9, PyTorch 2.4.1+cu124, etc.)
uv sync
```

`uv sync` will automatically:
1. Create a virtual environment (`.venv/`) with Python 3.9
2. Install PyTorch 2.4.1 + CUDA 12.4 (from the PyTorch wheel index)
3. Install all other dependencies (numpy, scipy, pandas, tqdm, fastdtw, tslearn, h5py, etc.)

#### Re-installing / Updating Dependencies

If you encounter environment issues or need a clean reinstall:

```shell
# Force-reinstall all packages from scratch
uv sync --reinstall

# Or remove the venv entirely and recreate it
rm -rf .venv && uv sync
```

#### Running Commands

> **Important**: `Run.py` uses relative paths (`../conf/`) to locate config files, so it **must be run from inside the `OpenCity/model/` directory**.

```shell
cd /path/to/OpenCity/model   # ⚠️  required working directory for all run commands
```

Use `uv run` to execute scripts within the managed environment (no manual activation needed):

```shell
# Zero-shot evaluation example (OpenCity-plus on PEMS07M)
cd /path/to/OpenCity/model
uv run python Run.py -mode test -model OpenCity \
  -load_pretrain_path OpenCity-plus.pth -batch_size 2 \
  --embed_dim 512 --skip_dim 512 --enc_depth 6
```

Or activate the virtual environment first and then run `python` directly:

```shell
cd /path/to/OpenCity/model
source ../.venv/bin/activate

# Zero-shot evaluation example (OpenCity-plus on PEMS07M)
python Run.py -mode test -model OpenCity \
  -load_pretrain_path OpenCity-plus.pth -batch_size 2 \
  --embed_dim 512 --skip_dim 512 --enc_depth 6
```

#### Adding New Dependencies

```shell
uv add <package-name>
```

<details>
<summary><b>Alternative: conda/pip setup (legacy)</b></summary>

```shell
conda create -n opencity python=3.9.13
conda activate opencity
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124 -f https://download.pytorch.org/whl/cu124/torch_stable.html
pip install -r requirements.txt
```
</details>

<span id='Training OpenCity'/>

### 3. Training OpenCity <a href='#all_catelogue'>[Back to Top]</a>

<span id='Preparing Pre-trained Data'/>

#### 3.1. Preparing Pre-trained Data <a href='#all_catelogue'>[Back to Top]</a>

* The model's generalization capabilities and predictive performance were extensively evaluated using a diverse set of large-scale, real-world public datasets covering various traffic-related data categories, including **Traffic Flow**, **Taxi Demand**, **Bicycle Trajectories**, **Traffic Speed Statistics**, and **Traffic Index Statistics**, from regions across the United States and China, such as New York City, Chicago, Los Angeles, the Bay Area, Shanghai, Shenzhen, and Chengdu. <br />
* These data are organized in [OpenCity-dataset](https://huggingface.co/datasets/hkuds/OpenCity-dataset/tree/main). Please download it and put it at ./data. Subsequently, unzip all files and run [generate_ca_data.py](https://github.com/HKUDS/OpenCity/blob/main/data/generate_ca_data.py).

<span id='Pre-training'/>

#### 3.2. Pre-training <a href='#all_catelogue'>[Back to Top]</a>

* To pretrain the OpenCity model with different configurations, execute `Run.py` from inside the `OpenCity/model/` directory:

```bash
cd /path/to/OpenCity/model

# OpenCity-plus
uv run python Run.py -mode pretrain -model OpenCity \
  -save_pretrain_path OpenCity-plus2.0.pth -batch_size 4 \
  --embed_dim 512 --skip_dim 512 --enc_depth 6

# OpenCity-base
uv run python Run.py -mode pretrain -model OpenCity \
  -save_pretrain_path OpenCity-base2.0.pth -batch_size 8 \
  --embed_dim 256 --skip_dim 256 --enc_depth 3

# OpenCity-mini
uv run python Run.py -mode pretrain -model OpenCity \
  -save_pretrain_path OpenCity-mini2.0.pth -batch_size 16 \
  --embed_dim 128 --skip_dim 128 --enc_depth 3
```

* Parameter setting instructions. The parameter settings consist of two parts: the pretrain config and other configs. To avoid any confusion arising from potential overlapping parameter names, we employ a hyphen (-) to specify the parameters of pretrain config and use a double hyphen (--) to specify the parameters of other configs. Please note that if two parameters have the same name, **the settings of the latter can override those of the former.**

<span id='Evaluating'/>

### 4. Evaluating <a href='#all_catelogue'>[Back to Top]</a>

* **Preparing Checkpoints of OpenCity**. You can download our model using the following link: [OpenCity-Plus](https://huggingface.co/hkuds/OpenCity-Plus/tree/main), [OpenCity-Base](https://huggingface.co/hkuds/OpenCity-Base/tree/main), [OpenCity-Mini](https://huggingface.co/hkuds/OpenCity-Mini/tree/main)

#### ⚠️ Important: Configure `pretrain.conf` Before Evaluation

Before running `test`, `eval`, or `ori` mode, you **must manually edit** `conf/general_conf/pretrain.conf` to set the correct `dataset_use`, `val_ratio`, and `test_ratio` for your target dataset. Different datasets require different split ratios to match the paper's experimental setup.

**Step 1**: Set `dataset_use` to a **single dataset** (evaluation should run one dataset at a time):
```ini
dataset_use = ['PEMS07M']
```

**Step 2**: Set `val_ratio` and `test_ratio` according to the dataset category:

| Category | Datasets | val_ratio | test_ratio | Train/Val/Test |
|----------|----------|-----------|------------|----------------|
| **Zero-shot** | CAD3, CAD5, PEMS07M, TrafficSH | 0.1 | 0.4 | 50%/10%/40% |
| **Zero-shot** | CHI_TAXI, NYC_BIKE-3 | 0.2 | 0.6 | 20%/20%/60% |
| **Fast Adaptation** | CD_DIDI, SZ_DIDI | 0.1 | 0.4 | 50%/10%/40% |
| **Supervised (in pretrain)** | PEMS_BAY | 0.1 | 0.4 | 50%/10%/40% |
| **Supervised (in pretrain)** | CAD8-1, CAD8-2, CAD12-2 | 0.1 | 0.1 | 80%/10%/10% |
| **Supervised (in pretrain)** | PEMS04, PEMS08, METR_LA, CAD4-*, CAD7-*, CAD12-1, TrafficHZ, TrafficZZ, TrafficCD, TrafficJN | 0.1 | 0.4 | 50%/10%/40% |

> **Note**: NYC_TAXI uses a custom date-based split (2016–2020 train, Jan–Feb 2021 val, Mar–Dec 2021 test) which is handled internally in `data_process.py`. Set `val_ratio = 0.028` and `test_ratio = 0.139` as approximations, or use the default code logic.

#### Available Datasets

| Dataset | Nodes | Interval | Data Category |
|---------|-------|----------|---------------|
| PEMS04 | 307 | 5 min | Traffic Flow |
| PEMS08 | 170 | 5 min | Traffic Flow |
| PEMS07M | 228 | 5 min | Traffic Flow |
| PEMS_BAY | 325 | 5 min | Traffic Speed |
| METR_LA | 207 | 5 min | Traffic Speed |
| CAD3 | 480 | 5 min | CA Highway Flow |
| CAD4-1/2/3/4 | 621/610/593/528 | 5 min | CA Highway Flow |
| CAD5 | 211 | 5 min | CA Highway Flow |
| CAD7-1/2/3 | 666/634/559 | 5 min | CA Highway Flow |
| CAD8-1/2 | 510/512 | 5 min | CA Highway Flow |
| CAD12-1/2 | 453/500 | 5 min | CA Highway Flow |
| NYC_TAXI | 263 | 30 min | Taxi Demand |
| CHI_TAXI | 77 | 30 min | Taxi Demand |
| NYC_BIKE-3 | 540 | 30 min | Bicycle Trajectories |
| CD_DIDI | 524 | 10 min | Ride-hailing Demand |
| SZ_DIDI | 627 | 10 min | Ride-hailing Demand |
| TrafficHZ/ZZ/CD/JN | 672/676/728/576 | 30 min | Traffic Index |
| TrafficSH | 896 | 30 min | Traffic Index |

#### 4.1 Zero-shot Evaluation (`test` mode)

Directly evaluate a pretrained model on a dataset **without any training**. This is used for both in-pretrain datasets (supervised evaluation) and out-of-pretrain datasets (zero-shot evaluation).

```bash
# First, edit conf/general_conf/pretrain.conf:
#   dataset_use = ['PEMS07M']
#   val_ratio = 0.1
#   test_ratio = 0.4

# ⚠️  Run from OpenCity/model/ (not the project root):
cd /path/to/OpenCity/model

# Use OpenCity-plus to evaluate
uv run python Run.py -mode test -model OpenCity \
  -load_pretrain_path OpenCity-plus.pth -batch_size 2 \
  --embed_dim 512 --skip_dim 512 --enc_depth 6

# Use OpenCity-base to evaluate
uv run python Run.py -mode test -model OpenCity \
  -load_pretrain_path OpenCity-base.pth -batch_size 2 \
  --embed_dim 256 --skip_dim 256 --enc_depth 3

# Use OpenCity-mini to evaluate
uv run python Run.py -mode test -model OpenCity \
  -load_pretrain_path OpenCity-mini.pth -batch_size 2 \
  --embed_dim 128 --skip_dim 128 --enc_depth 3
```

#### 4.2 Fast Adaptation / Efficient Fine-tuning (`eval` mode)

Load a pretrained model, **freeze all backbone parameters**, and only fine-tune the **prediction head** (the last linear layer) for a few epochs. This is used for fast adaptation to unseen data categories (e.g., CD_DIDI, SZ_DIDI which are traffic index data not seen during pretraining).

```bash
# First, edit conf/general_conf/pretrain.conf:
#   dataset_use = ['CD_DIDI']
#   val_ratio = 0.1
#   test_ratio = 0.4

# ⚠️  Run from OpenCity/model/ (not the project root):
cd /path/to/OpenCity/model

# Fast Adaptation with OpenCity-plus (3 epochs, batch size 64)
uv run python Run.py -mode eval -model OpenCity \
  -load_pretrain_path OpenCity-plus.pth -batch_size 64 -epochs 3 \
  --embed_dim 512 --skip_dim 512 --enc_depth 6

# Fast Adaptation with OpenCity-base
uv run python Run.py -mode eval -model OpenCity \
  -load_pretrain_path OpenCity-base.pth -batch_size 64 -epochs 3 \
  --embed_dim 256 --skip_dim 256 --enc_depth 3
```

> **What `eval` mode does**: Loads pretrained weights → freezes all parameters → unfreezes only `model.predictor.linear` (the prediction head) → trains for the specified number of epochs with early stopping.

#### 4.3 Supervised Training from Scratch (`ori` mode)

Train a model from scratch on a single dataset with full train/val/test split and early stopping. Used for baseline comparisons.

```bash
# First, edit conf/general_conf/pretrain.conf:
#   dataset_use = ['CD_DIDI']
#   val_ratio = 0.1
#   test_ratio = 0.4

# ⚠️  Run from OpenCity/model/ (not the project root):
cd /path/to/OpenCity/model

# Run STGCN baseline (100 epochs, early stop after 15)
uv run python Run.py -mode ori -model STGCN \
  -batch_size 64 -epochs 100 \
  -early_stop True -early_stop_patience 15 --real_value False

# Run OpenCity from scratch (for comparison)
uv run python Run.py -mode ori -model OpenCity \
  -batch_size 8 --embed_dim 256 --skip_dim 256 --enc_depth 3
```

#### 4.4 In-Context Traffic Forecasting (`ict` mode)

Run zero-shot inference using **In-Context Traffic** (ICT): a pretrained model is loaded with all parameters frozen, and a set of demonstration (prefix) traffic sequences is prepended to the query to guide prediction — no gradient updates occur.

ICT parameters (controlled via `conf/ICT/ICT.conf` or CLI flags):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-num_demonstrations` | `1` | Number of demonstration pairs (K) prepended to each query |
| `-num_prefix_selections` | `1` | Number of prefix candidates sampled per query |
| `-demo_selection` | `random` | Strategy for selecting demonstrations (`random`) |

```bash
# First, edit conf/general_conf/pretrain.conf:
#   dataset_use = ['PEMS07M']   # single target dataset
#   val_ratio = 0.1
#   test_ratio = 0.4

# ⚠️  Run from OpenCity/model/ (not the project root):
cd /path/to/OpenCity/model

# ICT inference with OpenCity-plus (1 demonstration, random selection)
uv run python Run.py -mode ict -model OpenCity \
  -load_pretrain_path OpenCity-plus.pth -batch_size 2 \
  -num_demonstrations 1 -num_prefix_selections 1 -demo_selection random \
  --embed_dim 512 --skip_dim 512 --enc_depth 6

# ICT inference with OpenCity-base
uv run python Run.py -mode ict -model OpenCity \
  -load_pretrain_path OpenCity-base.pth -batch_size 2 \
  -num_demonstrations 1 -num_prefix_selections 1 -demo_selection random \
  --embed_dim 256 --skip_dim 256 --enc_depth 3

# ICT inference with OpenCity-mini
uv run python Run.py -mode ict -model OpenCity \
  -load_pretrain_path OpenCity-mini.pth -batch_size 2 \
  -num_demonstrations 1 -num_prefix_selections 1 -demo_selection random \
  --embed_dim 128 --skip_dim 128 --enc_depth 3
```

> **What `ict` mode does**: Loads pretrained weights → freezes all parameters → builds ICT dataloaders with demonstration prefixes → runs inference via `test_ict()` (no training, no weight updates).

#### Summary of Modes

| Mode | Description | Parameters Updated | Typical Use Case |
|------|-------------|-------------------|-----------------|
| `pretrain` | Multi-dataset joint pretraining | All | Building the foundation model |
| `test` | Pure inference, no training | None | Zero-shot & supervised evaluation |
| `eval` | Efficient fine-tuning | Prediction head only (`linear` layer) | Fast adaptation to unseen data |
| `ori` | Full supervised training | All | Baseline comparisons |
| `ict` | In-context inference with demonstration prefixes | None | Zero-shot ICT evaluation |

<!--
## Contact
For any questions or feedback, feel free to contact [Zhonghang Li](mailto:bjdwh.zzh@gmail.com).
-->

## Citation

If you find OpenCity useful in your research or applications, please kindly cite:

```
@misc{li2024opencity,
      title={OpenCity: Open Spatio-Temporal Foundation Models for Traffic Prediction}, 
      author={Zhonghang Li and Long Xia and Lei Shi and Yong Xu and Dawei Yin and Chao Huang},
      year={2024},
      eprint={2408.10269},
      archivePrefix={arXiv}
}
```


<!--
## Acknowledgements
You may refer to related work that serves as foundations for our framework and code repository, 
[Vicuna](https://github.com/lm-sys/FastChat). We also partially draw inspirations from [GraphGPT](https://github.com/HKUDS/GraphGPT). The design of our website and README.md was inspired by [NExT-GPT](https://next-gpt.github.io/), and the design of our system deployment was inspired by [gradio](https://www.gradio.app) and [Baize](https://huggingface.co/spaces/project-baize/chat-with-baize). Thanks for their wonderful works.
-->
