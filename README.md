# 🚦 ICL-Traffic: In-Context Learning for Spatio-Temporal Forecasting

ICL-Traffic is a lightweight framework that enables **in-context learning (ICL)** for pretrained **spatio-temporal foundation models (ST-FMs)** without any fine-tuning or gradient updates.

Pretrained models like OpenCity perform well in zero-shot settings but degrade under **distribution shifts** (e.g., new cities or modalities).  
ICL-Traffic addresses this by **retrieving similar examples and correcting predictions in output space**, keeping the backbone completely frozen.

---

## 💡 Method Overview

### 🔷 Pipeline (Block Diagram) 

<img width="648" height="130" alt="image" src="https://github.com/user-attachments/assets/54b33a17-871d-4416-a917-8e49a48e1ea8" />

---

## ⚙️ Key Components

- **Frozen Backbone** → No parameter updates  
- **KNN Retrieval** → Finds similar traffic patterns  
- **Residual Correction** → Captures systematic model error  
- **Cross-Attention Aggregator** → Learns how to combine corrections  

---

## 📊 Final Results

### 🟢 In-Distribution (PEMS07M)

<img width="766" height="212" alt="image" src="https://github.com/user-attachments/assets/2758cb4d-a1eb-4f01-8a64-c9a6c92a74dc" />


---

### 🔴 Cross-Dataset (Out-of-Distribution)

<img width="766" height="230" alt="image" src="https://github.com/user-attachments/assets/42e2ee19-9378-4105-8043-4709412ea990" />

---

## 🧠 Key Insight

> Retrieval finds relevant examples, and aggregation selects useful corrections — enabling strong adaptation without modifying the model.

---

## 🚀 Highlights

- ✅ No fine-tuning or gradients at test time  
- ✅ Fully frozen backbone  
- ✅ Only ~0.8% extra parameters  
- ✅ Recovers **~78–84% of fine-tuning gains**  

---

## 📄 References

- See [Final Project Report](CSC722_Final%20Project%20Report_Group2.pdf) and implementation details in the project repository.


<span id='Environment'/>

### 1.Environment <a href='#all_catelogue'>[Back to Top]</a>

We use [uv](https://docs.astral.sh/uv/) to manage the Python environment and dependencies. All dependencies (including PyTorch with CUDA 12.4) are declared in `pyproject.toml`.

#### Quick Start

```shell
# Step 1: Clone the repo and enter the OpenCity project directory
git clone -b xzhou38 https://github.com/Parth291102/Traffic-Forecasting-TSFM.git
cd Traffic-Forecasting-TSFM/OpenCity
# ⚠️  All subsequent commands in this guide must be run from this directory.
#     Verify before proceeding:
pwd   # should end with /OpenCity

# Step 2: Install uv into the project's own bin/ directory
#   UV_INSTALL_DIR pins uv to the repo root so it is self-contained
#   and independent from any system-wide or user-wide uv installation.
curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$(pwd)/bin" sh

# Step 3: Add the local bin/ to PATH for the current shell session
export PATH="$(pwd)/bin:$PATH"

# To persist across sessions you can add the absolute path to your shell profile:
# echo 'export PATH="/absolute/path/to/OpenCity/bin:$PATH"' >> ~/.bashrc   # bash
# echo 'export PATH="/absolute/path/to/OpenCity/bin:$PATH"' >> ~/.zshrc    # zsh

# Step 4: Install all dependencies (Python 3.9, PyTorch 2.4.1+cu124, etc.)
uv sync

# Step 5: Download datasets and CA raw files from Hugging Face
#   Model weights are already in the repo (model_weights/OpenCity/).
#   Full list: https://huggingface.co/datasets/hkuds/OpenCity-dataset/tree/main
cd data
HF_BASE="https://huggingface.co/datasets/hkuds/OpenCity-dataset/resolve/main"
for f in \
  PEMS04.zip PEMS07M.zip PEMS08.zip PEMS_BAY.zip METR_LA.zip \
  NYC_TAXI.zip CHI_TAXI.zip NYC_BIKE-3.zip \
  CD_DIDI.zip SZ_DIDI.zip \
  TrafficCD.zip TrafficHZ.zip TrafficJN.zip TrafficNJ.zip TrafficSH.zip TrafficTJ.zip TrafficZZ.zip \
  ca_his_raw_2020.h5.zip ca_meta.zip ca_rn_adj.npy.zip; do
  echo "=== ${f} ==="
  curl -L -o "${f}" "${HF_BASE}/${f}"
  unzip -q "${f}" && rm "${f}"
done

# Step 6: Generate California highway datasets (CAD3, CAD4-*, CAD5, CAD7-*, CAD8-*, CAD12-*)
#   generate_ca_data.py uses relative paths, so it must run from inside data/
uv run python generate_ca_data.py
cd ..
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

### 2. Training OpenCity <a href='#all_catelogue'>[Back to Top]</a>

<span id='Preparing Pre-trained Data'/>

#### 2.1. Preparing Pre-trained Data <a href='#all_catelogue'>[Back to Top]</a>

All datasets are hosted on [Hugging Face](https://huggingface.co/datasets/hkuds/OpenCity-dataset/tree/main). Follow **Steps 5–6** in the [Quick Start](#Environment) above to download all files and generate the California highway subsets.

<span id='Pre-training'/>

#### 2.2. Pre-training <a href='#all_catelogue'>[Back to Top]</a>

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

### 3. Evaluating <a href='#all_catelogue'>[Back to Top]</a>

* **Model Weights**: The pretrained model weights (`OpenCity-plus.pth`, `OpenCity-base.pth`, `OpenCity-mini.pth`) are stored directly in the repository under `model_weights/OpenCity/`. No additional download is needed after cloning.
  * If the weights are missing or you need to re-download them, use these Hugging Face links: [OpenCity-Plus](https://huggingface.co/hkuds/OpenCity-Plus/tree/main), [OpenCity-Base](https://huggingface.co/hkuds/OpenCity-Base/tree/main), [OpenCity-Mini](https://huggingface.co/hkuds/OpenCity-Mini/tree/main)

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

#### 3.1 Zero-shot Evaluation (`test` mode)

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

#### 3.2 Fast Adaptation / Efficient Fine-tuning (`eval` mode)

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

#### 3.3 Supervised Training from Scratch (`ori` mode)

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

#### 3.4 In-Context Traffic Forecasting (`ict` mode)

Run zero-shot inference using **In-Context Traffic** (ICT): a pretrained model is loaded with all parameters frozen, and a set of demonstration (prefix) traffic sequences is prepended to the query to guide prediction — no gradient updates occur.

ICT supports two modes controlled by `-ict_mode`:
- **`residual`** (default): naive average of demo corrections — no training, no weight updates.
- **`learned`**: uses a trained `DemoAggregator` module to weight demo corrections via cross-attention. Requires running `ict_train_aggregator` first to train the aggregator.

ICT parameters (CLI flags in `lib/Params_pretrain.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-num_demonstrations` | `1` | Number of demonstration pairs (K) prepended to each query |
| `-num_prefix_selections` | `1` | Number of prefix candidates sampled per query |
| `-demo_selection` | `random` | Strategy for selecting demonstrations: `random`, `recent`, or `similar` (KNN-based) |
| `-ict_mode` | `residual` | ICT correction mode: `residual` (naive average) or `learned` (trained aggregator) |
| `-aggregator_type` | `attention` | Aggregator architecture: `simple` (cosine similarity, ~8K params) or `attention` (cross-attention, ~30K-200K params) |

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

> **What `ict` mode does**: Loads pretrained weights → freezes all parameters → builds ICT dataloaders with demonstration prefixes → runs inference via `test_ict()` (no training, no weight updates). When `-ict_mode learned`, it additionally loads a pre-trained `DemoAggregator` from `aggregator_best.pth` to weight demo corrections.

#### 3.5 Learned Demo Aggregation (`ict_train_aggregator` mode)

Train a lightweight `DemoAggregator` module that learns to weight demo corrections based on query-demo feature similarity.  The base model stays **fully frozen** — only the aggregator parameters (~30K-200K) are updated.

Inspired by [In-Context Fine-Tuning for Time-Series Foundation Models (Das et al., 2024)](https://arxiv.org/abs/2410.24087): the model should **learn how to use demos** rather than naively averaging corrections.

Aggregator training parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-aggregator_type` | `attention` | `simple` (cosine similarity, ~8K params) or `attention` (cross-attention + gating, ~30K-200K params) |
| `-aggregator_epochs` | `5` | Number of training epochs for the aggregator |
| `-aggregator_lr` | `1e-4` | Learning rate for aggregator training |
| `-debug_batches` | `0` | Limit batches per phase for smoke testing (0=unlimited) |

```bash
# First, edit conf/general_conf/pretrain.conf:
#   dataset_use = ['PEMS07M']   # single target dataset
#   val_ratio = 0.1
#   test_ratio = 0.4

# ⚠️  Run from OpenCity/model/ (not the project root):
cd /path/to/OpenCity/model

# Step 1: Train the aggregator (base model frozen, CPU-friendly)
#   Training uses online forward passes (no disk caching).
#   Checkpoint is saved after every batch to aggregator_ckpt.pth.
#   If interrupted, re-run the same command to resume from last checkpoint.
uv run python Run.py -mode ict_train_aggregator -model OpenCity \
  -load_pretrain_path OpenCity-plus.pth -batch_size 64 \
  -num_demonstrations 3 -demo_selection similar \
  -aggregator_type attention -aggregator_epochs 3 -aggregator_lr 1e-4 \
  -early_stop True -early_stop_patience 3 \
  -use_cpu True \
  -log_step 1 \
  --embed_dim 512 --skip_dim 512 --enc_depth 6

# Smoke test (1 batch only, for pipeline validation):
uv run python Run.py -mode ict_train_aggregator -model OpenCity \
  -load_pretrain_path OpenCity-plus.pth -batch_size 4 \
  -num_demonstrations 3 -demo_selection similar \
  -aggregator_type attention -aggregator_epochs 2 -aggregator_lr 1e-4 \
  -debug_batches 1 -use_cpu True -log_step 1 \
  --embed_dim 512 --skip_dim 512 --enc_depth 6

# Step 2: Evaluate with learned aggregation
uv run python Run.py -mode ict -model OpenCity \
  -load_pretrain_path OpenCity-plus.pth -batch_size 2 \
  -num_demonstrations 3 -demo_selection similar \
  -ict_mode learned -aggregator_type attention \
  --embed_dim 512 --skip_dim 512 --enc_depth 6

# Compare against baseline (naive averaging)
uv run python Run.py -mode ict -model OpenCity \
  -load_pretrain_path OpenCity-plus.pth -batch_size 2 \
  -num_demonstrations 3 -demo_selection similar \
  -ict_mode residual \
  -use_cpu True \
  --embed_dim 512 --skip_dim 512 --enc_depth 6
```

> **What `ict_train_aggregator` does**: Loads pretrained weights → freezes all base model parameters → initializes a `DemoAggregator` → trains aggregator online (each batch: base model forward passes + aggregator gradient step) → saves checkpoint after every batch (`aggregator_ckpt.pth`) for resume support → runs validation at end of each epoch with early stopping → saves `aggregator_best.pth` → runs `test_ict` with `ict_mode='learned'`.
>
> **Resume support**: If training is interrupted (e.g., server shutdown), re-run the exact same command. The trainer auto-detects `aggregator_ckpt.pth` and resumes from the last completed batch, preserving optimizer state, epoch position, and best validation loss.
>
> **Skipping validation**: Pass `-early_stop False` to skip validation cache pre-computation and run all training epochs without early stopping. This is useful for large datasets (e.g., SZ_DIDI with 627 nodes) where Phase 1 pre-computation is expensive and time-constrained — skipping the val cache saves ~1.5 hours on DIDI-scale datasets. The aggregator weights from the last epoch are saved directly as `aggregator_best.pth`.

The `DemoAggregator` module is defined in `model/OpenCity/DemoAggregator.py` and provides two variants:
- **`SimpleDemoAggregator`**: cosine similarity between projected query and demo encoder features → per-node softmax weights → weighted correction sum. ~8K params.
- **`DemoAggregator`**: multi-head cross-attention (Q·K over demo features, H=4 heads) → per-head weighted corrections → head combination → per-node softplus scale. ~198K params (D=512).

#### Summary of Modes

| Mode | Description | Parameters Updated | Typical Use Case |
|------|-------------|-------------------|-----------------|
| `pretrain` | Multi-dataset joint pretraining | All | Building the foundation model |
| `test` | Pure inference, no training | None | Zero-shot & supervised evaluation |
| `eval` | Efficient fine-tuning | Prediction head only (`linear` layer) | Fast adaptation to unseen data |
| `ori` | Full supervised training | All | Baseline comparisons |
| `ict` | In-context inference with demonstration prefixes | None | Zero-shot ICT evaluation (residual or learned) |
| `ict_train_aggregator` | Train learned demo aggregator | DemoAggregator only (~30K-200K params) | Train aggregator for learned ICT mode |

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
