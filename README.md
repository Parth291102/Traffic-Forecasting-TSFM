
<span id='Environment'/>

### 2.Environment <a href='#all_catelogue'>[Back to Top]</a>

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

### 3. Training OpenCity <a href='#all_catelogue'>[Back to Top]</a>

<span id='Preparing Pre-trained Data'/>

#### 3.1. Preparing Pre-trained Data <a href='#all_catelogue'>[Back to Top]</a>

All datasets are hosted on [Hugging Face](https://huggingface.co/datasets/hkuds/OpenCity-dataset/tree/main). Follow **Steps 5–6** in the [Quick Start](#Environment) above to download all files 
