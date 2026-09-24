#!/usr/bin/env bash
# Install Miniconda, create the `ovgraph` env (python 3.10), install
# PyTorch 2.1.2 (cu118) and the CUDA 11.8 toolkit (nvcc, for compiling
# detectron2 / CropFormer CUDA extensions).
# Run inside WSL/Linux:
#   bash setup/00_install_conda_torch.sh
set -e

cd ~

# 1. Miniconda (TUNA mirror)
if [ ! -f ~/miniconda3/bin/conda ]; then
    echo "[download] Miniconda"
    wget -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p ~/miniconda3
fi
source ~/miniconda3/etc/profile.d/conda.sh
~/miniconda3/bin/conda init bash >/dev/null 2>&1 || true

# 2. conda channels -> TUNA mirror (also sidesteps the Anaconda ToS check)
~/miniconda3/bin/conda config --set show_channel_urls yes
~/miniconda3/bin/conda config --set default_channels \
    '["https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main", "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r"]' 2>/dev/null || true
~/miniconda3/bin/conda config --set custom_channels.conda-forge \
    https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud 2>/dev/null || true
~/miniconda3/bin/conda tos accept --override-channels \
    --channel https://repo.anaconda.com/pkgs/main \
    --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# 3. ovgraph env
~/miniconda3/bin/conda create -y -q -n ovgraph python=3.10
conda activate ovgraph

# 3. PyTorch cu118 (official index; the aliyun pytorch-wheels mirror is dead)
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -q torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

echo "CONDA_TORCH_DONE"
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
echo "(CUDA toolkit / nvcc is installed separately by 00b_install_cuda_toolkit.sh, needs root)"
