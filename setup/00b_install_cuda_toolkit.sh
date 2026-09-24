#!/usr/bin/env bash
# Install CUDA 11.8 toolkit (nvcc) — needed to compile the CUDA extensions
# of detectron2 / CropFormer / chamferdist.
# Needs root. Run inside WSL/Linux:
#   wsl -u root -- bash /home/<user>/OV-Octree-Graph/setup/00b_install_cuda_toolkit.sh
set -e

if [ "$(whoami)" != "root" ]; then
    echo "please run this script as root: wsl -u root -- bash $0"
    exit 1
fi

echo "[install] cuda-toolkit-11-8"
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends cuda-toolkit-11-8

export PATH=/usr/local/cuda-11.8/bin:$PATH
nvcc --version | tail -1
echo "CUDA_TOOLKIT_DONE"
