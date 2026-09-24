#!/usr/bin/env bash
# Download pretrained weights into pretrained_weights/.
# - TAP: hf-mirror (no auth needed)
# - CropFormer: the Adobe_EntitySeg dataset is gated on HuggingFace.
#   You must request access at
#     https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg
#   and create a read token at https://huggingface.co/settings/tokens,
#   then re-run: HF_TOKEN=hf_xxx bash setup/04_download_weights.sh
# Run inside WSL/Linux:
#   bash setup/04_download_weights.sh
set -e

cd "$(dirname "$0")/.." || exit 1
mkdir -p pretrained_weights
cd pretrained_weights

# CropFormer (Mask2Former hornet 3x) -- gated on HF, needs a token
if [ ! -f Mask2Former_hornet_3x_576d0b.pth ]; then
    if [ -n "$HF_TOKEN" ]; then
        echo "[download] CropFormer (with HF token)"
        pip install -q -U "huggingface_hub[cli]" 2>/dev/null || true
        HF_ENDPOINT=https://hf-mirror.com HF_TOKEN="$HF_TOKEN" \
            python -c "
import os
from huggingface_hub import hf_hub_download
p = hf_hub_download('qqlu1992/Adobe_EntitySeg',
                    'CropFormer_model/Entity_Segmentation/Mask2Former_hornet_3x/Mask2Former_hornet_3x_576d0b.pth',
                    token=os.environ['HF_TOKEN'])
print(p)
"
        mv Mask2Former_hornet_3x_576d0b.pth ~/.cache/huggingface/hub/**/Mask2Former_hornet_3x_576d0b.pth 2>/dev/null || \
        find ~/.cache/huggingface/hub -name "Mask2Former_hornet_3x_576d0b.pth" -exec cp {} . \;
    else
        echo "[WARN] CropFormer skipped: the dataset qqlu1992/Adobe_EntitySeg is gated on HuggingFace."
        echo "       1. request access: https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg"
        echo "       2. create a token: https://huggingface.co/settings/tokens"
        echo "       3. re-run: HF_TOKEN=hf_xxx bash setup/04_download_weights.sh"
    fi
else
    echo "[skip] CropFormer"
fi

# TAP
if [ ! -f tap_vit_h_v1_1.pkl ]; then
    echo "[download] TAP model"
    wget -q --continue --show-progress -O tap_vit_h_v1_1.pkl \
        "https://hf-mirror.com/BAAI/tokenize-anything/resolve/main/models/tap_vit_h_v1_1.pkl"
else
    echo "[skip] TAP model"
fi
if [ ! -f merged_2560.pkl ]; then
    echo "[download] TAP concepts"
    wget -q --continue --show-progress -O merged_2560.pkl \
        "https://hf-mirror.com/BAAI/tokenize-anything/resolve/main/concepts/merged_2560.pkl"
else
    echo "[skip] TAP concepts"
fi

# OVSeg (Google Drive, may need a proxy; try gdown)
if [ ! -f ovseg_swinbase_vitL14_ft_mpt.pth ]; then
    echo "[download] OVSeg (gdown, from Google Drive)"
    pip install -q gdown 2>/dev/null || true
    gdown 1cn-ohxgXDrDfkzC1QdO-fi8IjbjXmgKy -O ovseg_swinbase_vitL14_ft_mpt.pth || \
        echo "[WARN] OVSeg download failed; download it manually and place it as pretrained_weights/ovseg_swinbase_vitL14_ft_mpt.pth"
else
    echo "[skip] OVSeg"
fi

ls -lh
