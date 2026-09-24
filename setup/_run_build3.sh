#!/usr/bin/env bash
# Third attempt: setuptools 60.2.0 lacks the PEP 660 build_editable hook, so
# every `pip install -e` failed. Upgrade setuptools to 69.5.1 (has PEP 660,
# pkg_resources and legacy develop) and redo all editable installs.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ovgraph
export PATH=/usr/local/cuda-11.8/bin:${PATH}
export CUDA_HOME=/usr/local/cuda-11.8
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=8
cd ~/OV-Octree-Graph

pip install -q "setuptools==69.5.1" wheel
echo "setuptools: $(pip show setuptools | grep Version)"

for pkg in \
    3rdparty/ovseg/third_party/CLIP \
    3rdparty/tokenize-anything \
    3rdparty/CLIP \
    3rdparty/chamferdist \
    3rdparty/gradslam \
    3rdparty/detectron2 \
    .; do
    echo "[editable] $pkg"
    pip install -q --no-build-isolation -e "$pkg"
done

echo "[build] CropFormer entity_api (CUDA op)"
make -j8 -C 3rdparty/Entity/Entityv2/CropFormer/entity_api/PythonAPI
echo "[build] CropFormer mask2former ops (MSDeformAttn)"
sh 3rdparty/Entity/Entityv2/CropFormer/mask2former/modeling/pixel_decoder/ops/make.sh

echo "=== verify ==="
python -c "import chamferdist, gradslam, detectron2, ovclip, clip, open_clip, tokenize_anything, open3d, mmcv, ovgraph; print('ALL IMPORTS OK')"
echo "BUILD3_DONE"
