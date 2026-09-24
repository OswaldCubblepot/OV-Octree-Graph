#!/usr/bin/env bash
# Continuation of the build: the steps that failed under pip 26's build
# isolation. All editable installs now use --no-build-isolation so that the
# real env's setuptools/pkg_resources/torch are visible.
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ovgraph
export PATH=/usr/local/cuda-11.8/bin:${PATH}
export CUDA_HOME=/usr/local/cuda-11.8
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=8
cd ~/OV-Octree-Graph

echo "[pip] ovclip (ovseg vendored CLIP)"
pip install -q --no-build-isolation -e 3rdparty/ovseg/third_party/CLIP
echo "[pip] tokenize-anything"
pip install -q --no-build-isolation -e 3rdparty/tokenize-anything
echo "[pip] openai CLIP"
pip install -q --no-build-isolation -e 3rdparty/CLIP

echo "[build] chamferdist"
pip install -q --no-build-isolation -e 3rdparty/chamferdist
echo "[build] gradslam"
pip install -q --no-build-isolation -e 3rdparty/gradslam
echo "[build] detectron2"
pip install -q --no-build-isolation -e 3rdparty/detectron2

echo "[build] CropFormer entity_api (CUDA op)"
make -j8 -C 3rdparty/Entity/Entityv2/CropFormer/entity_api/PythonAPI
echo "[build] CropFormer mask2former ops (MSDeformAttn)"
sh 3rdparty/Entity/Entityv2/CropFormer/mask2former/modeling/pixel_decoder/ops/make.sh

echo "[build] ovgraph"
pip install -q --no-build-isolation -e .

echo "=== verify ==="
python -c "import chamferdist, gradslam, detectron2, ovclip, clip, open_clip, tokenize_anything, open3d, mmcv, ovgraph; print('ALL IMPORTS OK')"
echo "BUILD2_DONE"
