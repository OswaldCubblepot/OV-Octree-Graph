# OV-Octree-Graph 论文复现总结

> **ICCV 2025** — *Open-Vocabulary Octree-Graph for 3D Scene Understanding*
> arXiv: [2411.16253](https://arxiv.org/abs/2411.16253)
>
> - 原始仓库（约 80% 完整）：https://github.com/yifeisu/OV-Octree-Graph
> - 补齐后仓库：https://github.com/OswaldCubblepot/OV-Octree-Graph
> - 复现日期：2026-09-21（代码补齐）→ 2026-09-24（端到端复现完成）

---

## 〇、结论摘要

原仓库只发布了约 80% 的代码：**八叉树模块（octree-graph / 检索 / 路径规划）完全没有实现**，vocab 类别特征文件缺失，且存在若干导致 ScanNet 检索 / 实例评测直接崩溃的 bug。本次工作：

1. **补全了缺失的 20%**（新增 4 个文件 + 修改 12 个文件），把整条 pipeline 从"数据加载 → 八叉树图 → 下游任务"完全接通；
2. 在 **WSL2 + RTX 4060 (8GB)** 上完成 **Replica 端到端复现**，语义分割 **mIoU 0.3158**（论文 0.320，误差约 1.3%）；
3. 渲染 8 个场景 × {预测 / GT / 八叉树} 共 **24 张结果图**，输出在 `renders/`，指标在 `results/`。

---

## 一、论文整体大纲（整个 pipeline）

这是一个**免训练（training-free）**的开放词汇 3D 场景理解框架：输入 RGB-D 扫描序列 + 重建点云，输出"实例地图 + 八叉树图"。

### Stage 1 — 2D 实例提议
对每帧 RGB 用 **CropFormer**（通用实体分割）生成类别无关的实例掩码。
→ `generate_2d_proposals()`（`ovgraph/scene/scene_replica.py` / `scene_scannet.py`）

### Stage 2 — 2D 特征提取
对每个掩码用 **OVSeg 的 CLIP 适配器**提取 768 维区域视觉特征（可选 TAP 生成 caption 文本特征）。
→ `compute_2d_feature()`

### Stage 3 — 3D 反投影 + 时序分组融合（CGSM，核心贡献①）
把 2D 掩码通过深度/可见性测试反投影到重建点云上，然后按**时间顺序分组**（Replica interval=50 帧、ScanNet interval=100 帧）合并 3D 片段：
1. 组内"轻量合并"：IoU ≥ 0.5 且特征余弦相似度 ≥ 0.8 → 并入已有 anchor；
2. 组边界处：DBSCAN 去噪 → 可见性过滤（≥ 0.1）→ 实例尺寸过滤 → 欠分割过滤；
3. **线性衰减阈值**的连通分量聚类，迭代合并（recall/IoU/similarity 阈值逐轮下调）。
→ `obtain_3d_segment()`（合并逻辑在 `ovgraph/utils/merging.py`）

### Stage 4 — 反向再投影
每个 3D 实例反投影回它出现过的所有帧，逐帧提取区域视觉特征 + TAP caption，形成"多视角特征袋"。
→ `graph_back_projcection()`

### Stage 5 — 特征聚合（IFA，核心贡献②）
对每个实例的多视角特征袋做去噪加权聚合——更靠近自己聚类中心、更远离邻居实例中心的特征权重更高（`rep_dif_denoise`），而不是简单平均。

### Stage 6 — 开放词汇分类
实例特征与类别名文本特征（vocab features，`.npy`）做余弦匹配，视觉特征与 caption 特征按 `scale_f = 0.7` 融合：
- 3D 语义分割：Replica mIoU 0.320 / ScanNet mIoU 0.393, mAcc 0.601；
- 3D 实例分割：ScanNet200 类别相关 / 类别无关 AP（openmask3d 协议）。
→ `evaluate_segmentation_opt()` / `evaluate_segmentation_20_opt()` / `evaluate_retrieval()`

### Stage 7 — 八叉树图（核心贡献③）
- 每个实例用一个**自适应八叉树**表示占用：根立方体按物体形状自适应初始化（适合墙/地板等大长宽比物体）、只在被占据处递归细分；
- 节点 = 实例（语义特征 + 中心 + 八叉树），边 = 空间语义关系（"left of / above / …" + 距离 + 3D 偏移向量）；
- 支持下游任务：**文本实例检索、占用查询、路径规划**；存储开销远小于点云（论文：全部自适应八叉树仅 42 KB vs 680 万点）。
→ 本次复现中补全：`ovgraph/structure/octree.py`、`ovgraph/structure/octree_graph.py`

---

## 二、代码审计结果（约 80% 完整 / 20% 缺失）

### 完整的部分（80%）
数据加载（Replica/ScanNet 的 RGB-D 流）、CropFormer/OVSeg/TAP 推理、3D 反投影、
时序分组合并、back-projection 特征提取、语义/实例分割评测、可视化脚本——全部存在且自洽（被调用的函数均有定义）。

### 缺失 / 未完成的部分（20%）

| 缺失项 | 说明 |
|---|---|
| **八叉树模块整个不存在** | README 宣称的 octree-graph、检索、路径规划没有任何代码（git 历史里也没有） |
| **vocab 特征文件缺失** | `ovgraph/evaluation/voc_features/` 下缺 4 个 `.npy`（`replica101/scannet20/scannet200_clip_l_14_vild`、`scannet200_clip_h_14`） |
| **配置键错误** | `evaluate_retrieval` 引用不存在的 `cfg.model.*`（应为 `cfg.extractor.*`）→ 直接崩溃 |
| **ScanNet 评估被注释掉** | `scripts/scannet_build.py` 里 `evaluate_retrieval` / `evaluate_segmentation_20_opt` 被注释，导致 eval 脚本读不到结果文件 |
| **若干 bug** | ① CropFormer `Instances(image_size=(H,H))` 应为 `(H,W)`；② `top1_cls` 用 CLASS_LABELS_200 索引 20 类结果；③ `rep_dif_denoise` 在 `eval.feature_name: feature` 时因 None 列表崩溃；④ `comput_voc_feature.py` 输出到当前目录而非 `voc_features/` |

---

## 三、相对原仓库的代码改动（补全 20%）

### 3.1 新增文件（4 个，均通过合成场景端到端测试）

| 文件 | 内容 |
|---|---|
| `ovgraph/structure/octree.py` | 自适应八叉树：根立方体按物体 AABB 自适应初始化、只细分被占据的 cell、批量占用查询（scipy cKDTree，零额外依赖） |
| `ovgraph/structure/octree_graph.py` | 八叉树图：节点=实例（语义/中心/八叉树），边=空间关系词+距离+偏移向量；下游任务 `query_by_feature`（检索）、`occupancy_query`（占位查询）、`plan_path`（A* 自由体素路径规划 + 安全余量） |
| `scripts/generate_vocab_features.py` | 复现缺失的 vocab 特征（OVSeg 协议 = OpenAI CLIP ViT-L/14 + ImageNet 提示词集成，不需要 OVSeg 权重；`*_clip_h_14` 用 open_clip ViT-H-14） |
| `scripts/octree_demo.py` | `info / retrieve / occupancy / plan` 四个下游任务演示；内置零依赖配置解析器（omegaconf → yaml → 内建解析逐级降级），无需 `pip install -e .` 即可运行 |

### 3.2 修改文件（12 个）

- `ovgraph/scene/scene_replica.py` / `scene_scannet.py`：新增 `build_octree_graph()` 方法；`rep_dif_denoise` 增加 None 保护（`feature_name: feature` 时返回 None，调用方保留均值特征）；修复 `evaluate_retrieval` 的 `cfg.model.*` → 正确的路径与文件名；`top1_cls` 改用 `CLASS_LABELS_20`
- `scripts/replica_build.py` / `scannet_build.py`：接通完整链路——ScanNet 启用 `evaluate_retrieval()` + `evaluate_segmentation_20_opt()`，两者末尾都调用 `build_octree_graph()`
- `configs/replica/*.yaml` / `configs/scannet/*.yaml`：新增 `octree: {min_cell_size: 0.05, max_depth: 8, edge_dist: 1.0}` 配置段
- `ovgraph/detector/CropFormer/inference.py`：`image_size=(H, W)` 修复 + 空掩码保护
- `ovgraph/evaluation/comput_voc_feature.py`：输出路径改为 `ovgraph/evaluation/voc_features/`
- `scripts/scannet_eval_instance_segment.py`：`rep_dif_denoise` 同样的 None 保护
- `README.md`：新增第 2 节（Vocabulary Features）和第 6 节（Octree-Graph and Downstream Tasks），修正实例分割评测命令

---

## 四、端到端复现：环境搭建与运行期修复

除上面 20% 的代码补全外，为了让原代码在真实环境跑起来，还做了一批**运行期/环境修复**（这些同样是"相对原仓库"的改变，原代码在标准环境里根本无法直接跑通）。

### 4.1 复现环境

| 项 | 值 |
|---|---|
| 系统 | Windows 11 + **WSL2**（Ubuntu） |
| Python | conda env `ovgraph`，Python 3.10 |
| PyTorch | 2.1.2 + cu118 |
| CUDA | 11.8 工具链（nvcc 11.8 + gcc-11 + cuda-cccl 的 thrust/cub 头，`TORCH_CUDA_ARCH_LIST=8.9`） |
| GPU | NVIDIA RTX 4060（**8GB 显存**，因此单卡顺序跑） |

### 4.2 依赖 / 构建修复

| 问题 | 修复 |
|---|---|
| `pip install mmcv==2.1.0` 在无 `CUDA_HOME` 时构建失败 | 改 `mmcv-lite==2.1.0`（推理只需 `import mmcv`） |
| setuptools ≥81 移除 `pkg_resources`，旧 setup.py（ovclip、mmcv）构建即崩 | `pip install --no-build-isolation`（含一切 setup.py 里 `import torch` 的包：detectron2/chamferdist/gradslam） |
| `tokenize-anything` 可编辑安装不生成 `tokenize_anything/version.py` | 手写 `version.py`（`version = "1.1.0a0"`） |
| CUDA 11.8 缺 `thrust`/`cub` 头 | 从 `cuda-cccl` 补 `include/{thrust,cub,cuda,nv}` |

### 4.3 运行期关键修复

| 问题 | 表现 | 修复 |
|---|---|---|
| **flash-attn 无法编译** | 原代码 `except ImportError` 后把 `apply_rotary_emb`/`flash_attn_func`/`flash_attn_with_kvcache` 置为 `None`，调用即崩 | 在 `3rdparty/tokenize-anything/tokenize_anything/modeling/text_decoder.py` 手写等价的**纯 PyTorch 因果注意力 + 旋转位置编码**回退（与手动 causal softmax 逐元素对比误差 0.0） |
| **OVSeg 污染 CropFormer 的 detectron2 注册表** | 第 2 个场景起报 `KeyError: 'multi_scale_pixel_decoder'` | OVSeg 导入时 monkeypatch fvcore `Registry._do_register` 为"后者覆盖前者"，把 CropFormer 的 `MaskFormerHead` 覆盖掉。在 `3rdparty/ovseg/open_vocab_seg/__init__.py` 改回 last-wins；同时 `multiprocessing.Pool(..., maxtasksperchild=1)` 让每个场景独立进程 |
| **8GB 显存 OOM** | CropFormer 与 OVSeg 两个模型同时驻留显存 | `ovgraph/scene/scene_replica.py` 在 `generate_2d_proposals`/`compute_2d_feature` 末尾 `del self.detector`/`del self.extractor` + `torch.cuda.empty_cache()` |
| **Replica 网格 PLY 解析失败** | Replica mesh PLY 的自定义 face 布局让 open3d RPly 报错 | `ovgraph/scene/scene_replica.py` 改用 `plyfile.PlyData.read()` |
| **`No module named 'datasets'`** | `datasets` 是仓库根的 namespace package（无 `__init__.py`），可编辑安装不覆盖它 | 启动时 `export PYTHONPATH=/mnt/d/OV-Octree-Graph-main:$PYTHONPATH` |

---

## 五、复现结果

### 5.1 Replica 语义分割（`results/*-segment.csv`）

| 场景 | mIoU | f-mIoU | mAcc | pAcc |
|---|---|---|---|---|
| room0 | 0.3745 | 0.6164 | 0.5283 | 0.7541 |
| room1 | 0.3524 | 0.5152 | 0.5820 | 0.6564 |
| room2 | 0.3603 | 0.5387 | 0.4406 | 0.5773 |
| office0 | 0.2983 | 0.6360 | 0.3518 | 0.7130 |
| office1 | 0.2027 | 0.3426 | 0.2305 | 0.3517 |
| office2 | 0.3069 | 0.6656 | 0.3351 | 0.7343 |
| office3 | 0.2325 | 0.4860 | 0.3898 | 0.6216 |
| office4 | 0.3991 | 0.5880 | 0.4411 | 0.6552 |
| **平均** | **0.3158** | 0.5485 | 0.4124 | 0.6329 |

> 论文报告 Replica 语义分割 **mIoU = 0.320**，本次复现 **0.3158**，相对误差约 **1.3%**，基本一致。

### 5.2 结果渲染图

8 个场景 × {预测 `_pred` / 真值 `_gt` / 八叉树图 `_octree`} 共 **24 张**，输出到 `renders/*.png`（open3d OffscreenRenderer，EGL 在 WSL 下可用）。

---

## 六、如何运行

### 6.1 完整论文复现（Linux + CUDA）

```bash
# 1. conda 环境（按 README 1.1~1.3 节）
conda create -n ovgraph python=3.10
# 安装 torch 2.1.2(cu118) + detectron2 + CropFormer(Entity) + OVSeg + TAP + gradslam
# 注：pytorch3d、faiss、torch-scatter、flash-attn、MinkowskiEngine 主流程实际用不到（仅死代码引用），可先跳过
# 关键：mmcv 用 mmcv-lite，安装统一加 --no-build-isolation，tokenize-anything 手写 version.py

# 2. 下载权重（README 2.5 节）：CropFormer、OVSeg、TAP 三个权重到 pretrained_weights/

# 3. 生成 vocab 特征（本次新增）
python scripts/generate_vocab_features.py --models l14vild,h14

# 4. 准备数据集
#    Replica：公开，按 README 3 节组织到 data/replica
#    ScanNet：需签协议，运行 tools/preprocess_scannet.py

# 5. 构建 + 评测（WSL 下需先 export PYTHONPATH）
export PYTHONPATH=/path/to/OV-Octree-Graph-main:$PYTHONPATH
bash run/replica_build.sh                      # Replica（含语义分割评测 + 八叉树图）
python scripts/replica_eval_semantic_segment.py
bash run/scannet_bulid.sh                      # ScanNet（含检索/语义评测 + 八叉树图）
python scripts/scannet_eval_semantic_segment.py
python scripts/scannet_eval_instance_segment.py
python scripts/scannet_eval_instance_segment_class_agnostic.py
```

### 6.2 八叉树图下游任务（演示，轻量依赖）

八叉树模块本身只需要 numpy/scipy/networkx，构建完成后即可用：

```bash
# 图统计信息
python scripts/octree_demo.py --config configs/replica/replica_cropformer_ovseg_tap.yaml \
    --scene room0 --task info --verbose

# 文本实例检索（需要 clip 或 open_clip 编码查询文本）
python scripts/octree_demo.py --config configs/replica/replica_cropformer_ovseg_tap.yaml \
    --scene room0 --task retrieve --query "a red chair" --top_k 5

# 占用查询：某个 3D 点被哪个实例占据
python scripts/octree_demo.py --config configs/replica/replica_cropformer_ovseg_tap.yaml \
    --scene room0 --task occupancy --point=1.0,0.5,1.2

# 路径规划：A* 在自适应八叉树的自由体素上搜索
python scripts/octree_demo.py --config configs/scannet/scannet_cropformer_ovseg_tap.yaml \
    --scene scene0011_00 --task plan --start=0,0,0 --goal=2,0,3 --margin 0.1 --vis
```

---

## 附：关键文件索引

| 文件 | 作用 |
|---|---|
| `ovgraph/scene/scene_replica.py` / `scene_scannet.py` | 主流水线（2D 提议 → 特征 → 3D 融合 → 反投影 → 评测 → 八叉树图） |
| `ovgraph/utils/merging.py` | 投影/反投影、IoU/召回矩阵、合并匹配（numba 加速） |
| `ovgraph/structure/scene_graph.py` | `Segment` / `InstanceList` 数据结构 |
| `ovgraph/structure/octree.py` | **新增**：自适应八叉树（占用表示） |
| `ovgraph/structure/octree_graph.py` | **新增**：八叉树图 + 检索/占位查询/路径规划 |
| `ovgraph/utils/metric.py` | 检索 AP、语义分割 mIoU 等评测指标 |
| `ovgraph/detector/CropFormer/inference.py` | CropFormer 检测器封装 |
| `ovgraph/extractor/OVSeg/inference.py` | OVSeg + TAP 特征提取器封装 |
| `3rdparty/tokenize-anything/.../text_decoder.py` | **修改**：flash-attn 纯 PyTorch 回退 |
| `3rdparty/ovseg/open_vocab_seg/__init__.py` | **修改**：fvcore 注册表 last-wins，避免覆盖 CropFormer |
| `scripts/*_build.py` / `scripts/*_eval*.py` | 构建与评测入口 |
| `scripts/generate_vocab_features.py` | **新增**：生成缺失的类别文本特征 |
| `scripts/octree_demo.py` | **新增**：下游任务演示 |
| `configs/*/*.yaml` | 管线配置（含新增 `octree` 段） |
| `results/` | 本次复现的指标 CSV（`segment.csv` 平均 mIoU 0.3158） |
| `renders/` | 本次复现的 24 张结果图 |
