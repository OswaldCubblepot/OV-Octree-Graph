# OV-Octree-Graph 论文复现总结

> ICCV 2025 — *Open-Vocabulary Octree-Graph for 3D Scene Understanding*
> arXiv: 2411.16253 | 代码仓库: https://github.com/yifeisu/OV-Octree-Graph
> 复现日期: 2026-09-21

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

## 三、本次复现补齐的内容

### 新增文件（4 个，均通过合成场景端到端测试）

| 文件 | 内容 |
|---|---|
| `ovgraph/structure/octree.py` | 自适应八叉树：根立方体按物体 AABB 自适应初始化、只细分被占据的 cell、批量占用查询（scipy cKDTree，零额外依赖） |
| `ovgraph/structure/octree_graph.py` | 八叉树图：节点=实例（语义/中心/八叉树），边=空间关系词+距离+偏移向量；下游任务 `query_by_feature`（检索）、`occupancy_query`（占位查询）、`plan_path`（A* 自由体素路径规划 + 安全余量） |
| `scripts/generate_vocab_features.py` | 复现缺失的 vocab 特征（OVSeg 协议 = OpenAI CLIP ViT-L/14 + ImageNet 提示词集成，不需要 OVSeg 权重；`*_clip_h_14` 用 open_clip ViT-H-14） |
| `scripts/octree_demo.py` | `info / retrieve / occupancy / plan` 四个下游任务演示；内置零依赖配置解析器（omegaconf → yaml → 内建解析逐级降级），无需 `pip install -e .` 即可运行 |

### 修改文件（12 个）

- `ovgraph/scene/scene_replica.py` / `scene_scannet.py`：新增 `build_octree_graph()` 方法；`rep_dif_denoise` 增加 None 保护（`feature_name: feature` 时返回 None，调用方保留均值特征）；修复 `evaluate_retrieval` 的 `cfg.model.*` → 正确的路径与文件名；`top1_cls` 改用 `CLASS_LABELS_20`
- `scripts/replica_build.py` / `scannet_build.py`：接通完整链路——ScanNet 启用 `evaluate_retrieval()` + `evaluate_segmentation_20_opt()`，两者末尾都调用 `build_octree_graph()`
- `configs/replica/*.yaml` / `configs/scannet/*.yaml`：新增 `octree: {min_cell_size: 0.05, max_depth: 8, edge_dist: 1.0}` 配置段
- `ovgraph/detector/CropFormer/inference.py`：`image_size=(H, W)` 修复 + 空掩码保护
- `ovgraph/evaluation/comput_voc_feature.py`：输出路径改为 `ovgraph/evaluation/voc_features/`
- `scripts/scannet_eval_instance_segment.py`：`rep_dif_denoise` 同样的 None 保护
- `README.md`：新增第 2 节（Vocabulary Features）和第 6 节（Octree-Graph and Downstream Tasks），修正实例分割评测命令

---

## 四、如何运行

### 4.1 完整论文复现（需要 Linux + CUDA）

当前机器是 Windows + 系统 Python 3.12（无 torch）。README 的安装步骤全部是 Linux 指令，detectron2 / CropFormer 在 Windows 上很难编译，**建议用 WSL2 或 Linux 机器**：

```bash
# 1. conda 环境（按 README 1.1~1.3 节）
conda create -n ovgraph python=3.10
# 安装 torch 2.1.2(cu118) + detectron2 + CropFormer(Entity) + OVSeg + TAP + gradslam
# 注：pytorch3d、faiss、torch-scatter、flash-attn、MinkowskiEngine 主流程实际用不到（仅死代码引用），可先跳过

# 2. 下载权重（README 2.5 节）：CropFormer、OVSeg、TAP 三个权重到 pretrained_weights/

# 3. 生成 vocab 特征（本次新增）
python scripts/generate_vocab_features.py --models l14vild,h14

# 4. 准备数据集
#    Replica：公开，按 README 3 节组织到 data/replica
#    ScanNet：需签协议，运行 tools/preprocess_scannet.py

# 5. 构建 + 评测
bash run/replica_build.sh                      # Replica（含语义分割评测 + 八叉树图）
python scripts/replica_eval_semantic_segment.py
bash run/scannet_bulid.sh                      # ScanNet（含检索/语义评测 + 八叉树图）
python scripts/scannet_eval_semantic_segment.py
python scripts/scannet_eval_instance_segment.py
python scripts/scannet_eval_instance_segment_class_agnostic.py
```

### 4.2 八叉树图下游任务（演示，轻量依赖）

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
| `scripts/*_build.py` / `scripts/*_eval*.py` | 构建与评测入口 |
| `scripts/generate_vocab_features.py` | **新增**：生成缺失的类别文本特征 |
| `scripts/octree_demo.py` | **新增**：下游任务演示 |
| `configs/*/*.yaml` | 管线配置（含新增 `octree` 段） |
