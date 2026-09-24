"""
Downstream-task demo on top of the octree-graph.

Supported tasks (run after ``scripts/*_build.py`` has produced
``*-octree_graph.pkl`` in the scene's graph_prediction folder):

  info       print graph statistics (nodes, edges, storage)
  retrieve   text-based instance retrieval, e.g. --query "chair"
  occupancy  which instance occupies a 3D point, e.g. --point 1.0 0.5 1.2
  plan       collision-free path planning, e.g. --start ... --goal ...
             (A* on the free voxels of the adaptive octrees)

Examples:
  python scripts/octree_demo.py --config configs/replica/replica_cropformer_ovseg_tap.yaml \
      --scene room0 --task retrieve --query "a red chair" --top_k 5
  python scripts/octree_demo.py --config configs/scannet/scannet_cropformer_ovseg_tap.yaml \
      --scene scene0011_00 --task plan --start 0 0 0 --goal 2 0 3 --margin 0.1 --vis
"""

import argparse
import os
import sys

import numpy as np

# make the demo runnable without `pip install -e .`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ovgraph.structure.octree_graph import OctreeGraph
from ovgraph.utils.scene_path import parse_scene_path


class _DictCfg(dict):
    """Attribute-style access over a plain dict (cfg.data.root -> dict access)."""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(key)
        return _DictCfg(value) if isinstance(value, dict) else value


def _load_config(path):
    """Load the pipeline config; degrade gracefully if omegaconf / yaml are absent."""
    try:
        from omegaconf import OmegaConf
        return OmegaConf.load(path)
    except ImportError:
        pass
    try:
        import yaml
        with open(path) as f:
            return _DictCfg(yaml.safe_load(f))
    except ImportError:
        pass
    # minimal indentation-aware parser for the configs shipped with the repo
    def _cast(value):
        if value in ("True", "False"):
            return value == "True"
        if value.lower() in ("none", "null", "~"):
            return None
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value.strip("'\"")

    cfg = {}
    stack = [(-1, cfg)]  # (indent, dict)
    for raw in open(path):
        line = raw.split("#")[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        key = key.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if value.strip():  # leaf
            stack[-1][1][key] = _cast(value.strip())
        else:  # new section
            section = {}
            stack[-1][1][key] = section
            stack.append((indent, section))
    return _DictCfg(cfg)

# prompt templates used to encode the text query (same family as the
# official vocab features, see scripts/generate_vocab_features.py)
IMAGENET_PROMPT = [
    "a photo of a {}.",
    "a photo of the {}.",
    "There is a {} in the scene",
    "a photo of a {} in the scene",
    "a bad photo of a {}.",
]


def find_octree_graph(cfg, scene_id):
    _, _, _, _, output_predict_folder, _, out_file_prefix = parse_scene_path(cfg, scene_id)
    output_file = f"proposed_fusion_{out_file_prefix}_interval-{cfg.merge.interval}-octree_graph.pkl"
    path = os.path.join(output_predict_folder, output_file)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} does not exist; run scripts/{cfg.data.type}_build.py first")
    return OctreeGraph.load(path)


def encode_text(query, feat_dim, device):
    """Encode the query text with a text encoder matching ``feat_dim``."""
    try:
        import torch
    except ImportError:
        raise RuntimeError(
            "text encoding needs torch; install `torch` and `clip` (feat_dim=768) "
            "or `open_clip_torch` (feat_dim=1024), or pass a pre-computed feature "
            "vector to `OctreeGraph.query_by_feature` directly")
    if feat_dim == 768:
        try:
            import clip
        except ImportError:
            raise RuntimeError(
                "instance features are 768-d (OVSeg CLIP ViT-L/14); install the "
                "`clip` package (pip install git+https://github.com/openai/CLIP.git) "
                "so the query text can be encoded in the same space")
        model, _ = clip.load("ViT-L/14", device=device)
        model.eval()
        with torch.no_grad():
            bucket = []
            for template in IMAGENET_PROMPT:
                tokens = clip.tokenize(template.format(query)).to(device)
                feat = model.encode_text(tokens)
                feat /= feat.norm(dim=-1, keepdim=True)
                bucket.append(feat)
            feat = torch.stack(bucket).mean(0)
            feat /= feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()
    elif feat_dim == 1024:
        try:
            import open_clip
        except ImportError:
            raise RuntimeError(
                "instance features are 1024-d (open_clip ViT-H-14); install "
                "`open_clip_torch` so the query text can be encoded in the same space")
        model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
        model = model.to(device)
        model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-H-14")
        with torch.no_grad():
            tokens = tokenizer(f"a photo of a {query}").to(device)
            feat = model.encode_text(tokens)
            feat /= feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()
    else:
        raise ValueError(f"unsupported feature dim {feat_dim}")


def task_retrieve(G, args, device):
    feat_dim = None
    for _, attrs in G.nodes(data=True):
        if attrs["feature"] is not None:
            feat_dim = attrs["feature"].shape[-1]
            break
    query_feat = encode_text(args.query, feat_dim, device)
    results = G.query_by_feature(query_feat, top_k=args.top_k)
    print(f"\nTop-{args.top_k} retrieved instances for query \"{args.query}\":")
    for rank, (score, node) in enumerate(results, 1):
        attrs = G.nodes[node]
        center = attrs["center"]
        print(f"  #{rank} node {node}: score {score:.3f} | class {attrs['top1_cls']} | "
              f"center ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) | "
              f"{attrs['n_pts']} pts | {len(attrs['octree'])} cells")


def task_occupancy(G, args):
    pt = np.array([float(v) for v in args.point.split(",")])
    occ = G.occupancy_query(pt)
    if occ is None:
        print(f"point {pt} is in free space")
    else:
        attrs = G.nodes[occ]
        print(f"point {pt} is occupied by instance {occ} "
              f"(class {attrs['top1_cls']}, center {attrs['center']})")


def task_plan(G, args):
    start = np.array([float(v) for v in args.start.split(",")])
    goal = np.array([float(v) for v in args.goal.split(",")])
    waypoints, info = G.plan_path(start, goal, resolution=args.resolution, margin=args.margin)
    if waypoints is None:
        print(f"No path found from {start} to {goal}: {info.get('reason')}")
        return
    print(f"Path found with {len(waypoints)} waypoints "
          f"(expanded {info['expanded_nodes']} nodes, resolution {info['resolution']}m)")
    print("waypoints (start -> goal):")
    for p in waypoints[:: max(1, len(waypoints) // 10)]:
        print(f"  ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")

    if args.vis:
        visualize_path(G, waypoints, args)


def task_info(G, args):
    n_cells = sum(len(a["octree"]) for _, a in G.nodes(data=True))
    n_pts = sum(a["n_pts"] for _, a in G.nodes(data=True))
    print(f"scene: {G.graph.get('scan')}")
    print(f"nodes (instances): {G.number_of_nodes()}, edges (relations): {G.number_of_edges()}")
    print(f"occupied octree cells: {n_cells}, original points: {n_pts}")
    print(f"octree storage: {G.storage_bytes() / 1024:.1f} KB vs "
          f"point storage: {n_pts * 3 * 4 / 1024:.1f} KB")
    if args.verbose:
        for node, attrs in G.nodes(data=True):
            print(f"  node {node}: {attrs['top1_cls']} at {attrs['center']}, "
                  f"{attrs['n_pts']} pts, {len(attrs['octree'])} cells")
        for u, v, attrs in list(G.edges(data=True))[:50]:
            print(f"  edge {u}-{v}: {attrs['relation']} (dist {attrs['distance']:.2f})")


def visualize_path(G, waypoints, args):
    import open3d as o3d
    # occupied cells of every instance as a sparse point cloud
    cells = np.concatenate([a["octree"].cells[:, :3] for _, a in G.nodes(data=True)
                            if len(a["octree"].cells)]) if G.number_of_nodes() else np.empty((0, 3))
    obstacles = o3d.geometry.PointCloud()
    obstacles.points = o3d.utility.Vector3dVector(cells)
    obstacles.paint_uniform_color([0.8, 0.3, 0.3])

    # instance centers
    centers = np.stack([a["center"] for _, a in G.nodes(data=True)]) if G.number_of_nodes() else np.empty((0, 3))
    centers_pcd = o3d.geometry.PointCloud()
    centers_pcd.points = o3d.utility.Vector3dVector(centers)
    centers_pcd.paint_uniform_color([0.3, 0.6, 0.9])

    # path
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(waypoints)
    line.lines = o3d.utility.Vector2iVector(
        np.stack([np.arange(len(waypoints) - 1), np.arange(1, len(waypoints))], axis=1))
    line.paint_uniform_color([0.1, 0.8, 0.2])

    start_sphere = o3d.geometry.TriangleMesh().create_sphere(radius=0.05)
    start_sphere.translate(waypoints[0])
    start_sphere.paint_uniform_color([0.0, 1.0, 0.0])
    goal_sphere = o3d.geometry.TriangleMesh().create_sphere(radius=0.05)
    goal_sphere.translate(waypoints[-1])
    goal_sphere.paint_uniform_color([1.0, 0.8, 0.0])

    vis = [obstacles, centers_pcd, line, start_sphere, goal_sphere]
    if args.save_ply:
        out = os.path.join("results", f"{G.graph.get('scan')}-octree_path.ply")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        merged = obstacles + centers_pcd
        o3d.io.write_point_cloud(out, merged)
        print(f"saved {out}")
    o3d.visualization.draw_geometries(vis, window_name="octree-graph path planning")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--task", type=str, required=True,
                        choices=["info", "retrieve", "occupancy", "plan"])
    parser.add_argument("--query", type=str, default="chair")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--point", type=str, default="0,0,0")
    parser.add_argument("--start", type=str, default="0,0,0")
    parser.add_argument("--goal", type=str, default="1,0,1")
    parser.add_argument("--resolution", type=float, default=None)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--save_ply", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    G = find_octree_graph(cfg, args.scene)

    if args.task == "info":
        task_info(G, args)
    elif args.task == "retrieve":
        task_retrieve(G, args, args.device)
    elif args.task == "occupancy":
        task_occupancy(G, args)
    elif args.task == "plan":
        task_plan(G, args)


if __name__ == "__main__":
    main()
