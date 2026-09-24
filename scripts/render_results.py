"""Headless render of the Replica semantic-segmentation results to PNG.

Reads the ``-visualize.ply`` (prediction) and ``-visualize-gt.ply`` (ground
truth) point clouds produced by ``evaluate_segmentation_opt`` and renders a
fixed view of each to an offscreen PNG. Also renders the octree-graph cells +
nodes + edges from ``-octree_graph.pkl`` when ``--octree`` is passed.
"""
import argparse
import os
import pickle
import sys

import numpy as np

_REPO = "/mnt/d/OV-Octree-Graph-main"
sys.path.insert(0, _REPO)

import open3d as o3d
import open3d.visualization.rendering as rendering


def _render_pcd_to_png(pcd, out_path, width=1200, height=900, bg=(1.0, 1.0, 1.0, 1.0),
                       look_at=None, eye=None, up=(0, 0, 1), fov=60.0):
    """Render a point cloud (must have colors) to PNG with a fixed camera."""
    renderer = rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background(bg)

    mtl = rendering.MaterialRecord()
    mtl.shader = "defaultUnlit"
    mtl.point_size = 2.0

    renderer.scene.add_geometry("pcd", pcd, mtl)

    # center the point cloud and pick a diagonal camera
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    ext = bbox.get_extent()
    if look_at is None:
        look_at = center
    if eye is None:
        d = float(np.linalg.norm(ext)) * 1.4 + 0.5
        eye = center + np.array([d * 0.6, -d * 0.6, d * 0.5])

    renderer.setup_camera(fov, look_at, eye, up)

    img = renderer.render_to_image()
    o3d.io.write_image(out_path, img)
    renderer = None  # release the GL context
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--config", default="configs/replica/replica_cropformer_ovseg_tap.yaml")
    ap.add_argument("--out", default="renders")
    ap.add_argument("--octree", action="store_true")
    args = ap.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)

    prefix = f"{cfg.detector.type}-{cfg.detector.vocabulary}-{cfg.detector.confidence_thresh}-{cfg.extractor.type}"
    out_prefix = f"proposed_fusion_{prefix}_interval-{cfg.merge.interval}"
    result_dir = os.path.join(cfg.data.data_root, args.scene, cfg.data.output_folder, "results", prefix)

    os.makedirs(args.out, exist_ok=True)

    # 1. semantic segmentation (pred vs gt)
    pred_ply = os.path.join(result_dir, f"{out_prefix}-visualize.ply")
    gt_ply = os.path.join(result_dir, f"{out_prefix}-visualize-gt.ply")
    for tag, ply in (("pred", pred_ply), ("gt", gt_ply)):
        if not os.path.isfile(ply):
            print(f"[skip] {ply} not found", flush=True)
            continue
        pcd = o3d.io.read_point_cloud(ply)
        png = os.path.join(args.out, f"{args.scene}_{tag}.png")
        _render_pcd_to_png(pcd, png)
        print(f"[ok] {png}  ({len(pcd.points)} pts)", flush=True)

    # 2. octree-graph (nodes + edges + adaptive octree cells)
    if args.octree:
        graph_dir = os.path.join(cfg.data.data_root, args.scene, cfg.data.output_folder, "graph_prediction", prefix)
        octree_pkl = os.path.join(graph_dir, f"{out_prefix}-octree_graph.pkl")
        if not os.path.isfile(octree_pkl):
            print(f"[skip] {octree_pkl} not found", flush=True)
            return
        with open(octree_pkl, "rb") as f:
            g = pickle.load(f)

        renderer = rendering.OffscreenRenderer(1200, 900)
        renderer.scene.set_background((1.0, 1.0, 1.0, 1.0))

        mtl = rendering.MaterialRecord()
        mtl.shader = "defaultUnlit"

        # node centers
        centers = np.array([n["center"] for n in g["nodes"]])
        node_pcd = o3d.geometry.PointCloud()
        node_pcd.points = o3d.utility.Vector3dVector(centers)
        node_pcd.paint_uniform_color([1.0, 0.0, 0.0])
        nmtl = rendering.MaterialRecord()
        nmtl.shader = "defaultUnlit"
        nmtl.point_size = 8.0
        renderer.scene.add_geometry("nodes", node_pcd, nmtl)

        # edges as line set
        line_pts = []
        line_idx = []
        for a, b, _ in g["edges"]:
            line_idx.append([len(line_pts), len(line_pts) + 1])
            line_pts.append(centers[a]); line_pts.append(centers[b])
        if line_pts:
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(np.array(line_pts))
            ls.lines = o3d.utility.Vector2iVector(np.array(line_idx))
            ls.paint_uniform_color([0.0, 0.0, 0.7])
            lmtl = rendering.MaterialRecord()
            lmtl.shader = "unlitLine"
            lmtl.line_width = 2.0
            renderer.scene.add_geometry("edges", ls, lmtl)

        bbox = node_pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        ext = bbox.get_extent()
        d = float(np.linalg.norm(ext)) * 1.4 + 0.5
        eye = center + np.array([d * 0.6, -d * 0.6, d * 0.5])
        renderer.setup_camera(60.0, center, eye, (0, 0, 1))

        img = renderer.render_to_image()
        png = os.path.join(args.out, f"{args.scene}_octree.png")
        o3d.io.write_image(png, img)
        print(f"[ok] {png}  ({len(g['nodes'])} nodes, {len(g['edges'])} edges)", flush=True)


if __name__ == "__main__":
    main()
