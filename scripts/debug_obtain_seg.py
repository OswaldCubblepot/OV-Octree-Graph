"""Diagnose why obtain_3d_segment produces 0 nodes for room0."""
import copy
import os
import sys

import numpy as np
import open3d as o3d
import torch
from omegaconf import OmegaConf

REPO = "/mnt/d/OV-Octree-Graph-main"
os.chdir(REPO)
sys.path.insert(0, REPO)

from ovgraph.scene import ReplicaSceneGraph
from ovgraph.utils.merging import (
    compute_projected_pts, compute_visibility_mask, compute_visible_masked_pts,
    resolve_overlapping_masks,
)
from ovgraph.utils.utils import read_detectron_instances

cfg = OmegaConf.load("configs/replica/replica_cropformer_ovseg_tap.yaml")

sg = ReplicaSceneGraph(cfg, scene_id="room0", process_idx=0, model_device="cuda")
print(f"n rgbd frames = {len(sg.rgbd)}")
print(f"gt points = {len(sg.gt_xyz)}")
print(f"output_instance_folder = {sg.output_instance_folder}")

# run the first few frames and report back-projection stats
from tqdm import tqdm

for idx in [0, 1, 5, 10, 50, 100]:
    frame_id = os.path.basename(sg.rgbd.color_paths[idx]).split('.')[0]
    detect_result_path = os.path.join(sg.output_instance_folder, f'{frame_id}.pkl')
    detect_output = read_detectron_instances(detect_result_path, rle_to_mask=True)
    pred_scores = detect_output.scores.numpy()
    pred_masks = detect_output.pred_masks.numpy()
    pred_features = detect_output.pred_box_features.to(sg.data_device)
    pred_masks = resolve_overlapping_masks(pred_masks, pred_scores, device=sg.data_device)

    _, depth_im, cam_intr, pose = sg.rgbd[idx]
    cam_intr = cam_intr[:3, :3]
    pcd = copy.deepcopy(sg.gt_point).transform(np.linalg.inv(pose))
    scene_pts = np.asarray(pcd.points)
    projected_pts = compute_projected_pts(scene_pts, cam_intr)
    visibility_mask = compute_visibility_mask(scene_pts, projected_pts, depth_im, depth_thresh=cfg.merge.depth_thresh)
    n_visible = int(visibility_mask.sum())

    masked_pts = compute_visible_masked_pts(scene_pts, projected_pts, visibility_mask, pred_masks)
    mask_areas = masked_pts.sum(axis=1)

    print(f"frame {idx} ({frame_id}): n_inst={pred_scores.shape[0]} "
          f"n_visible_pts={n_visible} mask_areas min/med/max = "
          f"{mask_areas.min() if len(mask_areas) else -1}/{np.median(mask_areas) if len(mask_areas) else -1}/{mask_areas.max() if len(mask_areas) else -1} "
          f"n_valid(>=25)={(mask_areas >= cfg.merge.size_thresh).sum()}")
