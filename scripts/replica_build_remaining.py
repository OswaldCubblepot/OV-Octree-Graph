"""Run the full Replica pipeline for the 6 scenes that are not yet processed.

room0 and room1 have already been fully processed (proposals, features,
downstream stages, octree graph), so they are skipped here. The downstream
stages for the remaining scenes are identical to scripts/replica_build.py.
"""
import open3d as o3d  # noqa: F401
import warnings

warnings.filterwarnings('ignore')

import argparse
import multiprocessing
from time import perf_counter

import torch
from omegaconf import OmegaConf

from ovgraph.scene import ReplicaSceneGraph


def mgl_graph(dataset, video, idx, cfg, available_devices):
    device = available_devices.get()
    print(f"Processing {video = }, {idx = }, {device = } \n", flush=True)

    try:
        torch.cuda.set_device(f"cuda:{device}")

        scene_graph = ReplicaSceneGraph(
            cfg,
            scene_id=video,
            process_idx=idx,
            model_device=f"cuda",
        )

        scene_graph.generate_2d_proposals()
        scene_graph.compute_2d_feature()
        scene_graph.obtain_3d_segment()
        scene_graph.graph_back_projcection()
        scene_graph.evaluate_segmentation_opt()
        scene_graph.build_octree_graph()
    finally:
        # Always return the device token so a per-scene failure cannot starve
        # the remaining scenes (the token is otherwise leaked on exception).
        available_devices.put(device)


def main():
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--num_gpus", type=int, default=-1)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    dataset = cfg.data.data_root
    videos = ["room2", "office0", "office1", "office2", "office3", "office4"]

    if args.num_gpus == -1:
        max_workers = min(torch.cuda.device_count(), len(videos))
    else:
        max_workers = min(args.num_gpus, torch.cuda.device_count())

    print(f"{dataset = }", flush=True)
    print(f"{len(videos) = }", flush=True)
    print(f"{max_workers = }", flush=True)

    _available_devices = multiprocessing.Manager().Queue()
    for i in range(max_workers):
        _available_devices.put(i)

    tic = perf_counter()
    # maxtasksperchild=1 forces a fresh worker process per scene. The OVSeg
    # extractor (imported by compute_2d_feature) overwrites detectron2 registry
    # entries last-wins, which breaks the CropFormer detector build on the next
    # scene in a reused worker (KeyError: 'multi_scale_pixel_decoder').
    p = multiprocessing.Pool(max_workers, maxtasksperchild=1)
    for idx, scene in enumerate(videos):
        p.apply_async(
            mgl_graph, args=(dataset, scene, idx, cfg, _available_devices),
            error_callback=lambda e, s=scene: print(f"[ERROR] {s}: {type(e).__name__}: {e}", flush=True),
        )

    p.close()
    p.join()
    print(f"Process {len(videos)} takes {perf_counter() - tic}s", flush=True)


if __name__ == "__main__":
    main()
