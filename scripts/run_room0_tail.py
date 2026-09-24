"""Run the un-verified downstream stages for room0 (proposals + features are
already on disk) in the main process so any error prints a full traceback."""
import os
import sys

from omegaconf import OmegaConf

REPO = "/mnt/d/OV-Octree-Graph-main"
os.chdir(REPO)
sys.path.insert(0, REPO)

from ovgraph.scene import ReplicaSceneGraph

cfg = OmegaConf.load("configs/replica/replica_cropformer_ovseg_tap.yaml")
sg = ReplicaSceneGraph(cfg, scene_id="room0", process_idx=0, model_device="cuda")

print(">>> obtain_3d_segment", flush=True)
sg.obtain_3d_segment()
print(">>> graph_back_projcection", flush=True)
sg.graph_back_projcection()
print(">>> evaluate_segmentation_opt", flush=True)
sg.evaluate_segmentation_opt()
print(">>> build_octree_graph", flush=True)
sg.build_octree_graph()
print("DONE room0 tail", flush=True)
