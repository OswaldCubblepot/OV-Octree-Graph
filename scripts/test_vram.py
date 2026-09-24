"""Measure VRAM: detector alone, then detector + extractor together.

IMPORTANT: mirrors the real pipeline order from scripts/replica_build.py ->
ReplicaSceneGraph.generate_2d_proposals() builds the detector FIRST and
compute_2d_feature() builds the extractor SECOND. The two forks (Entity
mask2former / OVSeg open_vocab_seg) register overlapping detectron2 component
names, so the detector MUST be fully built before open_vocab_seg is imported.
"""
import os
import sys

import torch

REPO = "/mnt/d/OV-Octree-Graph-main"
os.chdir(REPO)
sys.path.insert(0, REPO)


def vram():
    return torch.cuda.memory_allocated() / 1024**2


def reset():
    torch.cuda.reset_peak_memory_stats()


from ovgraph.detector.CropFormer.inference import CropFormerDetector  # noqa: E402

CFG_DET = os.path.join(REPO, "3rdparty/Entity/Entityv2/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml")
CFG_EXT = os.path.join(REPO, "3rdparty/ovseg/configs/ovseg_swinB_vitL_demo.yaml")

print("[1] build detector ...", flush=True)
det = CropFormerDetector(CFG_DET, device="cuda")
print(f"    allocated after detector: {vram():.0f} MiB", flush=True)

print("[2] build extractor ...", flush=True)
from ovgraph.extractor.OVSeg.inference import OVsegTapExtractor  # noqa: E402
try:
    ext = OVsegTapExtractor(CFG_EXT, device="cuda")
    print(f"    allocated after extractor: {vram():.0f} MiB  (peak {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB)", flush=True)
    print("[ok] BOTH FIT", flush=True)
except Exception as e:
    print(f"[FAIL] extractor build raised: {type(e).__name__}: {e}", flush=True)
    print(f"    peak allocated: {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB", flush=True)
