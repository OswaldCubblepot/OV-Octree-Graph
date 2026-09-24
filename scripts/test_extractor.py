"""Smoke-test the full OVsegTapExtractor (OVSeg + TAP + open_clip ViT-H-14).

This is the exact object the Replica pipeline builds in compute_2d_feature();
it downloads the open_clip ViT-H-14 checkpoint into the HF cache on first run.
Run from repo root with HF_ENDPOINT set for the mirror.
"""
import os
import sys

REPO = "/mnt/d/OV-Octree-Graph-main"
os.chdir(REPO)
sys.path.insert(0, REPO)

from ovgraph.extractor.OVSeg.inference import OVsegTapExtractor  # noqa: E402

print("[load] OVsegTapExtractor (OVSeg + TAP + open_clip ViT-H-14) ...", flush=True)
ext = OVsegTapExtractor(
    os.path.join(REPO, "3rdparty/ovseg/configs/ovseg_swinB_vitL_demo.yaml"),
    device="cuda",
)
print("[ok] EXTRACTOR READY", flush=True)
