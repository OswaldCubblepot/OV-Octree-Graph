"""Smoke-test the CropFormer detector on one Replica frame."""
import os
import sys

import cv2

REPO = "/mnt/d/OV-Octree-Graph-main"
os.chdir(REPO)
sys.path.insert(0, REPO)

from ovgraph.detector.CropFormer.inference import CropFormerDetector  # noqa: E402

CFG = os.path.join(REPO, "3rdparty/Entity/Entityv2/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml")

print("[load] CropFormer detector ...", flush=True)
det = CropFormerDetector(CFG, device="cuda")

img_path = os.path.join(REPO, "data/replica/room0/results/frame000000.jpg")
img = cv2.imread(img_path)
print("[run] on", img_path, img.shape, flush=True)
inst = det(img)
print("[ok] instances:", inst.scores.shape[0], "masks:", inst.pred_masks.shape, flush=True)
