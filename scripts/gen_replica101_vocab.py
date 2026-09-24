"""Generate ONLY replica101_clip_l_14_vild.npy (the single vocab feature the
Replica pipeline needs), via OVSeg's CLIP adapter text encoder. Skips the
scannet20/200 and open_clip ViT-H-14 features that the full script also builds.

Run from repo root:
    python scripts/gen_replica101_vocab.py
"""
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from datasets.constants.replica.replica_constants import REPLICA_CLASSES  # noqa: E402

sys.path.insert(0, os.path.join(REPO_ROOT, "3rdparty", "ovseg"))
from detectron2.config import get_cfg  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from open_vocab_seg import add_ovseg_config  # noqa: E402
from open_vocab_seg.utils import VisualizationDemo  # noqa: E402

OUT = os.path.join(REPO_ROOT, "ovgraph", "evaluation", "voc_features",
                   "replica101_clip_l_14_vild.npy")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(os.path.join(REPO_ROOT, "3rdparty/ovseg/configs/ovseg_swinB_vitL_demo.yaml"))
    cfg.merge_from_list([
        "MODEL.WEIGHTS",
        os.path.join(REPO_ROOT, "pretrained_weights/ovseg_swinbase_vitL14_ft_mpt.pth"),
        "MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME",
        os.path.join(REPO_ROOT, "pretrained_weights/openai_vitl14_from_ovseg.pt"),
    ])
    cfg.freeze()

    print("[load] OVSeg model (Swin-B + CLIP ViT-L/14 adapter) ...", flush=True)
    demo = VisualizationDemo(cfg)
    print("[compute] text features for", len(REPLICA_CLASSES), "classes ...", flush=True)
    with torch.no_grad():
        feats = demo.predictor.model.clip_adapter.get_text_features(REPLICA_CLASSES)
    feats = feats.detach().cpu().numpy()
    np.save(OUT, feats)
    print(f"[saved] {OUT}  shape={feats.shape}", flush=True)


if __name__ == "__main__":
    main()
