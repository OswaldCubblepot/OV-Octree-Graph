"""
Generate the 4 vocab features the pipeline actually needs, faithful to the
official protocol of ovgraph/evaluation/comput_voc_feature.py:

  - *_clip_l_14_vild.npy  (replica101 / scannet20 / scannet200)
        -> OVSeg checkpoint's CLIP adapter text encoder (no openai download;
           azureedge is blocked here anyway)
  - scannet200_clip_h_14.npy
        -> open_clip ViT-H-14 laion2b_s32b_b79k (downloads from HF; set
           HF_ENDPOINT=https://hf-mirror.com in China)

The ViT-B/32 (Detic) features from the official script are not used by the
pipeline and are skipped (they would need the blocked openai CDN).

Usage (from repo root):
    HF_ENDPOINT=https://hf-mirror.com python scripts/generate_vocab_features_ovseg.py
"""

import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# `datasets` is an implicit namespace package at the repo root (no
# __init__.py); it only resolves when the root is on sys.path.
sys.path.insert(0, REPO_ROOT)

from datasets.constants.replica.replica_constants import REPLICA_CLASSES
from datasets.constants.scannet.scannet200_constants import CLASS_LABELS_20, CLASS_LABELS_200

VOCAB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "ovgraph", "evaluation", "voc_features")


def generate_h14(class_labels, out_path, device):
    """open_clip ViT-H-14, single prompt — identical to the official script."""
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-H-14")

    feats = []
    with torch.no_grad():
        for name in class_labels:
            tokenized = tokenizer(f"a picture of {name}").to(device)
            feat = model.encode_text(tokenized)
            feat /= feat.norm(dim=-1, keepdim=True)
            feats.append(feat.detach().cpu().numpy())
    np.save(out_path, np.concatenate(feats, axis=0))
    print(f"[saved] {out_path}")


def generate_l14vild(class_labels, out_path):
    """OVSeg checkpoint CLIP adapter — identical to the official script."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "3rdparty", "ovseg"))
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from open_vocab_seg import add_ovseg_config
    from open_vocab_seg.utils import VisualizationDemo

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(os.path.join(REPO_ROOT, "3rdparty/ovseg/configs/ovseg_swinB_vitL_demo.yaml"))
    cfg.merge_from_list(["MODEL.WEIGHTS",
                         os.path.join(REPO_ROOT, "pretrained_weights/ovseg_swinbase_vitL14_ft_mpt.pth"),
                         "MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME",
                         os.path.join(REPO_ROOT, "pretrained_weights/openai_vitl14_from_ovseg.pt")])
    cfg.freeze()

    demo = VisualizationDemo(cfg)
    feats = demo.predictor.model.clip_adapter.get_text_features(class_labels)
    np.save(out_path, feats.detach().cpu().numpy())
    print(f"[saved] {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(VOCAB_DIR, exist_ok=True)
    os.chdir(REPO_ROOT)

    targets_l14 = {
        "replica101": REPLICA_CLASSES,
        "scannet20": CLASS_LABELS_20,
        "scannet200": CLASS_LABELS_200,
    }
    for tag, labels in targets_l14.items():
        out_path = os.path.join(VOCAB_DIR, f"{tag}_clip_l_14_vild.npy")
        if os.path.isfile(out_path):
            print(f"[skip] {out_path} already exists")
            continue
        print(f"[compute] {tag} ({len(labels)} classes) via OVSeg clip_adapter")
        generate_l14vild(labels, out_path)

    out_path = os.path.join(VOCAB_DIR, "scannet200_clip_h_14.npy")
    if os.path.isfile(out_path):
        print(f"[skip] {out_path} already exists")
    else:
        print(f"[compute] scannet200 ({len(CLASS_LABELS_200)} classes) via open_clip ViT-H-14")
        generate_h14(CLASS_LABELS_200, out_path, args.device)

    print("\nDone. Files under", VOCAB_DIR)


if __name__ == "__main__":
    main()
