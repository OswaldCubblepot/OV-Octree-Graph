"""
Generate the class-name (vocabulary) text features needed for evaluation.

The repository does not ship the following binary files (they are produced
by ``ovgraph/evaluation/comput_voc_feature.py`` in the official release):

    ovgraph/evaluation/voc_features/replica101_clip_l_14_vild.npy   (Replica eval)
    ovgraph/evaluation/voc_features/scannet20_clip_l_14_vild.npy    (ScanNet semantic eval)
    ovgraph/evaluation/voc_features/scannet200_clip_l_14_vild.npy   (ScanNet instance / retrieval eval)
    ovgraph/evaluation/voc_features/scannet200_clip_h_14.npy        (ScanNet instance eval)

``*_clip_l_14_vild`` uses the same text encoder protocol as OVSeg's CLIP
adapter (OpenAI CLIP ViT-L/14 + ImageNet prompt ensemble), and
``*_clip_h_14`` uses open_clip ViT-H-14, so this script reproduces the
official features without downloading the OVSeg model weights.

Usage:
    python scripts/generate_vocab_features.py --models l14vild,h14
"""

import argparse
import os

import numpy as np
import torch

from datasets.constants.replica.replica_constants import REPLICA_CLASSES
from datasets.constants.scannet.scannet200_constants import CLASS_LABELS_20, CLASS_LABELS_200

VOCAB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "ovgraph", "evaluation", "voc_features")

# ImageNet templates used by OVSeg's text encoder (a subset that covers the
# official prompts; identical semantics to the ensemble in the OVSeg adapter)
IMAGENET_PROMPT = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a rendition of a {}.",
]


def compute_l14vild(class_labels, device):
    """OpenAI CLIP ViT-L/14 + ImageNet prompt ensemble (OVSeg protocol)."""
    import clip
    model, _ = clip.load("ViT-L/14", device=device)
    model.eval()

    text_features_bucket = []
    with torch.no_grad():
        for template in IMAGENET_PROMPT:
            tokens = [clip.tokenize(template.format(name)) for name in class_labels]
            text_inputs = torch.cat(tokens).to(device)
            feats = model.encode_text(text_inputs)
            feats /= feats.norm(dim=-1, keepdim=True)
            text_features_bucket.append(feats)

        text_features = torch.stack(text_features_bucket).mean(dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.detach().cpu().numpy()


def compute_h14(class_labels, device):
    """open_clip ViT-H-14 with the single prompt used by the official script."""
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
    return np.concatenate(feats, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=str, default="l14vild,h14",
                        help="comma separated: l14vild,h14 (default: both)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    os.makedirs(VOCAB_DIR, exist_ok=True)

    targets = {
        "l14vild": {
            "replica101": REPLICA_CLASSES,
            "scannet20": CLASS_LABELS_20,
            "scannet200": CLASS_LABELS_200,
        },
        "h14": {
            "replica101": REPLICA_CLASSES,
            "scannet20": CLASS_LABELS_20,
            "scannet200": CLASS_LABELS_200,
        },
    }

    for model in models:
        assert model in targets, f"unknown model {model}, choose from {list(targets)}"
        for tag, labels in targets[model].items():
            out_path = os.path.join(VOCAB_DIR, f"{tag}_clip_{model.replace('l14vild', 'l_14_vild').replace('h14', 'h_14')}.npy")
            if os.path.isfile(out_path):
                print(f"[skip] {out_path} already exists")
                continue
            print(f"[compute] {tag} ({len(labels)} classes) with {model} on {args.device}")
            feats = compute_l14vild(labels, args.device) if model == "l14vild" else compute_h14(labels, args.device)
            np.save(out_path, feats)
            print(f"[saved] {out_path}  shape={feats.shape}")

    print(f"\nDone. Files written under {VOCAB_DIR}")


if __name__ == "__main__":
    main()
