# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

# --------------------------------------------------------------------------- #
# The Entity/CropFormer mask2former fork (imported first by the detector) and
# this OVSeg fork both register overlapping detectron2 components
# (D2SwinTransformer, BasePixelDecoder, MaskFormerHead, ...) into the shared
# detectron2 registries, and detectron2's Registry._do_register asserts on the
# second registration. The detector is fully built *before* this package is
# imported (see ovgraph/scene/building.py), so making registration last-wins
# lets the OVSeg variants override the detector's — exactly what the extractor
# needs — with no effect on the already-built detector model.
# --------------------------------------------------------------------------- #
from fvcore.common.registry import Registry as _FvcoreRegistry


def _lenient_do_register(self, name, obj):
    self._obj_map[name] = obj  # last-wins override


_FvcoreRegistry._do_register = _lenient_do_register

from . import data
from . import modeling
from .config import add_ovseg_config

from .test_time_augmentation import SemanticSegmentorWithTTA
from .ovseg_model import OVSeg, OVSegDEMO
