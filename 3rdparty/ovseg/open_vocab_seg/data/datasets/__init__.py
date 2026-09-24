# Copyright (c) Facebook, Inc. and its affiliates.
# The CropFormer (Entity) mask2former fork — imported first by the detector —
# registers the same training datasets (coco-stuff / voc / cc3m / ade20k /
# pascal-context). detectron2's `DatasetCatalog.register` asserts on duplicates,
# so importing these again raises `AssertionError: already registered`. These
# registrations are training-only and never touched by the inference pipeline,
# so make each import idempotent.
try:
    from . import register_coco_stuff, register_voc_seg
except AssertionError:
    pass
try:
    from . import register_cc3m
except AssertionError:
    pass
try:
    from . import register_ade20k_full
except AssertionError:
    pass
try:
    from . import register_pascal_context
except AssertionError:
    pass
