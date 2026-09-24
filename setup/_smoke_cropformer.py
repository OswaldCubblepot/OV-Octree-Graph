import sys

import numpy as np
import torch

sys.path.insert(0, "./3rdparty/Entity/Entityv2/CropFormer/")
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from demo_cropformer.predictor import VisualizationDemo

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file(
    "./3rdparty/Entity/Entityv2/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml"
)
cfg.merge_from_list(["MODEL.WEIGHTS", "pretrained_weights/Mask2Former_hornet_3x_576d0b.pth"])
cfg.freeze()
demo = VisualizationDemo(cfg)

img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
out = demo.run_on_image(img)
inst = out["instances"]
print("CropFormer OK | masks:", tuple(inst.pred_masks.shape),
      "| scores:", len(inst.scores),
      "| VRAM:", round(torch.cuda.max_memory_allocated() / 1e9, 2), "GB")
