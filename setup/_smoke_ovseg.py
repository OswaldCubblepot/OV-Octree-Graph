import sys
import numpy as np
import torch

sys.path.insert(0, "./3rdparty/ovseg")
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from open_vocab_seg import add_ovseg_config
from open_vocab_seg.utils import VisualizationDemo

cfg = get_cfg()
add_deeplab_config(cfg)
add_ovseg_config(cfg)
cfg.merge_from_file("./3rdparty/ovseg/configs/ovseg_swinB_vitL_demo.yaml")
cfg.merge_from_list([
    "MODEL.WEIGHTS", "pretrained_weights/ovseg_swinbase_vitL14_ft_mpt.pth",
    "MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME", "pretrained_weights/openai_vitl14_from_ovseg.pt",
])
cfg.freeze()
demo = VisualizationDemo(cfg)

img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
mask = torch.zeros(480, 640, dtype=torch.float32, device="cuda")
mask[100:300, 100:400] = 1
feat = demo.run_on_regions(img, mask, "cuda")
print("OVSeg region feat:", tuple(feat.shape), "| VRAM:",
      round(torch.cuda.max_memory_allocated() / 1e9, 2), "GB")
