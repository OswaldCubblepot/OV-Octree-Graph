import os

from huggingface_hub import hf_hub_download

token = os.environ["HF_TOKEN"]
p = hf_hub_download(
    "qqlu1992/Adobe_EntitySeg",
    "CropFormer_model/Entity_Segmentation/Mask2Former_hornet_3x/Mask2Former_hornet_3x_576d0b.pth",
    token=token,
    repo_type="dataset",
    local_dir="/home/ctx27/OV-Octree-Graph/pretrained_weights",
)
print("DOWNLOADED:", p)
