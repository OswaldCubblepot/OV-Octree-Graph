"""Convert ReplicaOcc layout to the pipeline's nice-slam layout.

ReplicaOcc:  sequences/<scene>/color/N.jpg  depth/N.png  pose/N.txt
Pipeline:    <data_root>/<scene>/results/frame%06d.jpg  depth%06d.png
             <data_root>/<scene>/traj.txt  (one 4x4 matrix per line)

Uses hardlinks (no copy) and verifies counts + depth PNG bit depth.
"""
import os

import numpy as np
from PIL import Image

SRC = os.path.expanduser("~/OV-Octree-Graph/data/replica_occ/Replica_OCC/sequences")
DST = os.path.expanduser("~/OV-Octree-Graph/data/replica")
SCENES = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]


def main():
    for scene in SCENES:
        s = os.path.join(SRC, scene)
        if not os.path.isdir(s):
            print(f"[warn] missing {s}")
            continue
        d = os.path.join(DST, scene)
        res = os.path.join(d, "results")
        os.makedirs(res, exist_ok=True)
        print(f"[convert] {scene}")

        colors = sorted(f for f in os.listdir(os.path.join(s, "color")) if f.endswith(".jpg"))
        depths = sorted(f for f in os.listdir(os.path.join(s, "depth")) if f.endswith(".png"))
        poses = sorted(f for f in os.listdir(os.path.join(s, "pose")) if f.endswith(".txt"))

        for f in colors:
            n = int(f[:-4])
            os.link(os.path.join(s, "color", f), os.path.join(res, f"frame{n:06d}.jpg"))
        for f in depths:
            n = int(f[:-4])
            os.link(os.path.join(s, "depth", f), os.path.join(res, f"depth{n:06d}.png"))

        with open(os.path.join(d, "traj.txt"), "w") as out:
            for f in poses:
                with open(os.path.join(s, "pose", f)) as p:
                    out.write(p.read().strip() + "\n")

        # sanity checks
        im = Image.open(os.path.join(res, "depth000000.png"))
        arr = np.array(im)
        n_frames = len(os.listdir(res)) // 2
        n_pose_lines = sum(1 for _ in open(os.path.join(d, "traj.txt")))
        print(f"    {scene}: frames={n_frames} pose_lines={n_pose_lines} "
              f"depth_mode={im.mode} depth_max={arr.max()} "
              f"OK={n_frames == len(poses) == len(colors)}")


if __name__ == "__main__":
    main()
