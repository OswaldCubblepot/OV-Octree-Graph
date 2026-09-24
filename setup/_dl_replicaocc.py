import os
import time

from huggingface_hub import snapshot_download

# hf-mirror rate-limits (429) with high concurrency; download scene by scene
# with 4 workers and retries. Already-downloaded files are reused (resume).
SCENES = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
local_dir = os.path.expanduser("~/OV-Octree-Graph/data/replica_occ")

for scene in SCENES:
    patterns = [f"Replica_OCC/sequences/{scene}/*"]
    for attempt in range(5):
        try:
            snapshot_download(
                "the-masses/ReplicaOcc",
                repo_type="dataset",
                allow_patterns=patterns,
                local_dir=local_dir,
                max_workers=4,
            )
            print(f"[done] {scene}")
            break
        except Exception as e:
            print(f"[retry {attempt+1}/5] {scene}: {type(e).__name__}: {str(e)[:120]}")
            time.sleep(30 * (attempt + 1))

print("REPLICA_OCC_DONE ->", local_dir)
