"""Download ReplicaOcc sequences with aria2, bypassing the rate-limited API.

- Probe max frame index per scene via HEAD requests (color/N.jpg).
- Generate a per-file URL list (only for missing files) for aria2c.
- Run aria2c with 4 concurrent downloads and resume support.
"""
import os
import subprocess
import urllib.request

COMMIT = "25fd9c968f916a74d55260207d701b3c0203e8b4"
BASE = f"https://hf-mirror.com/datasets/the-masses/ReplicaOcc/resolve/{COMMIT}/Replica_OCC/sequences"
SCENES = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
LOCAL = os.path.expanduser("~/OV-Octree-Graph/data/replica_occ/Replica_OCC/sequences")


def head_ok(url):
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except Exception as e:
        code = getattr(e, "code", None)
        if code == 429:
            raise e
        return False


def probe_max(scene):
    """Find the largest N with color/N.jpg present."""
    lo, hi = 0, 2048
    last_ok = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        ok = head_ok(f"{BASE}/{scene}/color/{mid}.jpg")
        if ok:
            last_ok = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return last_ok


def main():
    aria_input = os.path.expanduser("~/OV-Octree-Graph/setup/_replicaocc_urls.txt")
    with open(aria_input, "w") as f:
        for scene in SCENES:
            d = os.path.join(LOCAL, scene)
            try:
                max_n = probe_max(scene)
            except Exception as e:
                print(f"[probe retry needed] {scene}: {e}")
                max_n = None
            if max_n is None or max_n < 0:
                print(f"[skip] {scene} (probe failed)")
                continue
            print(f"[scene] {scene}: 0..{max_n} ({max_n + 1} frames)")
            for kind, ext in (("color", "jpg"), ("depth", "png"), ("pose", "txt")):
                kind_dir = os.path.join(d, kind)
                os.makedirs(kind_dir, exist_ok=True)
                for n in range(max_n + 1):
                    target = os.path.join(kind_dir, f"{n}.{ext}")
                    if os.path.exists(target) and os.path.getsize(target) > 0:
                        continue
                    f.write(f"{BASE}/{scene}/{kind}/{n}.{ext}\n")
                    f.write(f"  dir={kind_dir}\n")
                    f.write(f"  out={n}.{ext}\n")

    print("running aria2c ...")
    subprocess.run(
        ["aria2c", "-i", aria_input, "-j", "4", "-x", "1", "-s", "1",
         "--continue", "--max-tries", "6", "--retry-wait", "15",
         "--timeout", "60", "--connect-timeout", "30",
         "--allow-overwrite=false", "--auto-file-renaming=false",
         "--summary-interval", "30", "-q"],
        check=False,
    )

    print("--- verify ---")
    for scene in SCENES:
        d = os.path.join(LOCAL, scene)
        c = len(os.listdir(os.path.join(d, "color"))) if os.path.isdir(os.path.join(d, "color")) else 0
        dp = len(os.listdir(os.path.join(d, "depth"))) if os.path.isdir(os.path.join(d, "depth")) else 0
        p = len(os.listdir(os.path.join(d, "pose"))) if os.path.isdir(os.path.join(d, "pose")) else 0
        print(f"{scene}: color={c} depth={dp} pose={p} ok={c == dp == p and c > 0}")


if __name__ == "__main__":
    main()
