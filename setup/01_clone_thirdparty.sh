#!/usr/bin/env bash
# Clone the third-party repositories required by the pipeline.
# Run inside the WSL / Linux environment:
#   bash setup/01_clone_thirdparty.sh
# Notes: GitHub is unreliable in mainland China; we disable HTTP/2 (RPC
# framing errors) and fall back to the ghfast.top proxy. The script is
# resumable: already-cloned repos are skipped.
set -e

cd "$(dirname "$0")/../3rdparty" || exit 1

git config --global http.version HTTP/1.1
git config --global http.postBuffer 524288000

clone() {
    local url="$1" dir="$2" branch="${3:-}"
    if [ -d "$dir/.git" ]; then
        echo "[skip] $dir already exists"
        return 0
    fi
    if [ ! -d "$dir" ]; then
        echo "[clone] $url -> $dir"
        if ! timeout 1200 git clone -q "$url" "$dir" 2>/tmp/clone_err.log; then
            rm -rf "$dir"
            echo "[retry via proxy] $url"
            timeout 1200 git clone -q "https://ghfast.top/$url" "$dir"
        fi
    fi
    if [ -n "$branch" ] && [ -d "$dir/.git" ]; then
        echo "[checkout] $dir -> $branch"
        git -C "$dir" checkout -q "$branch"
    fi
}

clone https://github.com/qqlu/Entity.git Entity
clone https://github.com/baaivision/tokenize-anything.git tokenize-anything
clone https://github.com/facebookresearch/detectron2.git detectron2 v0.6
clone https://github.com/gradslam/gradslam.git gradslam conceptfusion
clone https://github.com/krrish94/chamferdist.git chamferdist
clone https://github.com/openai/CLIP.git CLIP

echo "ALL_CLONES_DONE"
ls -d */
