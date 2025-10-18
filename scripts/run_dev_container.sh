#!/usr/bin/env bash
set -e

CONFIG=~/config.yaml

# Ensure yq is available
if ! command -v yq >/dev/null 2>&1; then
    echo "Installing yq..."
    sudo apt update -y && sudo apt install -y yq
fi

# Read Docker config
REGISTRY=$(yq '.common.REGISTRY' "$CONFIG")
USERNAME=$(yq '.common.USERNAME' "$CONFIG")
IMAGE_NAME=$(yq '.common.IMAGE_NAME' "$CONFIG")
TAG=$(yq '.common.TAG' "$CONFIG")
IMAGE="${REGISTRY}/${USERNAME}/${IMAGE_NAME}:${TAG}"

docker pull "$IMAGE"

echo "Launching dev container: $IMAGE"

# GPU detection
GPU_FLAG=""
if nvidia-smi >/dev/null 2>&1 && docker info 2>/dev/null | grep -q "Runtimes:.*nvidia"; then
    GPU_FLAG="--gpus all"
fi

# Run
exec docker run -it --rm \
    -v "$(pwd):/app:rw" \
    -w /app \
    $GPU_FLAG \
    "$IMAGE" /bin/bash
