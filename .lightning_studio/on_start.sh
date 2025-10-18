#!/bin/bash

# This script runs every time your Studio starts, from your home directory.

# Logs from previous runs can be found in ~/.lightning_studio/logs/

# List files under fast_load that need to load quickly on start (e.g. model checkpoints).
#
# ! fast_load
# <your file here>

# Add your startup commands below.
if ! command -v yq &> /dev/null; then
    echo "yq not found, installing..."
    sudo apt update && sudo apt install -y yq
fi

# Extract values from config.yaml
REGISTRY=$(yq '.common.REGISTRY' /teamspace/studios/this_studio/config.yaml)
USERNAME=$(yq '.common.USERNAME' /teamspace/studios/this_studio/config.yaml)
IMAGE_NAME=$(yq '.common.IMAGE_NAME' /teamspace/studios/this_studio/config.yaml)
TAG=$(yq '.common.TAG' /teamspace/studios/this_studio/config.yaml)

# Pull the Docker image
docker pull "${REGISTRY}/${USERNAME}/${IMAGE_NAME}:${TAG}"

# Check if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    docker run -it --rm --volume=$(pwd):/app/:rw -w /app --gpus all "${REGISTRY}/${USERNAME}/${IMAGE_NAME}:${TAG}" /bin/bash
else
    docker run -it --rm --volume=$(pwd):/app/:rw -w /app "${REGISTRY}/${USERNAME}/${IMAGE_NAME}:${TAG}" /bin/bash
fi