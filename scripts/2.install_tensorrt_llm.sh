#!/bin/bash

# Description: This script automates the installation process for TensorRT-LLM. Prior to running this script, ensure that Git and Git LFS ('apt-get install git-lfs') are installed.

# Step 1: Defining folder path and version
echo "Step 1: Defining folder path and version"
TENSORRT_BACKEND_LLM_VERSION=v0.8.0
TENSORRT_DIR="/DEPLOY_T5/$TENSORRT_BACKEND_LLM_VERSION"

# Step 2: Enter the installation folder and clone
echo "Step 2: Enter the installation folder and clone"
[ ! -d "$TENSORRT_DIR" ] && mkdir -p "$TENSORRT_DIR"
cd "$TENSORRT_DIR" || { echo "Failed to change directory to $TENSORRT_DIR"; exit 1; }
git clone -b "$TENSORRT_BACKEND_LLM_VERSION" https://github.com/triton-inference-server/tensorrtllm_backend.git --progress --verbose || { echo "Failed to clone repository"; exit 1; }
cd "$TENSORRT_DIR"/tensorrtllm_backend || { echo "Failed to change directory to $TENSORRT_DIR/tensorrtllm_backend"; exit 1; }
git submodule update --init --recursive || { echo "Failed to update submodules"; exit 1; }
git lfs install || { echo "Failed to install Git LFS"; exit 1; }
git lfs pull || { echo "Failed to pull Git LFS files"; exit 1; }
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev
# Step 3: Enter the backend folder and Install backend related dependencies
echo "Step 3: Enter the backend folder and Install backend related dependencies"
cd "$TENSORRT_DIR"/tensorrtllm_backend || { echo "Failed to change directory to $TENSORRT_DIR/tensorrtllm_backend"; exit 1; }
apt-get update && apt-get install -y --no-install-recommends rapidjson-dev python-is-python3 || { echo "Failed to install dependencies"; exit 1; }
pip3 install -r requirements.txt --extra-index-url https://pypi.ngc.nvidia.com || { echo "Failed to install Python dependencies"; exit 1; }

# Step 4: Install tensorrt-llm library
echo "Step 4: Install tensorrt-llm library"
pip install tensorrt_llm=="$TENSORRT_BACKEND_LLM_VERSION" -U --pre --extra-index-url https://pypi.nvidia.com || { echo "Failed to install tensorrt-llm library"; exit 1; }
# (cd /DEPLOY_T5/v0.8.0/tensorrtllm_backend/tensorrt_llm/  &&
#     bash docker/common/install_cmake.sh &&
#     export PATH=/usr/local/cmake/bin:$PATH &&
#     python3 ./scripts/build_wheel.py --trt_root="/usr/local/tensorrt" &&
#     pip3 install ./build/tensorrt_llm*.whl)