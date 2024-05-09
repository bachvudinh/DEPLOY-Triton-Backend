#!/bin/bash
CUDA_VERSION="12.1.0"
echo "Start a basic docker and install tensorrt_llm "
docker run --rm --runtime=nvidia --gpus all --entrypoint /bin/bash -it nvidia/cuda:$CUDA_VERSION-devel-ubuntu22.04

# Install dependencies, TensorRT-LLM requires Python 3.10
apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev
# Install the latest stable version (corresponding to the release branch) of TensorRT-LLM.
pip3 install tensorrt_llm -U --extra-index-url https://pypi.nvidia.com
# Check installation
python3 -c "import tensorrt_llm"

git lfs install
git clone -b v0.8.0 https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
make -C docker release_build