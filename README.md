# Triton TensorRT LLM Model Preparation and Deployment

## Overview

This repository provides scripts and instructions for preparing and deploying Large Language Models (LLMs) using the Triton Inference Server with TensorRT-LLM. With a streamlined workflow, you can efficiently deploy LLM models in production environments.

## Getting Started

To get started, follow these steps:

1. Cd to main directory:
   ```
   cd DEPLOY_T5
   ```

2. Build a Docker image from the DockerFile:
   ```
   docker build -t triton-llm-deployment .
   ```

3. Create a container and run the provided bash files inside the container:
   ```
   docker run -it --name triton-llm-container triton-llm-deployment
   ```

4. Within the container, execute the bash files to prepare and deploy your LLM model:
   ```
   bash scripts/1.install_git_and_lfs.sh
   bash scripts/2.install_tensorrt_llm.sh
   bash scripts/3.trendyol_llm_tensorrt_engine_build_and_test.sh
   bash scripts/4.create_triton_model_repository.sh
   bash scripts/5.run triton.sh
   bash scripts/6.call_triton_model_curl_example.sh
   ```

## Repository Structure

- `1.install_git_and_lfs.sh`: Installs Git and Git LFS if not already installed.
- `2.install_tensorrt_llm.sh`: Fetches the tensorrtllm_backend project and its submodules, ensuring compatibility with Triton.
- `3.trendyol_llm_tensorrt_engine_build_and_test.sh`: Downloads and converts the Trendyol/Trendyol-LLM-7b-chat-v1.0 model from Hugging Face into a TensorRT-LLM checkpoint, and then creates a TensorRT-LLM engine from this checkpoint for testing.
- `4.create_triton_model_repository.sh`: Creates the Triton model repository and copies model definition files into it.
- `5.run_triton.sh`: Starts the Triton server to serve the deployed LLM model.
- `6.call_triton_model_curl_example.sh`: Provides an example of how to make requests to the deployed LLM model using cURL.
## Resources

- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT/tree/master/llm](https://github.com/NVIDIA/TensorRT-LLM))