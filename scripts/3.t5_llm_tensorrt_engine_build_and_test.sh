#!/bin/bash

# Description: This script automates the installation and inference process for a Hugging Face model using TensorRT-LLM. Ensure that Git and Git LFS ('apt-get install git-lfs') are installed before running this script. Before running this script, run the following scripts sequentially: 1. install_git_and_lfs.sh 2. install_tensorrt_llm.sh

MODEL_NAME="T5-ner"
CHECKPOINT_PATH="/model/checkpoint-t5"
# Clone the Hugging Face model repository

# Convert the model checkpoint to TensorRT format
export MODEL_TYPE="t5"
export INFERENCE_PRECISION="bfloat16"
export TP_SIZE=1
export PP_SIZE=1
export WORLD_SIZE=1
export MAX_BEAM_WIDTH=1
export BATCH_SIZE=16
export MAX_OUTPUT_LENGTH=128
python3  /DEPLOY_T5/v0.8.0/tensorrtllm_backend/tensorrt_llm/examples/enc_dec/t5/convert.py \
        -i $CHECKPOINT_PATH \
        -o /DEPLOY_T5/tensorrt-models/${MODEL_NAME}/v0.8.0/trt-checkpoints/${INFERENCE_PRECISION}/1-gpu/ \
        --weight_data_type float32 \
        --dtype ${INFERENCE_PRECISION}

# Build TensorRT engine
python3 /DEPLOY_T5/v0.8.0/tensorrtllm_backend/tensorrt_llm/examples/enc_dec/build.py --model_type t5 \
                --weight_dir /DEPLOY_T5/tensorrt-models/$MODEL_NAME/v0.8.0/trt-checkpoints/${INFERENCE_PRECISION}/1-gpu/ \
                -o /DEPLOY_T5/tensorrt-models/$MODEL_NAME/v0.8.0/trt-engines/${INFERENCE_PRECISION}/1-gpu/ \
                --engine_name t5-small \
                --remove_input_padding \
                --use_bert_attention_plugin \
                --use_gpt_attention_plugin \
                --use_gemm_plugin \
                --dtype float32 \
                --max_beam_width ${MAX_BEAM_WIDTH}
# --gpt_attention_plugin is necessary in Enc-Dec.
# Try --gemm_plugin to prevent accuracy issue.

# Run inference with the TensorRT engine
# Inferencing w/ single GPU greedy search, compare results with HuggingFace FP32
python3 /DEPLOY_T5/v0.8.0/tensorrtllm_backend/tensorrt_llm/examples/enc_dec/run.py --engine_dir /DEPLOY_T5/tensorrt-models/$MODEL_NAME/v0.8.0/trt-engines/${INFERENCE_PRECISION}/1-gpu --engine_name ${MODEL_NAME} --model_name $CHECKPOINT_PATH --max_new_token=64 --num_beams=1 --compare_hf_fp32
