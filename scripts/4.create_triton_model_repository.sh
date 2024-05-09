#!/bin/bash


mkdir -p  /DEPLOY_T5/triton-repos/T5-ner/
cp /DEPLOY_T5/v0.8.0/tensorrtllm_backend/all_models/inflight_batcher_llm/* /DEPLOY_T5/triton-repos/T5-ner/ -r


python3 /DEPLOY_T5/v0.8.0/tensorrtllm_backend/tools/fill_template.py -i /DEPLOY_T5/triton-repos/T5-ner/preprocessing/config.pbtxt tokenizer_dir:/DEPLOY_T5/models/T5-ner,tokenizer_type:t5,triton_max_batch_size:16,preprocessing_instance_count:1
python3 /DEPLOY_T5/v0.8.0/tensorrtllm_backend/tools/fill_template.py -i /DEPLOY_T5/triton-repos/T5-ner/postprocessing/config.pbtxt tokenizer_dir:/DEPLOY_T5/models/T5-ner,tokenizer_type:t5,triton_max_batch_size:16,postprocessing_instance_count:1
python3 /DEPLOY_T5/v0.8.0/tensorrtllm_backend/tools/fill_template.py -i /DEPLOY_T5/triton-repos/T5-ner/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:16,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False
python3 /DEPLOY_T5/v0.8.0/tensorrtllm_backend/tools/fill_template.py -i /DEPLOY_T5/triton-repos/T5-ner/ensemble/config.pbtxt triton_max_batch_size:16
python3 /DEPLOY_T5/v0.8.0/tensorrtllm_backend/tools/fill_template.py -i /DEPLOY_T5/triton-repos/T5-ner/tensorrt_llm/config.pbtxt triton_max_batch_size:16,decoupled_mode:False,max_beam_width:1,engine_dir:/DEPLOY_T5/tensorrt-models/T5-ner/v0.8.0/trt-engines/fp16/1-gpu/,max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.9,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_batching,max_queue_delay_microseconds:600


