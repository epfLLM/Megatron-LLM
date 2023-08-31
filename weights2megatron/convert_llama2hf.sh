#!/bin/bash

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
NUM_SHARDS=8

python3 /pure-mlo-scratch/sfan/model-parallel-trainer/llama2megatron/convert_llama2hf.py \
    --input_dir /pure-mlo-scratch/llama/ --model_size 7B --output_dir /pure-mlo-scratch/llama/converted_HF_7B_8shard --num_output_shards $NUM_SHARDS
