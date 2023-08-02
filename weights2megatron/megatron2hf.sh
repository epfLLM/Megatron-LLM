#!/bin/bash

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
NUM_IN_SHARDS=4
NUM_OUT_SHARDS=8

# python3 /pure-mlo-scratch/sfan/model-parallel-trainer/llama2megatron/convert_llama2hf.py \
#     --input_dir /pure-mlo-scratch/llama/ --model_size 7B --output_dir /pure-mlo-scratch/llama/converted_HF_7B_8shard --num_output_shards $NUM_SHARDS


python3 /pure-mlo-scratch/sfan/model-parallel-trainer/llama2megatron/megatron2hf.py \
    --input_dir /pure-mlo-scratch/alhernan/megatron-data/checkpoints/llama2-7b-tp4-pp1-optim --model_size 7 --output_dir /pure-mlo-scratch/llama2/mega2HF_4tp_8shard --num_input_shards $NUM_IN_SHARDS --num_output_shards $NUM_OUT_SHARDS

