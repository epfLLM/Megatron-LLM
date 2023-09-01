#!/bin/bash

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# NUM_IN_SHARDS=4
# NUM_OUT_SHARDS=8
# INPUT_DIR=/pure-mlo-scratch/alhernan/megatron-data/checkpoints/llama2-7b-tp2-pp2
# UNSHARDED_DIR=/pure-mlo-scratch/alhernan/megatron-data/checkpoints/llama2-7b-tp1-pp1_unsharded
# OUTPUT_DIR=/pure-mlo-scratch/llama2/mega2HF_1tp_8shard
# MEGATRON_PATH=/pure-mlo-scratch/sfan/Megatron-LLM/

NUM_IN_SHARDS="#NUM of shards of megatron checkpoints"
NUM_OUT_SHARDS="#NUM of shards of converted huggingface checkpoints"
INPUT_DIR="path to the megatron checkpoints directory"
UNSHARDED_DIR=/tmp_unsharded
OUTPUT_DIR="path to the converted HF checkpoints"
MEGATRON_PATH="path to Megatron-LLM directory (root directory of the repo)"

cd $MEGATRON_PATH

if [ $NUM_IN_SHARDS -gt 1 ]; then
    python3 tools/checkpoint_util.py \
        --target_tensor_parallel_size 1 \
        --target_pipeline_parallel_size 1 \
        --load_dir $INPUT_DIR \
        --save_dir $UNSHARDED_DIR \
        --megatron_path $MEGATRON_PATH \
        --model_type llama2 \
        --true_vocab_size 32000 \
        --bf16
    python3 weights2megatron/megatron2hf.py \
        --input_dir $UNSHARDED_DIR \
        --output_dir $OUTPUT_DIR --num_output_shards $NUM_OUT_SHARDS
    rm -r $UNSHARDED_DIR
else
    python3 weights2megatron/megatron2hf.py \
        --input_dir $INPUT_DIR \
        --output_dir $OUTPUT_DIR --num_output_shards $NUM_OUT_SHARDS
fi
