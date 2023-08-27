#!/bin/bash

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
NUM_IN_SHARDS=4
NUM_OUT_SHARDS=7
INPUT_DIR=/pure-mlo-scratch/akoepf/checkpoints/codellama-13b-tp4-pp2-oa_codellama
UNSHARDED_DIR=/pure-mlo-scratch/akoepf/checkpoints/tmp-unsharded-oa_codellama
OUTPUT_DIR=/pure-mlo-scratch/akoepf/checkpoints/codellama-13b-oasst-sft10-hf
MEGATRON_PATH=/pure-mlo-scratch/akoepf/code/epfl-megatron

# llama/llama2
#MODEL_TYPE=llama2
#VOCAB_SIZE=32007

# codellama
VOCAB_SIZE=32023    # codellama
MODEL_TYPE=codellama

# NUM_IN_SHARDS="#NUM of shards of megatron checkpoints"
# NUM_OUT_SHARDS="#NUM of shards of converted huggingface checkpoints"
# INPUT_DIR="path to the megatron checkpoints directory"
# UNSHARDED_DIR=/tmp_unsharded
# OUTPUT_DIR="path to the converted HF checkpoints"
# MEGATRON_PATH="path to Megatron-LLM directory (root directory of the repo)"

cd $MEGATRON_PATH

if [ $NUM_IN_SHARDS -gt 1 ]; then
    python3 tools/checkpoint_util.py \
        --target_tensor_parallel_size 1 \
        --target_pipeline_parallel_size 1 \
        --load_dir $INPUT_DIR \
        --save_dir $UNSHARDED_DIR \
        --megatron_path $MEGATRON_PATH \
        --model_type $MODEL_TYPE \
        --true_vocab_size $VOCAB_SIZE \
        --bf16
    python3 weights2megatron/megatron2hf.py \
        --input_dir $UNSHARDED_DIR \
        --output_dir $OUTPUT_DIR --num_output_shards $NUM_OUT_SHARDS --model $MODEL_TYPE \
        --vocab_file /pure-mlo-scratch/akoepf/codellama/CodeLlama-13b/tokenizer.model \
        --vocab_extra_ids_list "<|im_start|>,<|im_end|>"
    rm -r $UNSHARDED_DIR
else
    python3 weights2megatron/megatron2hf.py \
        --input_dir $INPUT_DIR \
        --output_dir $OUTPUT_DIR --num_output_shards $NUM_OUT_SHARDS --model $MODEL_TYPE \
        --vocab_file /pure-mlo-scratch/akoepf/codellama/CodeLlama-13b/tokenizer.model \
        --vocab_extra_ids_list "<|im_start|>,<|im_end|>"
fi