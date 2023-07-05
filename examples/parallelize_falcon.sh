#! /bin/bash

TENSOR_PARALLELISM=1
PIPELINE_PARALLELISM=6
MODEL_SIZE=40  # set to 40 for falcon 40B
CHECKPOINT_PATH=/scratch/alhernan/megatron-data/checkpoints/falcon${MODEL_SIZE}b/
OUTPUT_PATH=/scratch/alhernan/megatron-data/checkpoints/falcon7${MODEL_SIZE}b-tp$TENSOR_P-pp$PIPELINE_P/


python tools/checkpoint_util.py \
       --target_tensor_parallel_size $TENSOR_PARALLELISM \
       --target_pipeline_parallel_size $PIPELINE_PARALLELISM \
       --save_dir $OUTPUT_PATH \
       --load_dir $CHECKPOINT_PATH \
       --model_type falcon
