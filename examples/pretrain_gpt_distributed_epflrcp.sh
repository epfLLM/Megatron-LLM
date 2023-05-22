#! /bin/bash

# Runs the "345M" parameter model

if [[ $# -ne 1 ]]; then
	echo "Usage: $0 <node rank>"
	exit 1
fi

set -e 

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=gpu012.rcp.epfl.ch
MASTER_PORT=6000
NNODES=2
NODE_RANK=$1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/scratch/wikitext-megatron/wikitext-train_text_document
EXP_DIR=/scratch/$(whoami)/exp
CHECKPOINT_PATH=${EXP_DIR}/checkpoint
TENSORBOARD_PATH=${EXP_DIR}/tensorboard

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor_model_parallel_size 2 \
       --pipeline_model_parallel_size 2 \
       --num_layers 24 \
       --hidden_size 1024 \
       --num_attention_heads 16 \
       --micro_batch_size 4 \
       --global_batch_size 16 \
       --seq_length 1024 \
       --max_position_embeddings 1024 \
       --train_iters 500000 \
       --lr_decay_iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data_path $DATA_PATH \
       --vocab_file gpt2-vocab/gpt2-vocab.json \
       --merge_file gpt2-vcoab/gpt2-merges.txt \
       --data_impl mmap \
       --split 949,50,1 \
       --distributed_backend nccl \
       --lr 0.00015 \
       --lr_decay_style cosine \
       --min_lr 1.0e-5 \
       --weight_decay 1e-2 \
       --clip_grad 1.0 \
       --lr_warmup_fraction .01 \
       --log_interval 100 \
       --save_interval 10000 \
       --eval_interval 1000 \
       --eval_iters 10 \
       --use_bias \
       --tensorboard_dir ${TENSORBOARD_PATH} \
       --bf16
