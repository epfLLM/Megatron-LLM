#!/bin/bash

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=<Specify path and file prefix>
CHECKPOINT_PATH=<Specify path>

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_t5.py \
       --tensor-model-parallel-size 2 \
       --num_layers 12 \
       --hidden_size 768 \
       --num_attention_heads 12 \
       --kv_channels 64 \
       --ffn_hidden_size 3072 \
       --encoder_seq_length 512 \
       --decoder_seq_length 128 \
       --micro_batch_size 16 \
       --global_batch_size 128 \
       --max_position_embeddings 512 \
       --train_iters 1000000 \
       --lr_decay_iters 1000000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data_path $DATA_PATH \
       --vocab_file t5-vocab.txt \
       --data_impl mmap \
       --split 949,50,1 \
       --lr 0.0001 \
       --min_lr 0.00001 \
       --lr_decay_style linear \
       --lr_warmup_fraction .01 \
       --weight_decay 1e-2 \
       --clip_grad 1.0 \
       --log_interval 100 \
       --save_interval 10000 \
       --eval_interval 1000 \
       --eval_iters 10 \
       --fp16  \
       --vocab_extra_ids 100
