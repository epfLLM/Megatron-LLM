#! /bin/bash

GPUS_PER_NODE=1
NNODES=1
RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/pure-mlo-scratch/pagliard/data/wikitext-falcon/wiki-train_text_document
TENSORBOARD_PATH=/pure-mlo-scratch/alhernan/megatron-data/tensorboard/

## if using falcon7:
# CHECKPOINT_PATH=/pure-mlo-scratch/alhernan/megatron-data/checkpoints/falcon7b/
CHECKPOINT_PATH=/pure-mlo-scratch/alhernan/megatron-data/checkpoints/falcon7b-tp1-pp1/
NUM_LAYERS=32
HIDDEN_SIZE=4544
KV=1
NUM_HEADS=71
SIZE=7
MODEL=falcon
## if using falcon40:
# CHECKPOINT_PATH=/scratch/alhernan/megatron-data/checkpoints/falcon40b/
# NUM_LAYERS=60
# HIDDEN_SIZE=8192
# KV=8
# NUM_HEADS=128
# SIZE=40
# MODEL=falcon


DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6000"


torchrun $DISTRIBUTED_ARGS verify_correctness.py \
       --tensor_model_parallel_size 1 \
       --pipeline_model_parallel_size 1 \
       --num_layers $NUM_LAYERS \
       --hidden_size $HIDDEN_SIZE \
       --use_flash_attn \
       --no_bias_gelu_fusion \
       --num_attention_heads_kv $KV \
       --num_attention_heads $NUM_HEADS \
       --micro_batch_size 1 \
       --global_batch_size 1 \
       --seq_length 2048 \
       --max_position_embeddings 2048 \
       --train_iters 500000 \
       --lr_decay_iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data_path $DATA_PATH \
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
       --hidden_dropout 0.0 \
       --tensorboard_dir ${TENSORBOARD_PATH} \
       --position_embedding_type rotary \
       --use_multiquery_attn \
       --parallel_attn \
       --no_bias_dropout_fusion \
       --huggingface_cache /pure-mlo-scratch/alhernan/huggingface_cache/ \
       --huggingface_device "cuda:1" \
       --model_size $SIZE
