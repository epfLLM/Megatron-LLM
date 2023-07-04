#! /bin/bash

if [[ $# -ne 1 ]]; then
	echo "Usage: $0 <node rank>"
	exit 1
fi

GPUS_PER_NODE=8
NNODES=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
RANK=$1

DATA_PATH=/scratch/wikitext-megatron/wikitext-train_text_document
TENSORBOARD_PATH=/scratch/alhernan/megatron-data/tensorboard/


## if random initialization:
CHECKPOINT_PATH=/scratch/alhernan/megatron-data/checkpoints/falcon-random/
NUM_LAYERS=8
HIDDEN_SIZE=8192
KV=8
NUM_HEADS=128
## if using falcon7:
# CHECKPOINT_PATH=/scratch/alhernan/megatron-data/checkpoints/falcon7b/
# NUM_LAYERS=32
# HIDDEN_SIZE=4544
# KV=1
# NUM_HEADS=71
## if using falcon40:
# CHECKPOINT_PATH=/scratch/alhernan/megatron-data/checkpoints/falcon40b/
# NUM_LAYERS=60
# HIDDEN_SIZE=8192
# KV=8
# NUM_HEADS=128


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $RANK --master_addr localhost --master_port 6000"


# torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
torchrun $DISTRIBUTED_ARGS finetune_falcon.py \
       --tensor_model_parallel_size 2 \
       --pipeline_model_parallel_size 4 \
       --num_layers $NUM_LAYERS \
       --hidden_size $HIDDEN_SIZE \
       --use_flash_attn \
       --no_bias_gelu_fusion \
       --num_attention_heads_kv $KV \
       --num_attention_heads $NUM_HEADS \
       --micro_batch_size 1 \
       --global_batch_size 8 \
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
       --save_interval 200 \
       --eval_interval 1000 \
       --eval_iters 10 \
       --hidden_dropout 0.0 \
       --tensorboard_dir ${TENSORBOARD_PATH} \
       --position_embedding_type rotary \
       --use_multiquery_attn \
       --parallel_attn \
       --no_bias_dropout_fusion \
       --sequence_parallel \
       --bf16
