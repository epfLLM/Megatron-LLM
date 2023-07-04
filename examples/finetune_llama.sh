#! /bin/bash

RANK=0
WORLD_SIZE=1

DATA_PATH=/scratch/wikitext-megatron/wikitext-train_text_document
CHECKPOINT_PATH=/scratch/alhernan/megatron-data/checkpoints/llama_checkpoint_test
TENSORBOARD_PATH=/scratch/alhernan/megatron-data/tensorboard/


DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6000"


torchrun $DISTRIBUTED_ARGS finetune_llama.py \
       --tensor_model_parallel_size 1 \
       --pipeline_model_parallel_size 1 \
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
       --vocab_file /scratch/alhernan/gpt2-vocab.json \
       --merge_file /scratch/alhernan/gpt2-merges.txt \
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
       --use_bias \
       --tensorboard_dir ${TENSORBOARD_PATH} \
       --use_rms_norm \
       --glu_activation swiglu \
       --position_embedding_type rotary \
       --no_bias_gelu_fusion \
       --bf16
