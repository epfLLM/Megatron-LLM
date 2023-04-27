#! /bin/bash

# Runs the "345M" parameter model

RANK=0
WORLD_SIZE=1

DATA_PATH=<Specify path and file prefix>_text_document
CHECKPOINT_PATH=<Specify path>


python pretrain_gpt.py \
       --num_layers 24 \
       --hidden_size 1024 \
       --num_attention_heads 16 \
       --micro_batch_size 4 \
       --global_batch_size 8 \
       --seq_length 1024 \
       --max_position_embeddings 1024 \
       --train_iters 500000 \
       --lr_decay_iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data_path $DATA_PATH \
       --vocab_file gpt2-vocab.json \
       --merge_file gpt2-merges.txt \
       --data_impl mmap \
       --split 949,50,1 \
       --distributed_backend nccl \
       --lr 0.00015 \
       --min_lr 1.0e-5 \
       --lr_decay_style cosine \
       --weight_decay 1e-2 \
       --clip_grad 1.0 \
       --lr_warmup_fraction .01 \
       --activations_checkpoint_method uniform \
       --log_interval 100 \
       --save_interval 10000 \
       --eval_interval 1000 \
       --eval_iters 10 \
       --fp16
