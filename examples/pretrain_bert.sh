#!/bin/bash

RANK=0
WORLD_SIZE=1
DATA_PATH=<Specify path and file prefix>_text_sentence
CHECKPOINT_PATH=<Specify path>

python pretrain_bert.py \
       --num_layers 24 \
       --hidden_size 1024 \
       --num_attention_heads 16 \
       --micro_batch_size 4 \
       --global_batch_size 8 \
       --seq_length 512 \
       --max_position_embeddings 512 \
       --train_iters 2000000 \
       --lr_decay_iters 990000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data_path $DATA_PATH \
       --vocab_file bert-vocab.txt \
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
       --fp16
