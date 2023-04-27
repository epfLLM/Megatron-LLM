#!/bin/bash

RANK=0
WORLD_SIZE=1
DATA_PATH=<Specify path and file prefix>
VOCAB_FILE=<Specify path to vocab.txt>
CHECKPOINT_PATH=<Specify path>

python pretrain_t5.py \
       --num_layers 12 \
       --hidden_size 768 \
       --num_attention_heads 12 \
       --kv_channels 64 \
       --ffn_hidden_size 3072 \
       --encoder_seq_length 512 \
       --decoder_seq_length 128 \
       --micro_batch_size 16 \
       --global_batch_size 16 \
       --max_position_embeddings 512 \
       --train_iters 1000000 \
       --lr_decay_iters 1000000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data_path $DATA_PATH \
       --vocab_file $VOCAB_FILE \
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
       --fp16 \
       --vocab_extra_ids 100
