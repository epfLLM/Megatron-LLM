#!/bin/bash

WORLD_SIZE=8

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

TRAIN_DATA="data/RACE/train/middle"
VALID_DATA="data/RACE/dev/middle \
            data/RACE/dev/high"
VOCAB_FILE=bert-vocab.txt
PRETRAINED_CHECKPOINT=checkpoints/bert_345m
CHECKPOINT_PATH=checkpoints/bert_345m_race

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/main.py \
               --task RACE \
               --seed 1234 \
               --train_data $TRAIN_DATA \
               --valid_data $VALID_DATA \
               --tokenizer_type BertWordPieceLowerCase \
               --vocab_file $VOCAB_FILE \
               --epochs 3 \
               --pretrained_checkpoint $PRETRAINED_CHECKPOINT \
               --tensor_model_parallel_size 1 \
               --num_layers 24 \
               --hidden_size 1024 \
               --num_attention_heads 16 \
               --micro_batch_size 4 \
               --activations_checkpoint_method uniform \
               --lr 1.0e-5 \
               --lr_decay_style linear \
               --lr_warmup_fraction 0.06 \
               --seq_length 512 \
               --max_position_embeddings 512 \
               --save_interval 100000 \
               --save $CHECKPOINT_PATH \
               --log_interval 10 \
               --eval_interval 100 \
               --eval_iters 50 \
               --weight_decay 1.0e-1 \
               --clip_grad 1.0 \
               --hidden_dropout 0.1 \
               --attention_dropout 0.1 \
               --fp16
