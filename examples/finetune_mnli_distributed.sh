#!/bin/bash

WORLD_SIZE=8

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

TRAIN_DATA="data/glue_data/MNLI/train.tsv"
VALID_DATA="data/glue_data/MNLI/dev_matched.tsv \
            data/glue_data/MNLI/dev_mismatched.tsv"
PRETRAINED_CHECKPOINT=checkpoints/bert_345m
VOCAB_FILE=bert-vocab.txt
CHECKPOINT_PATH=checkpoints/bert_345m_mnli

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/main.py \
               --task MNLI \
               --seed 1234 \
               --train_data $TRAIN_DATA \
               --valid_data $VALID_DATA \
               --tokenizer_type BertWordPieceLowerCase \
               --vocab_file $VOCAB_FILE \
               --epochs 5 \
               --pretrained_checkpoint $PRETRAINED_CHECKPOINT \
               --tensor_model_parallel_size 1 \
               --num_layers 24 \
               --hidden_size 1024 \
               --num_attention_heads 16 \
               --micro_batch_size 8 \
               --activations_checkpoint_method uniform \
               --lr 5.0e-5 \
               --lr_decay_style linear \
               --lr_warmup_fraction 0.065 \
               --seq_length 512 \
               --max_position_embeddings 512 \
               --save_interval 500000 \
               --save $CHECKPOINT_PATH \
               --log_interval 10 \
               --eval_interval 100 \
               --eval_iters 50 \
               --weight_decay 1.0e-1 \
               --fp16
