#!/bin/bash

TENSOR_MODEL_PARALLEL_SIZE=2

VOCAB_FILE=bert-vocab.txt
CHECKPOINT_PATH=checkpoints/bert_345m

WORLD_SIZE=$TENSOR_MODEL_PARALLEL_SIZE python tools/merge_mp_partitions.py \
                                --model_type BERT \
                                --tensor_model_parallel_size $TENSOR_MODEL_PARALLEL_SIZE \
                                --tokenizer_type BertWordPieceLowerCase \
                                --vocab_file $VOCAB_FILE \
                                --num_layers 24 \
                                --hidden_size 1024 \
                                --num_attention_heads 16 \
                                --seq_length 512 \
                                --max_position_embeddings 512 \
                                --load $CHECKPOINT_PATH
