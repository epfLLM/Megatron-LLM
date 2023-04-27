#!/bin/bash

WORLD_SIZE=8

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

TASK="LAMBADA"

VALID_DATA=<lambada path>
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
CHECKPOINT=checkpoints/gpt2_345m


python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/main.py \
               --task $TASK \
               --valid_data $VALID_DATA \
               --tokenizer_type GPT2BPETokenizer \
               --strict_lambada \
               --vocab_file $VOCAB_FILE \
               --merge_file $MERGE_FILE \
               --load $CHECKPOINT \
               --tensor_model_parallel_size 1 \
               --num_layers 24 \
               --hidden_size 1024 \
               --num_attention_heads 16 \
               --batch_size 8 \
               --activations_checkpoint_method uniform \
               --seq_length 1024 \
               --max_position_embeddings 1024 \
               --log_interval 10 \
               --fp16 \
               --no_load_optim \
               --no_load_rng
