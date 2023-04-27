#!/bin/bash


# SLURM options.
export SLURM_PARTITION=<slurm partition, used to feed -p option in slurm>
export SLURM_ACCOUNT=<slurm account, used to feed -A option in slurm>


# Source code.
export MEGATRON_CODE_DIR=<megatron source code directory>


# This variable is used to mount the relevant part of the filesystem
# inside the docker container. Note that the `MEGATRON_CODE_DIR` and the
# launch directory already get mounted; this variable should be used to
# mount the directories that contain the data and tokenizer files.
export DOCKER_MOUNT_DIR=<megatron dataset and bpe tokenizer vocab path>


# Data and tokenizer files.
MEGATRON_DATA=<path to megatron processed data>
BPE_VOCAB_FILE=<path to bpe vocab file>
BPE_MERGE_FILE=<path to bpe merges file>


# Megatron input parameters.
# `MEGATRON_EXTRA_PARAMS` can be used to provide any extra parameters
# that are not listed here. 
export MEGATRON_PARAMS=" ${MEGATRON_EXTRA_PARAMS} \
	--tensor_model_parallel_size ${TP} \
	--pipeline_model_parallel_size ${PP} \
	--micro_batch_size ${MBS} \
	--global_batch_size ${GBS} \
  --num_layers ${NLS} \
  --hidden_size ${HS} \
  --num_attention_heads ${NAH} \
	--DDP_impl ${DDP} \
	--data_path ${MEGATRON_DATA} \
	--vocab_file ${BPE_VOCAB_FILE} \
	--merge_file ${BPE_MERGE_FILE} \
  --log_interval 5 \
  --seq_length 2048 \
  --max_position_embeddings 2048 \
  --train_iters 500 \
  --lr_decay_iters 320 \
  --lr 0.0001 \
	--min_lr 0.00001 \
  --lr_decay_style cosine \
  --lr_warmup_fraction 0.01 \
  --split 969,30,1 \
  --eval_iters 100 \
  --eval_interval 1000 \
  --clip_grad 1.0 \
  --fp16 \
	--loss_scale 8192 "


