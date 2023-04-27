#!/bin/bash

# Finetune a BERT or pretrained ICT model using Google natural question data 
# Datasets can be downloaded from the following link:
# https://github.com/facebookresearch/DPR/blob/master/data/download_data.py

WORLD_SIZE=8

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT_PATH=<Specify path for the finetuned retriever model>

# Load either of the below
BERT_LOAD_PATH=<Path of BERT pretrained model>
PRETRAINED_CHECKPOINT=<Path of Pretrained ICT model>

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/main.py \
        --task RET-FINETUNE-NQ \
        --train_with_neg \
        --train_hard_neg 1 \
        --pretrained_checkpoint ${PRETRAINED_CHECKPOINT} \
        --num_layers 12 \
        --hidden_size 768 \
        --num_attention_heads 12 \
        --tensor_model_parallel_size 1 \
        --tokenizer_type BertWordPieceLowerCase \
        --train_data nq-train.json \
        --valid_data nq-dev.json \
        --save ${CHECKPOINT_PATH} \
        --load ${CHECKPOINT_PATH} \
        --vocab_file bert-vocab.txt \
        --bert_load ${BERT_LOAD_PATH} \
        --save_interval 5000 \
        --log_interval 10 \
        --eval_interval 20000 \
        --eval_iters 100 \
        --indexer_log_interval 1000 \
        --faiss_use_gpu \
        --DDP_impl torch \
        --fp16 \
        --retriever_report_topk_accuracies 1 5 10 20 100 \
        --seq_length 512 \
        --retriever_seq_length 256 \
        --max_position_embeddings 512 \
        --retriever_score_scaling \
        --epochs 80 \
        --micro_batch_size 8 \
        --eval_micro_batch_size 16 \
        --indexer_batch_size 128 \
        --lr 2e-5 \
        --lr_warmup_fraction 0.01 \
        --weight_decay 1e-1
