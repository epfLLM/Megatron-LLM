#!/bin/bash

# Evaluate natural question test data given Wikipedia embeddings and pretrained
# ICT model or a finetuned model for Natural Question task

# Datasets can be downloaded from the following link:
# https://github.com/facebookresearch/DPR/blob/master/data/download_data.py

EVIDENCE_DATA_DIR=<Specify path of Wikipedia dataset>
EMBEDDING_PATH=<Specify path of the embeddings>
CHECKPOINT_PATH=<Specify path of pretrained ICT model or finetuned model>

QA_FILE=<Path of the natural question dev or test dataset>

python tasks/main.py \
    --task RETRIEVER-EVAL \
    --tokenizer_type BertWordPieceLowerCase \
    --num_layers 12 \
    --hidden_size 768 \
    --num_attention_heads 12 \
    --tensor_model_parallel_size 1 \
    --micro_batch_size 128 \
    --activations_checkpoint_method uniform \
    --seq_length 512 \
    --max_position_embeddings 512 \
    --load ${CHECKPOINT_PATH} \
    --evidence_data_path ${EVIDENCE_DATA_DIR} \
    --embedding_path ${EMBEDDING_PATH} \
    --retriever_seq_length 256 \
    --vocab_file bert-vocab.txt\
    --qa_data_test ${QA_FILE} \
    --faiss_use_gpu \
    --retriever_report_topk_accuracies 1 5 20 100 \
    --fp16 \
    --indexer_log_interval 1000 \
    --indexer_batch_size 128


