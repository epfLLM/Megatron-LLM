#! /bin/bash

# Runs the "217M" parameter biencoder model for ICT retriever

RANK=0
WORLD_SIZE=1

PRETRAINED_BERT_PATH=<Specify path of pretrained BERT model>
TEXT_DATA_PATH=<Specify path and file prefix of the text data>
TITLE_DATA_PATH=<Specify path and file prefix od the titles>
CHECKPOINT_PATH=<Specify path>


python pretrain_ict.py \
        --num_layers 12 \
        --hidden_size 768 \
        --num_attention_heads 12 \
        --tensor_model_parallel_size 1 \
        --micro_batch_size 32 \
        --seq_length 256 \
        --max_position_embeddings 512 \
        --train_iters 100000 \
        --vocab_file bert-vocab.txt \
        --tokenizer_type BertWordPieceLowerCase \
        --DDP_impl torch \
        --bert_load ${PRETRAINED_BERT_PATH} \
        --log_interval 100 \
        --eval_interval 1000 \
        --eval_iters 10 \
        --retriever_report_topk_accuracies 1 5 10 20 100 \
        --retriever_score_scaling \
        --load $CHECKPOINT_PATH \
        --save $CHECKPOINT_PATH \
        --data_path ${TEXT_DATA_PATH} \
        --titles_data_path ${TITLE_DATA_PATH} \
        --lr 0.0001 \
        --lr_decay_style linear \
        --weight_decay 1e-2 \
        --clip_grad 1.0 \
        --lr_warmup_fraction 0.01 \
        --save_interval 4000 \
        --exit_interval 8000 \
        --query_in_block_prob 0.1 \
        --fp16
