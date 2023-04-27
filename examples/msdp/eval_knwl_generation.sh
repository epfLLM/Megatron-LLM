#!/bin/bash

#########################
# Evaluate the F1 scores.
#########################

WORLD_SIZE=1
DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"
                  
MODEL_GEN_PATH=<PATH_OF_THE_KNOWLEDGE_GENERATION> \ 
        (e.g., /testseen_knowledge_generations.txt)
GROUND_TRUTH_PATH=<PATH_OF_THE_GROUND_TRUTH_KNOWLEDGE> \ 
        (e.g., /testseen_knowledge_reference.txt)

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/msdp/main.py \
        --num_layers 24 \
        --hidden_size 1024 \
        --num_attention_heads 16 \
        --seq_length 2048 \
        --max_position_embeddings 2048 \
        --micro_batch_size 4 \
        --task MSDP-EVAL-F1 \
        --guess_file ${MODEL_GEN_PATH} \
        --answer_file ${GROUND_TRUTH_PATH}


############################################
# Evaluate BLEU, METEOR, and ROUGE-L scores.
############################################

# We follow the nlg-eval (https://github.com/Maluuba/nlg-eval) to 
# evaluate the BLEU, METEOR, and ROUGE-L scores. 

# To evaluate on these metrics, please setup the environments based on 
# the nlg-eval github, and run the corresponding evaluation commands.

nlg-eval \
    --hypothesis=<PATH_OF_THE_KNOWLEDGE_GENERATION> \
    --references=<PATH_OF_THE_GROUND_TRUTH_KNOWLEDGE>
