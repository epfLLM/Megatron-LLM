#! /bin/bash

DO_EVAL=0

if [ "$DO_EVAL" -eq 1 ]; then
    echo "PPL Validation"
    bash /main/meditron_run/finetune.sh llama2 \
        --rank 0 \
        --nodes 1 \
        --addr gpu008.rcp.epfl.ch \
        --tp 4 \
        --micro-batch 200 \
        --global-batch 2000 \
        --wandb \
        --test uptodate \
        --iter 8000 \
        --exp ex3-pubmed-replay-code \
        --do_eval
else
    echo "Pre-Training"
        bash /main/meditron_run/finetune.sh llama2 \
        --rank 0 \
        --nodes 4 \
        --addr gpu001.rcp.epfl.ch \
        --tp 4 \
        --micro-batch 10 \
        --global-batch 1000 \
        --wandb
fi


