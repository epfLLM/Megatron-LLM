import argparse
import os


NODES = ["gpu011.rcp.epfl.ch", "gpu012.rcp.epfl.ch"]

def parse_args():
    # get the expected world size from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws", type=int, default=2)
    args = parser.parse_args()
    return args

def create_config(rank:int, world_size:int):
    return f"""
#! /bin/bash

# Runs the "345M" parameter model
(
cd /mpt
set -e 

export CUDA_DEVICE_MAX_CONNECTIONS=1
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=gpu011.rcp.epfl.ch
MASTER_PORT=6000
NNODES={world_size}
NODE_RANK={rank}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/scratch/wikitext-megatron/wikitext-train_text_document
EXP_DIR=/scratch/$(whoami)/exp
CHECKPOINT_PATH=${{EXP_DIR}}/checkpoint
TENSORBOARD_PATH=${{EXP_DIR}}/tensorboard

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor_model_parallel_size 1 \
       --pipeline_model_parallel_size 1 \
       --num_layers 24 \
       --hidden_size 1024 \
       --num_attention_heads 16 \
       --micro_batch_size 4 \
       --global_batch_size 128 \
       --seq_length 1024 \
       --max_position_embeddings 1024 \
       --train_iters 500000 \
       --lr_decay_iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data_path $DATA_PATH \
       --vocab_file gpt2-vocab/gpt2-vocab.json \
       --merge_file gpt2-vocab/gpt2-merges.txt \
       --data_impl mmap \
       --split 949,50,1 \
       --distributed_backend nccl \
       --lr 0.00015 \
       --lr_decay_style cosine \
       --min_lr 1.0e-5 \
       --weight_decay 1e-2 \
       --clip_grad 1.0 \
       --lr_warmup_fraction .01 \
       --log_interval 100 \
       --save_interval 10000 \
       --eval_interval 1000 \
       --eval_iters 10 \
       --use_bias \
       --use_flash_attn \
       --tensorboard_dir ${{TENSORBOARD_PATH}} \
       --bf16
)
"""

DOCKER_COMMAND = """docker run --gpus all --rm --shm-size=32gb -v /scratch:/scratch --network host -v /home/axel/model-parallel-trainer/:/mpt epfllm /mpt/examples/pretrain_gpt_distributed_epflrcp.sh"""

if __name__ == "__main__":
    args = parse_args()
    for rank in range(args.ws):
        with open(f"config_{rank}.sh", "w") as f:
            f.write(create_config(rank, args.ws))
        # rsync the root folder with the nodes
        ret = os.system(f"rsync -r . {NODES[rank]}:~/model-parallel-trainer")
        if ret != 0:
            print("Rsync failed")
            exit(1)
        
        # copy the config to the node
        ret = os.system(f"scp config_{rank}.sh {NODES[rank]}:~/model-parallel-trainer/examples/pretrain_gpt_distributed_epflrcp.sh")

        # run the script on docker on the node
        print(f"Running on {NODES[rank]}: {DOCKER_COMMAND}")
        ret = os.system(f"ssh {NODES[rank]} {DOCKER_COMMAND}")