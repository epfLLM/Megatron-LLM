#!/bin/bash


#SBATCH <SLURM OPTIONS> --nodes=128 --exclusive --ntasks-per-node=8 --job-name=megatron_gpt3_175b


DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs


DATASET_1="<PATH TO THE FIRST DATASET>"
DATASET_2="<PATH TO THE SECOND DATASET>"
DATASET_3="<PATH TO THE THIRD DATASET>"
DATASET="0.2 ${DATASET_1} 0.3 ${DATASET_2} 0.5 ${DATASET_3}"


options=" \
	--tensor_model_parallel_size 8 \
	--pipeline_model_parallel_size 16 \
  --num_layers 96 \
  --hidden_size 12288 \
  --num_attention_heads 96 \
  --seq_length 2048 \
  --max_position_embeddings 2048 \
	--micro_batch_size 1 \
	--global_batch_size 1536 \
	--rampup_batch_size 16 16 5859375 \
	--train_samples 146484375 \
  --lr_decay_samples 126953125 \
  --lr_warmup_samples 183105 \
  --lr 6.0e-5 \
	--min_lr 6.0e-6 \
  --lr_decay_style cosine \
  --log_interval 10 \
  --eval_iters 40 \
  --eval_interval 1000 \
	--data_path ${DATASET} \
	--vocab_file <PATH TO gpt-vocab.json> \
	--merge_file <PATH TO gpt-merges.txt> \
	--save_interval 1000 \
	--save <PATH TO CHECKPOINTS DIRECTORY> \
	--load <PATH TO CHECKPOINTS DIRECTORY> \
  --split 98,2,0 \
  --clip_grad 1.0 \
	--weight_decay 0.1 \
	--adam_beta1 0.9 \
	--adam_beta2 0.95 \
	--init_method_std 0.006 \
	--tensorboard_dir <TENSORBOARD DIRECTORY> \
  --fp16 \
	--activations_checkpoint_method uniform "

run_cmd="python -u ${DIR}/pretrain_gpt.py $@ ${options}"


srun -l \
     --container-image "nvcr.io/nvidia/pytorch:20.12-py3" \
     --container-mounts "<DIRECTORIES TO MOUNT>" \
     --output=$DIR/logs/%x_%j_$DATETIME.log sh -c "${run_cmd}"


set +x

