#! /bin/bash


# default arguments
SIZE=13
TP=4
PP=2
GPUS_PER_NODE=8
MICRO_BATCH=2
#MICRO_BATCH=1
#GLOBAL_BATCH=8
GLOBAL_BATCH=64
RANK=0
N_NODES=1
ADDR=localhost
WANDB=0
HELP_STR="[--rank=$RANK] [--size=$SIZE] [--tp=$TP] [--pp=$PP] [--gpus=$GPUS_PER_NODE] \
[--micro-batch=$MICRO_BATCH] [--global-batch=$GLOBAL_BATCH] [--nodes=$N_NODES] \
[--addr=$ADDR] [--wandb] [--help]"


# define help function
help () {
	echo "Usage: $0 <gpt/llama/llama2/codellama/falcon> $HELP_STR"
}


# parse arguments, three modes
# mode1 = -h or --help requested
if [[ $# = 1 ]] && [[ $1 = "-h" ]] || [[ $1 = "--help" ]]; then
	help
	exit 0
# mode2 = not arguments given
elif [[ $# = 0 ]]; then
	help
	exit 1
fi
# mode3 = correct usage, read model
MODEL=$1
shift
while [[ $# -gt 0 ]]; do
	case $1 in
		--tp) TP="$2"; shift; shift;;
		--pp) PP="$2"; shift; shift;;
		--size) SIZE="$2"; shift; shift;;
		--gpus) GPUS_PER_NODE="$2"; shift; shift;;
		--micro-batch) MICRO_BATCH="$2"; shift; shift;;
		--global-batch) GLOBAL_BATCH="$2"; shift; shift;;
		--rank) RANK=$2; shift; shift;;
		--nodes) N_NODES=$2; shift; shift;;
		--addr) ADDR=$2; shift; shift;;
		--wandb) WANDB=1; shift;;
		*) echo unknown argument $1; help; exit 1;;
	esac
done


# set args
LR="3e-4"
CHECKPOINT_PATH=/pure-mlo-scratch/akoepf/checkpoints/${MODEL}-${SIZE}b-tp$TP-pp$PP
#CHECKPOINT_PATH=/pure-mlo-scratch/akoepf/checkpoints/llama2-13b-tp4-pp2-megacode2_min100
#CHECKPOINT_PATH=/pure-mlo-scratch/akoepf/checkpoints/${MODEL}-${SIZE}b-tp$TP-pp$PP-oasst_pre10
OUTPUT_PATH=/pure-mlo-scratch/akoepf/checkpoints/${MODEL}-${SIZE}b-tp$TP-pp$PP-oaast1
#OUTPUT_PATH=/pure-mlo-scratch/akoepf/checkpoints/${MODEL}-${SIZE}b-tp$TP-pp$PP-oasst_sft10
#OUTPUT_PATH=/pure-mlo-scratch/akoepf/checkpoints/${MODEL}-${SIZE}b-tp$TP-pp$PP-test_rope_scale2
TENSORBOARD_PATH=$OUTPUT_PATH/logging
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $N_NODES --node_rank
                  $RANK --master_addr $ADDR --master_port 6000"
if [[ $MODEL = falcon ]]; then
	#DATA_PATH=/pure-mlo-scratch/akoepf/data/megacode2_min100_falcon/megacode2-train
	#DATA_PATH=/pure-mlo-scratch/akoepf/data/oasst_top1_2023-07-23_falcon/oasst_top1-train
	TOKENIZER=FalconTokenizer
	EXTRA_ARGS='--parallel_attn --vocab_extra_ids_list "<|im_start|>,<|im_end|>"'
	SEQ_LEN=2048
elif [[ $MODEL = llama ]] || [[ $MODEL = llama2 ]] || [[ $MODEL = codellama ]]; then
	#DATA_PATH=/pure-mlo-scratch/akoepf/data/orcamegacode_best_llama2/orcamegacode_best-train
	DATA_PATH=/pure-mlo-scratch/akoepf/data/oasst_top1_2023-07-23_llama2/oasst_top1-train
	#DATA_PATH=/pure-mlo-scratch/akoepf/data/oasst_pre10_min25_llama2/oasst_sft10-train
	#DATA_PATH=/pure-mlo-scratch/akoepf/data/megacode2_min100/megacode2-train
	#DATA_PATH=/pure-mlo-scratch/akoepf/data/megacode2_frac05/megacode2-train
	TOKENIZER=SentencePieceTokenizer
	EXTRA_ARGS='--use_rms_norm
			--glu_activation swiglu --no_tie_embed_logits
			--vocab_extra_ids_list "<|im_start|>,<|im_end|>"'
	if [[ $MODEL = codellama ]]; then
		EXTRA_ARGS="$EXTRA_ARGS --vocab_file=/pure-mlo-scratch/akoepf/codellama/CodeLlama-7b/tokenizer.model --rope_theta 1e6"
	else
		EXTRA_ARGS="$EXTRA_ARGS --vocab_file=/pure-mlo-scratch/llama2/Llama-2-7b-hf/tokenizer.model"
	fi
	if [[ $MODEL == llama ]]; then
		SEQ_LEN=2048
		EXTRA_ARGS="$EXTRA_ARGS --layernorm_epsilon 1e-6"
	else  # llama 2
		SEQ_LEN=4096
		EXTRA_ARGS="$EXTRA_ARGS --layernorm_epsilon 1e-5"
		if (( $SIZE > 13 )); then  # llama 2, 34B and 70B
			LR="1.5e-4"
		fi
	fi
elif [[ $MODEL = gpt ]]; then
	DATA_PATH=/home/ubuntu/megatron-data/oa-top1/oa-top1-train_text_document
	TOKENIZER=FalconTokenizer
	EXTRA_ARGS="--num_layers 4 --hidden_size 512 --num_attention_heads 8"
	SEQ_LEN=2048
else
	echo "Model should be either gpt, llama or falcon, not $MODEL"
	help
	exit 1
fi
# COMMON_ARGS="--use_flash_attn --no_bias_gelu_fusion
# 	--seq_length $SEQ_LEN --max_position_embeddings $SEQ_LEN
# 	--log_interval 1 --save_interval 500 --eval_interval 50
# 	--eval_iters 10 --hidden_dropout 0.0 --position_embedding_type rotary
# 	--no_bias_dropout_fusion --use_checkpoint_args --train_iters 6123
# 	--attention_dropout 0.0 --adam_beta1 0.9 --adam_beta2 0.95 --adam_eps 1e-12
# 	--lr_decay_style cosine --lr_warmup_iters 100 --lr 1e-5 --min_lr 1e-6
# 	--weight_decay 0.000001 --sequence_parallel --recompute_granularity selective --log_timers_to_tensorboard
# 	--rope_scaling_factor 1.0"


# OA lima fine-tuning:
COMMON_ARGS="--use_flash_attn --no_bias_gelu_fusion
	--seq_length $SEQ_LEN --max_position_embeddings $SEQ_LEN
	--log_interval 1 --save_interval 346 --eval_interval 50
	--eval_iters 10 --hidden_dropout 0.25 --lima_dropout --position_embedding_type rotary
	--no_bias_dropout_fusion --use_checkpoint_args --train_iters 519
	--attention_dropout 0.0 --adam_beta1 0.9 --adam_beta2 0.95 --adam_eps 1e-12
	--lr_decay_style cosine --lr_warmup_iters 100 --lr 1e-5 --min_lr 1e-6
	--weight_decay 0.000001 --sequence_parallel --recompute_granularity selective --log_timers_to_tensorboard
	--rope_scaling_factor 1.0 --finetune"

# COMMON_ARGS="--use_flash_attn --no_bias_gelu_fusion
# 	     --seq_length $SEQ_LEN --max_position_embeddings $SEQ_LEN
#              --log_interval 1 --save_interval 50 --eval_interval 50
#              --eval_iters 10 --hidden_dropout 0.0 --position_embedding_type rotary
# 	     --no_bias_dropout_fusion --use_checkpoint_args --train_iters 10000
# 	     --attention_dropout 0.0 --adam_beta1 0.9 --adam_beta2 0.95 --adam_eps 1e-5
# 	     --lr_decay_style cosine --lr_warmup_iters 2000 --lr 2e-5 --min_lr 1e-6
# 	     --weight_decay 0.1 --sequence_parallel --recompute_granularity selective"

if [[ $WANDB = 1 ]]; then
#	COMMON_ARGS="$COMMON_ARGS --wandb_logger --wandb_project epfl-mt-sft --wandb_entity open-assistant --wandb_id run42_oasst_pre10"
	COMMON_ARGS="$COMMON_ARGS --wandb_logger --wandb_project epfl-mt-sft --wandb_entity open-assistant --wandb_id run52_oasst_codellama_7b"
fi

# print some args
echo
echo Settings:
echo RANK=$RANK
echo ADDR=$ADDR
echo N_NODES=$N_NODES
echo DATA_PATH=$DATA_PATH
echo CHECKPOINT_PATH=$CHECKPOINT_PATH
echo MODEL=$MODEL
echo TP=$TP
echo PP=$PP
echo MICRO_BATCH=$MICRO_BATCH
echo GLOBAL_BATCH=$GLOBAL_BATCH
echo COMMON_ARGS=$COMMON_ARGS
echo EXTRA_ARGS=$EXTRA_ARGS
echo


# finally, call finetune.py
CUDA_DEVICE_MAX_CONNECTIONS=1 OMP_NUM_THREADS=16 torchrun $DISTRIBUTED_ARGS finetune.py \
       --tensor_model_parallel_size $TP \
       --pipeline_model_parallel_size $PP  \
       --load $CHECKPOINT_PATH \
       --save $OUTPUT_PATH \
       --tensorboard_dir $TENSORBOARD_PATH \
       --data_path $DATA_PATH \
       --model_name $MODEL \
       --tokenizer_type $TOKENIZER \
       --bf16 \
       --global_batch_size $GLOBAL_BATCH \
       --micro_batch_size $MICRO_BATCH \
       $EXTRA_ARGS \
       $COMMON_ARGS
