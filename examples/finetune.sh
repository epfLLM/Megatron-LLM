#! /bin/bash

# define help function
help () {
	echo "Usage: $0 <gpt/llama/falcon> [--size=7] [--tp=1] [--pp=4] [--gpus=4] [--micro-batch=1] [--global-batch=1] [--help]"
}


# default arguments
SIZE=7
TP=1
PP=4
GPUS_PER_NODE=4
MICRO_BATCH=1
GLOBAL_BATCH=1


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
		*) echo unknown argument; help; exit 1;;
	esac
done


# set args
N_NODES=1
RANK=0
CHECKPOINT_PATH=/pure-mlo-scratch/alhernan/megatron-data/checkpoints/${MODEL}-${SIZE}b-tp$TP-pp$PP
TENSORBOARD_PATH=/scratch/alhernan/megatron-data/tensorboards/${MODEL}-${SIZE}b-tp$TP-pp$PP
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $N_NODES --node_rank
                  $RANK --master_addr localhost --master_port 6000"
if [[ $MODEL = falcon ]]; then
	DATA_PATH=/pure-mlo-scratch/pagliard/data/wikitext-falcon/wiki-train_text_document
	TOKENIZER=FalconTokenizer
	EXTRA_ARGS="--use_multiquery_attn --parallel_attn"
	SEQ_LEN=2048
elif [[ $MODEL = llama ]] || [[ $MODEL = llama2 ]]; then
	DATA_PATH=/pure-mlo-scratch/alhernan/data/wikitext-llama-32000/wiki-train_text_document
	TOKENIZER=SentencePieceTokenizer
	EXTRA_ARGS="--vocab_file=/pure-mlo-scratch/llama/tokenizer.model --use_rms_norm
	            --glu_activation swiglu --no_tie_embed_logits
		    --no_new_tokens"
	if [[ $MODEL == llama ]]; then
		SEQ_LEN=2048
		EXTRA_ARGS="$EXTRA_ARGS --layernorm_epsilon 1e-6"
	else
		SEQ_LEN=4096
		EXTRA_ARGS="$EXTRA_ARGS --layernorm_epsilon 1e-5"
	fi
elif [[ $MODEL = gpt ]]; then
	DATA_PATH=/scratch/wikitext-megatron/wikitext-train_text_document
	TOKENIZER=FalconTokenizer
	EXTRA_ARGS="--num_layers 4 --hidden_size 512 --num_attention_heads 8"
	SEQ_LEN=2048
else
	echo "Model should be either gpt, llama or falcon, not $MODEL"
	help
	exit 1
fi
COMMON_ARGS="--use_flash_attn --no_bias_gelu_fusion
	     --seq_length $SEQ_LEN --max_position_embeddings $SEQ_LEN
             --log_interval 100 --save_interval 1000 --eval_interval 1000
             --eval_iters 100 --hidden_dropout 0.0 --position_embedding_type rotary
	     --no_bias_dropout_fusion --use_checkpoint_args --train_iters 10000
	     --attention_dropout 0.0 --adam_beta1 0.9 --adam_beta2 0.95 --adam_eps 1e-5
	     --lr_decay_style cosine --lr_warmup_iters 2000 --lr 1e-5 --min_lr 1e-6
	     --weight_decay 0.1 --sequence_parallel --recompute_granularity selective"  # TODO: is lr fine?

# print some args
echo
echo Settings:
echo CHECKPOINT_PATH=$CHECKPOINT_PATH
echo MODEL=$MODEL
echo TP=$TP
echo PP=$PP
echo


# finally, call finetune.py
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun $DISTRIBUTED_ARGS finetune.py \
       --tensor_model_parallel_size $TP \
       --pipeline_model_parallel_size $PP  \
       --load $CHECKPOINT_PATH \
       --save $CHECKPOINT_PATH-trained \
       --tensorboard_dir $TENSORBOARD_PATH \
       --data_path $DATA_PATH \
       --model_name $MODEL \
       --tokenizer_type $TOKENIZER \
       --bf16 \
       --global_batch_size $GLOBAL_BATCH \
       --micro_batch_size $MICRO_BATCH \
       $EXTRA_ARGS \
       $COMMON_ARGS
