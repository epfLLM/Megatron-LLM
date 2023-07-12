#! /bin/bash


# define help function
help () {
	echo "Usage: $0 <gpt/llama/falcon> [--size=7] [--tp=1] [--pp=4] [--help]"
}


# default arguments
SIZE=7
TP=1
PP=4


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
		*) echo unknown argument; help; exit 1;;
	esac
done


# set args
GPUS_PER_NODE=4
N_NODES=1
RANK=0
TENSORBOARD_PATH=/pure-mlo-scratch/alhernan/megatron-data/tensorboard/
CHECKPOINT_PATH=/pure-mlo-scratch/alhernan/megatron-data/checkpoints/${MODEL}${SIZE}b-tp$TP-pp$PP/ \
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $N_NODES --node_rank
                  $RANK --master_addr localhost --master_port 6000"
if [[ $MODEL = falcon ]]; then
	DATA_PATH=/pure-mlo-scratch/pagliard/data/wikitext-falcon/wiki-train_text_document
	TOKENIZER=FalconTokenizer
	EXTRA_ARGS="--use_multiquery_attn --parallel_attn"
elif [[ $MODEL = llama ]]; then
	DATA_PATH=/pure-mlo-scratch/alhernan/data/wikitext-llama-32000/wiki-train_text_document
	TOKENIZER=SentencePieceTokenizer
	EXTRA_ARGS="--vocab_file=/pure-mlo-scratch/llama/tokenizer.model --use_rms_norm
       	            --glu_activation swiglu --layernorm_epsilon 1e-6 --no_tie_embed_logits
		    --no_new_tokens"
elif [[ $MODEL = gpt ]]; then
	DATA_PATH=/scratch/wikitext-megatron/wiki-train_text_document
	TOKENIZER=GPT2BPETokenizer
	EXTRA_ARGS="--num_layers 16"
else
	echo "Model should be either gpt, llama or falcon, not $MODEL"
	help
	exit 1
fi
COMMON_ARGS="--use_flash_attn --no_bias_gelu_fusion --micro_batch_size 1
             --global_batch_size 1 --seq_length 2048 --max_position_embeddings 2048
      	     --lr 0.00015 --log_interval 10 --save_interval 1000 --eval_interval 1000
       	     --eval_iters 10 --hidden_dropout 0.0 --position_embedding_type rotary
	     --no_bias_dropout_fusion --use_checkpoint_args --train_iters 500000"


# print smoe args
echo
echo Settings:
echo MODEL=$MODEL
echo TP=$TP
echo PP=$PP
echo


# finally, call finetune.py
torchrun $DISTRIBUTED_ARGS finetune.py \
       --tensor_model_parallel_size $TP \
       --pipeline_model_parallel_size $PP  \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data_path $DATA_PATH \
       --tensorboard_dir $TENSORBOARD_PATH \
       --model_name $MODEL \
       --tokenizer_type $TOKENIZER \
       --bf16 \
       $EXTRA_ARGS \
       $COMMON_ARGS

       # --num_layers $NUM_LAYERS \
       # --hidden_size $HIDDEN_SIZE \
       # --num_attention_heads_kv $KV \
       # --num_attention_heads $NUM_HEADS \
       # --lr_decay_iters 320000 \
       # --data_impl mmap \
       # --split 949,50,1 \
       # --distributed_backend nccl \
       # --min_lr 1.0e-5 \
       # --weight_decay 1e-2 \
       # --clip_grad 1.0 \
       # --lr_warmup_fraction .01 \
       # --lr_decay_style cosine \
       # --sequence_parallel \
