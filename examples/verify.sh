#! /bin/bash

# assert correct usage
if [[ $# -ne 2 ]]; then
	echo "Usage: $0 <llama/llama2/falcon> <7,13,30,34,40,65,70>"
	exit 1
fi


# extract variables from command line
MODEL=$1
SIZE=$2


# based on the model, determine args
if [[ $MODEL = falcon ]]; then
	DATA_PATH=/pure-mlo-scratch/pagliard/data/wikitext-falcon/wiki-train_text_document
	CACHE=/pure-mlo-scratch/alhernan/huggingface_cache/
	TOKENIZER=FalconTokenizer
	EXTRA_ARGS=""
elif [[ $MODEL = llama ]] || [[ $MODEL = llama2 ]]; then
	DATA_PATH=/pure-mlo-scratch/alhernan/data/wikitext-llama-32000/wiki-train_text_document
	TOKENIZER=SentencePieceTokenizer
	EXTRA_ARGS="--vocab_file=/pure-mlo-scratch/llama/tokenizer.model --no_new_tokens --use_rms_norm
	            --glu_activation swiglu --no_tie_embed_logits"
	if [[ $MODEL = llama ]]; then
		CACHE=/pure-mlo-scratch/llama/converted_HF_${SIZE}B/
		EXTRA_ARGS="$EXTRA_ARGS --layernorm_epsilon 1e-6"
	else
		CACHE=/pure-mlo-scratch/alhernan/llama2/llama-2-${SIZE}b/
		EXTRA_ARGS="$EXTRA_ARGS --layernorm_epsilon 1e-5"
	fi
else
	echo "Model should be either llama, llama2  or falcon, not $MODEL"
	exit 1
fi
COMMON_ARGS="--hidden_dropout 0.0 --attention_dropout 0.0 --no_bias_dropout_fusion
             --no_bias_gelu_fusion --use_flash_attn"


# finally call the script
DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8000"
torchrun $DISTRIBUTED_ARGS verify_correctness.py \
	--model_name $MODEL \
       --load /pure-mlo-scratch/alhernan/megatron-data/checkpoints/${MODEL}-${SIZE}b/ \
       --data_path $DATA_PATH \
       --huggingface_cache $CACHE \
       --huggingface_device "cuda:1" \
       --tokenizer_type $TOKENIZER \
       --model_size $SIZE \
       --bf16 \
       $COMMON_ARGS \
       $EXTRA_ARGS
