#! /bin/bash

# assert correct usage
if [[ $# -ne 2 ]]; then
	echo "Usage: $0 <llama/falcon> <7,13,30,40,65>"
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
elif [[ $MODEL = llama ]]; then
	DATA_PATH=/pure-mlo-scratch/alhernan/data/wikitext-llama-32000/wiki-train_text_document
	CACHE=/pure-mlo-scratch/llama/converted_HF_${SIZE}B/
	TOKENIZER=SentencePieceTokenizer
	EXTRA_ARGS="--vocab_file=/pure-mlo-scratch/llama/tokenizer.model --no_new_tokens --use_rms_norm
       	            --glu_activation swiglu --layernorm_epsilon 1e-6 --no_tie_embed_logits"
else
	echo "Model should be either llama or falcon, not $MODEL"
	exit 1
fi
COMMON_ARGS="--hidden_dropout 0.0 --attention_dropout 0.0 --no_bias_dropout_fusion
             --no_bias_gelu_fusion --use_flash_attn"


# finally call the script
DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8000"
torchrun $DISTRIBUTED_ARGS verify_correctness.py \
	--model_name $MODEL \
       --load /pure-mlo-scratch/alhernan/megatron-data/checkpoints/${MODEL}${SIZE}b/ \
       --data_path $DATA_PATH \
       --huggingface_cache $CACHE \
       --huggingface_device "cuda:1" \
       --tokenizer_type $TOKENIZER \
       --model_size $SIZE \
       $COMMON_ARGS \
       $EXTRA_ARGS \
