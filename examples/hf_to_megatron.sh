#! /bin/bash

# assert correct usage
if [[ $# -ne 2 ]]; then
	echo "Usage: $0 <llama/llama2/codellama/falcon> <7,13,30,34,40,65,70>"
	exit 1
fi


# extract variables
MODEL=$1
SIZE=$2


# determine cache directory (either raw llama or huggingface cache)
if [[ $MODEL = falcon ]]; then
	CACHE=/pure-mlo-scratch/huggingface_cache/
elif [[ $MODEL = llama ]]; then
	CACHE=/pure-mlo-scratch/llama/${SIZE}B/
elif [[ $MODEL = llama2 ]]; then
	CACHE=/pure-mlo-scratch/llama2/llama-2-${SIZE}b/
elif [[ $MODEL = codellama ]]; then
	CACHE=/pure-mlo-scratch/codellama/CodeLlama-${SIZE}b/
else
	echo "Model should be either llama, llama2, codellama or falcon, not $MODEL"
	exit 1
fi


# finally call the script
python weights_conversion/hf_to_megatron.py \
	$MODEL \
	--size=$SIZE \
	--out=/pure-mlo-scratch/alhernan/megatron-data/checkpoints/${MODEL}-${SIZE}b/ \
	--cache-dir=$CACHE
