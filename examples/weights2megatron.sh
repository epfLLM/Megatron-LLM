#! /bin/bash

# assert correct usage
if [[ $# -ne 2 ]]; then
	echo "Usage: $0 <llama/falcon> <7,13,30,34,40,65,70>"
	exit 1
fi


# extract variables
MODEL=$1
SIZE=$2


# determine cache directory (either raw llama or huggingface cache)
if [[ $MODEL = falcon ]]; then
	CACHE=/pure-mlo-scratch/alhernan/huggingface_cache/
elif [[ $MODEL = llama ]]; then
	CACHE=/pure-mlo-scratch/llama/${SIZE}B/
elif [[ $MODEL = llama2 ]]; then
	CACHE=/pure-mlo-scratch/alhernan/llama2/llama-2-${SIZE}b/
else
	echo "Model should be either llama or falcon, not $MODEL"
	exit 1
fi


# finally call the script
python weights2megatron/weights2megatron.py \
	$MODEL \
	--size=$SIZE \
	--out=/pure-mlo-scratch/alhernan/megatron-data/checkpoints/${MODEL}-${SIZE}b/ \
	--cache-dir=$CACHE
