#! /bin/bash

# assert correct usage
if [[ $# -ne 2 ]]; then
	echo "Usage: $0 <llama/falcon> <7,13,30,40,65>"
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
else
	echo "Model should be either llama or falcon, not $MODEL"
	exit 1
fi


# finally call the script
python weights2megatron/weights2megatron.py \
	$MODEL \
	--size=$SIZE \
	--out=/pure-mlo-scratch/alhernan/megatron-data/checkpoints/${MODEL}${SIZE}b/ \
	--cache-dir=$CACHE
