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
	CACHE=/home/ubuntu/falcon
elif [[ $MODEL = llama ]]; then
	CACHE=/home/ubuntu/llama/${SIZE}B/
elif [[ $MODEL = llama2 ]]; then
	CACHE=/home/ubuntu/llama2/Llama-2-${SIZE}b/
else
	echo "Model should be either llama or falcon, not $MODEL"
	exit 1
fi


# finally call the script
mkdir /home/ubuntu/megatron-data
python weights2megatron/weights2megatron.py \
	$MODEL \
	--size=$SIZE \
	--out=/home/ubuntu/megatron-data/${MODEL}-${SIZE}b/ \
	--cache-dir=$CACHE
