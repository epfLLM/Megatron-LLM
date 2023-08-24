#! /bin/bash

# assert correct usage
if [[ $# -ne 4 ]]; then
	echo "Usage: $0 <llama/llama2/codellama/falcon> <7,13,30,40,65,70> <tp> <pp>"
	exit 1
fi


# extract variables from command line
MODEL=$1
SIZE=$2
TENSOR_PARALLELISM=$3
PIPELINE_PARALLELISM=$4


# model-specific parameters
EXTRA_ARGS=""
if [[ $MODEL = falcon ]]; then
	#TRUE_VOCAB_SIZE=65024
	TRUE_VOCAB_SIZE=65026       # 2 new tokens: <|im_start|>,<|im_end|>
elif [[ $MODEL = llama ]] || [[ $MODEL = llama2 ]]; then
	#TRUE_VOCAB_SIZE=32017  # 17 new tokens
	TRUE_VOCAB_SIZE=32007  # 7 new tokens:  <CLS>,<SEP>,EOD>,<MASK>,<PAD>,<|im_start|>,<|im_end|>
	if (( $SIZE > 60 )); then
		EXTRA_ARGS="--bf16"
	fi
elif [[ $MODEL = codellama ]]; then
	TRUE_VOCAB_SIZE=32023  # 32016 + 7 new tokens:  <CLS>,<SEP>,EOD>,<MASK>,<PAD>,<|im_start|>,<|im_end|>
fi


# finally call the script
python tools/checkpoint_util.py \
       --target_tensor_parallel_size $TENSOR_PARALLELISM \
       --target_pipeline_parallel_size $PIPELINE_PARALLELISM \
       --load_dir /pure-mlo-scratch/akoepf/checkpoints/${MODEL}-${SIZE}b/ \
       --save_dir /pure-mlo-scratch/akoepf/checkpoints/${MODEL}-${SIZE}b-tp$TENSOR_PARALLELISM-pp$PIPELINE_PARALLELISM/ \
       --model_type $MODEL \
       --true_vocab_size $TRUE_VOCAB_SIZE \
       $EXTRA_ARGS
