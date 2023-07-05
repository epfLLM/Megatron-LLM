if [[ $# -ne 1 ]]; then
	echo "Usage: $0 <7 or 40>"
	exit 1
fi


python falcon2megatron/falcon2megatron.py --size=$1 --out=/scratch/alhernan/megatron-data/checkpoints/falcon${1}b/ --cache-dir=/scratch/alhernan/huggingface_cache/
