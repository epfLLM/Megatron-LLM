# echo Install dependencies for tokenization

# cd /scratch/model-parallel-trainer/
# python -m pip install --user transformers nltk
# python setup.py install --user

# echo Calling preprocess_data.py to tokenize
# python3 tools/preprocess_data.py --input /scratch/dummy-data/train.json --output_prefix wiki-train --dataset_impl mmap --tokenizer_type FalconTokenizer --workers 2 --chunk_size 2048
# python3 tools/preprocess_data.py --input /scratch/dummy-data/valid.json --output_prefix wiki-valid --dataset_impl mmap --tokenizer_type FalconTokenizer --workers 2 --chunk_size 2048

echo Calling preprocess_data.py to tokenize
python3 tools/preprocess_data.py \
    --input /home/ubuntu/data/openassistant_best_replies_train.jsonl \
    --output_prefix oa-top1-train \
    --dataset_impl mmap \
    --tokenizer_type=SentencePieceTokenizer \
    --vocab_file=/home/ubuntu/megatron-data/llama/tokenizer.model \
    --workers=4 \
    --chunk_size 2048

python3 tools/preprocess_data.py \
    --input /home/ubuntu/data/openassistant_best_replies_eval.jsonl \
    --output_prefix oa-top1-eval \
    --dataset_impl mmap \
    --tokenizer_type=SentencePieceTokenizer \
    --vocab_file=/home/ubuntu/megatron-data/llama/tokenizer.model \
    --workers=4 \
    --chunk_size 2048