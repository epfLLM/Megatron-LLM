# How to tokenize a dataset?

## Step 1: get the right json format

The training data requires preprocessing. First, place your training data in a loose json format, with one json containing a text sample per line. For example:
<pre>
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
</pre>

The name of the `text` field of the json can be changed by using the `--json-key` flag in `preprocess_data.py`.
The other metadata are optional and are not used in training.

## Step 2: Tokenize 

The loose json is then processed into a binary format for training. To convert the json into mmap, cached index file, or the lazy loader format use `preprocess_data.py`. Set the `--dataset_impl` flag to `mmap`, `cached`, or `lazy`, respectively (default is `mmap`). An example script to prepare data for Falcon training is:
<pre>
python3 tools/preprocess_data.py --input /scratch/dummy-data/train.json 
    --output_prefix wiki-train 
    --dataset_impl mmap 
    --tokenizer_type FalconTokenizer 
    --workers 2 
    --chunk_size 32
    --append_eod
</pre>

The output will be two files named, in this case, `my-bert_text_sentence.bin` and `my-bert_text_sentence.idx`. The `--data_path` specified in later BERT training is the full path and new filename, but without the file extension.

Other options of `preprocess_data.py`:

```
input data:                                                                   
  --input INPUT         Path to input JSON
  --json_keys JSON_KEYS [JSON_KEYS ...]      
                        space separate listed of keys to extract from json                                                                                   
  --split_sentences     Split documents into sentences.                                                                                                      
  --keep_newlines       Keep newlines between sentences when splitting.

tokenizer:
  --tokenizer_type {BertWordPieceLowerCase,BertWordPieceCase,GPT2BPETokenizer,SentencePieceTokenizer,FalconTokenizer}
                        What type of tokenizer to use.
  --vocab_file VOCAB_FILE
                        Path to the vocab file
  --merge_file MERGE_FILE
                        Path to the BPE merge file (if necessary).
  --append_eod          Append an <eod> token to the end of a document.
  --lang LANG           Language to use for NLTK-powered sentence splitting.

output data:
  --output_prefix OUTPUT_PREFIX
                        Path to binary output file without suffix
  --dataset_impl {lazy,cached,mmap}

runtime:
  --workers WORKERS     Number of worker processes to launch
  --chunk_size CHUNK_SIZE
                        Chunk size assigned to each worker process
  --log_interval LOG_INTERVAL
                        Interval between progress updates
  --vocab_extra_ids VOCAB_EXTRA_IDS
  --vocab_extra_ids_list VOCAB_EXTRA_IDS_LIST
                        comma separated list of special vocab ids to add to the tokenizer
  --no_new_tokens       Whether to add special tokens (e.g. CLS, MASK, etc) in the sentenciepiece tokenizer or not
```

If you want to tokenize using llama tokenizer:
```
python tools/preprocess_data.py \
        --input=/path/to/data.json \
        --output_prefix=wiki-train \
        --dataset_impl=mmap \
        --tokenizer_type=SentencePieceTokenizer \
        --vocab_file=/path/to/tokenizer.model \
        --workers=2 \
        --chunk_size=32
```
