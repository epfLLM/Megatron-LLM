# How to tokenize a dataset?

## Step 1: get the right json format

The training data requires preprocessing. First, place your training data in a loose json format, with one json containing a text sample per line. For example:
<pre>
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
</pre>

The name of the `text` field of the json can be changed by using the `--json-key` flag in [`preprocess_data.py`](./tools/preprocess_data.py) The other metadata are optional and are not used in training.

## Step 2: Tokenize 

The loose json is then processed into a binary format for training. To convert the json into mmap, cached index file, or the lazy loader format use `preprocess_data.py`. Set the `--dataset_impl` flag to `mmap`, `cached`, or `lazy`, respectively (default is `mmap`). An example script to prepare data for Falcon training is:
<pre>
python3 tools/preprocess_data.py --input /scratch/dummy-data/train.json 
    --output_prefix wiki-train 
    --dataset_impl mmap 
    --tokenizer_type FalconTokenizer 
    --workers 2 
    --chunk_size 2048
    --append_eod
</pre>

The output will be two files named, in this case, `my-bert_text_sentence.bin` and `my-bert_text_sentence.idx`. The `--data_path` specified in later BERT training is the full path and new filename, but without the file extension.

Other options of `preprocess_data.py`:

```bash
parser = argparse.ArgumentParser()
group = parser.add_argument_group(title='input data')
group.add_argument('--input', type=str, required=True,
                    help='Path to input JSON')
group.add_argument('--json_keys', nargs='+', default=['text'],
                    help='space separate listed of keys to extract from json')
group.add_argument('--split_sentences', action='store_true',
                    help='Split documents into sentences.')
group.add_argument('--keep_newlines', action='store_true',
                    help='Keep newlines between sentences when splitting.')

group = parser.add_argument_group(title='tokenizer')
group.add_argument('--tokenizer_type', type=str, required=True,
                    choices=['BertWordPieceLowerCase','BertWordPieceCase',
                            'GPT2BPETokenizer', 'SentencePieceTokenizer', 'FalconTokenizer'],
                    help='What type of tokenizer to use.')
group.add_argument('--vocab_file', type=str, default=None,
                    help='Path to the vocab file')
group.add_argument('--merge_file', type=str, default=None,
                    help='Path to the BPE merge file (if necessary).')
group.add_argument('--append_eod', action='store_true',
                    help='Append an <eod> token to the end of a document.')
group.add_argument('--lang', type=str, default='english',
                    help='Language to use for NLTK-powered sentence splitting.')
group = parser.add_argument_group(title='output data')
group.add_argument('--output_prefix', type=str, required=True,
                    help='Path to binary output file without suffix')
group.add_argument('--dataset_impl', type=str, default='mmap',
                    choices=['lazy', 'cached', 'mmap'])
group = parser.add_argument_group(title='runtime')
group.add_argument('--workers', type=int, required=True,
                    help='Number of worker processes to launch')
group.add_argument('--chunk_size', type=int, required=True,
                    help='Chunk size assigned to each worker process')
group.add_argument('--log_interval', type=int, default=100,
                    help='Interval between progress updates')
args = parser.parse_args()
```

## Tokenizing inside docker

The `run_dock.sh` contains a simple command:

```bash
sudo docker run --gpus 0 -it --rm --shm-size=2gb -v /scratch/pagliard/:/scratch --network host -v /home/pagliard/:/mpt epfllm -- /bin/bash -c 'bash /scratch/model-parallel-trainer/tokenize-utils/entrypoint.sh'
```

Replace the path to those corresponding to your setup, e.g. replace `pagliard` with your username ... 

The `run_dock.sh` creates a docker container, mount some folders, and call `entrypoint.sh`:

```bash
echo Install dependencies for tokenization

cd /scratch/model-parallel-trainer/
python -m pip install --user transformers nltk
python setup.py install --user

echo Calling preprocess_data.py to tokenize
python3 tools/preprocess_data.py --input /scratch/dummy-data/train.json --output_prefix wiki-train --dataset_impl mmap --tokenizer_type FalconTokenizer --workers 2 --chunk_size 2048 --append_eod
python3 tools/preprocess_data.py --input /scratch/dummy-data/valid.json --output_prefix wiki-valid --dataset_impl mmap --tokenizer_type FalconTokenizer --workers 2 --chunk_size 2048 --append_eod
```

Here as well you need to change the paths and commands to fit your config. 