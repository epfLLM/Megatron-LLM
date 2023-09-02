# Instruction code heavily inspired by Andreas Köpf
# source: https://github.com/andreaskoepf/epfl-megatron/tree/local_changes/
"""Processing data for instruction tuning.
Example:
python instruct/preprocess_instruct_data.py --input=/pure-mlo-scratch/alhernan/data/medmc/medmc-v1.jsonl \
    --output_prefix=/pure-mlo-scratch/alhernan/data/medmc/medmc-v1 \
    --tokenizer_type=SentencePieceTokenizer \
    --vocab_file=/pure-mlo-scratch/llama/tokenizer.model \
    --chunk_size=32 --workers=32 \
    --vocab_extra_ids_list "[bib_ref],[/bib_ref],[fig_ref],[/fig_ref],[bib],[/bib],[fig],[/fig],[table],[/table],[formula],[/formula],<|im_start|>,<|im_end|>" \
    --question_key=input \
    --answer_key=output \
    --system_key=instruction
"""

import sys
import json
import time
import itertools
from pathlib import Path
from typing import Optional
from multiprocessing import Pool
from argparse import ArgumentParser, Namespace

import torch

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from megatron.tokenizer import build_tokenizer
from megatron.tokenizer.tokenizer import AbstractTokenizer
from megatron.data.indexed_dataset import make_builder
from megatron.data.instruction_dataset import Role


class Encoder(object):
    tokenizer: Optional[AbstractTokenizer] = None

    def __init__(self, args: Namespace):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = build_tokenizer(self.args)

    def encode(self, line: str) -> tuple[int, list[int], list[int]]:
        # get data
        assert Encoder.tokenizer is not None
        data = json.loads(line)
        question = data[self.args.question_key]
        answer = data[self.args.answer_key]
        system = None if self.args.system_key is None else data[self.args.system_key]

        # now format messages
        if system is not None:
            system = format_message(system, "system")
        question = format_message(question, "question")
        answer = format_message(answer, "answer")

        # tokenize and get roles
        tokens = []
        roles = []
        if system is not None:
            system = Encoder.tokenizer.tokenize(system)
            tokens += system
            roles += [Role.system.value]*len(system)
        question = Encoder.tokenizer.tokenize(question)
        tokens += question
        roles += [Role.prompter.value]*len(question)
        answer = Encoder.tokenizer.tokenize(answer)
        tokens += answer
        roles += [Role.assistant.value]*len(answer)
        return len(line), tokens, roles

    @property
    def special_tokens(self) -> dict:
        return self.tokenizer._special_tokens


class DatasetWriter:
    def __init__(self, prefix: str, vocab_size: int, dataset_impl: str = "mmap",
                 feature: str = "text"):
        self.vocab_size = vocab_size
        self.dataset_impl = dataset_impl
        self.bin_fname = f"{prefix}-{feature}.bin"
        self.idx_fname = f"{prefix}-{feature}.idx"
        self.builder = None

    def add_item(self, tokens: list[int]):
        self.builder.add_item(torch.IntTensor(tokens))

    def __enter__(self):
        self.builder = make_builder(self.bin_fname, impl=self.dataset_impl,
                                    vocab_size=self.vocab_size)
        return self

    def __exit__(self, *_):
        self.builder.finalize(self.idx_fname)
        self.builder = None


def format_message(message: str, role: str) -> str:
    return f"<|im_start|>{role}\n{message}<|im_end|>\n"


def get_args():
    parser = ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, nargs="+",
                       help='Path(s) to input JSON file(s)')
    group.add_argument('--system_key',
                       help='key to extract system info from json (optional)')
    group.add_argument('--question_key', default='input',
                       help='key to extract questions from json')
    group.add_argument('--answer_key', default='output',
                       help='key to extract answers from json')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer_type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'SentencePieceTokenizer', 'FalconTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab_file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge_file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output_prefix', type=Path, required=True,
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
    group.add_argument('--vocab_extra_ids', type=int, default=0)
    group.add_argument('--vocab_extra_ids_list', type=str, default=None,
                       help='comma separated list of special vocab ids to add to the tokenizer')
    group.add_argument("--no_new_tokens", action="store_false", dest="new_tokens",
                       help=("Whether to add special tokens (e.g. CLS, MASK, etc) "
                             "in the sentencepiece tokenizer or not"))
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    return args


def main():
    args = get_args()
    startup_start = time.time()

    encoder = Encoder(args)
    vocab_size = build_tokenizer(args).vocab_size
    fs = map(open, args.input)
    with Pool(args.workers, initializer=encoder.initializer) as pool, \
            DatasetWriter(args.output_prefix, vocab_size, args.dataset_impl,
                          "text") as token_writer, \
            DatasetWriter(args.output_prefix, 16, args.dataset_impl,
                          "role") as role_writer:

        f = itertools.chain(*fs)
        docs = pool.imap(encoder.encode, f, args.chunk_size)
        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)

        for i, (size, tokens, roles) in enumerate(docs, start=1):
            total_bytes_processed += size
            token_writer.add_item(tokens)
            role_writer.add_item(roles)

            if i % args.log_interval == 0:
                elapsed = time.time() - proc_start
                mbs = total_bytes_processed/1024/1024/elapsed
                print(f"Processed {i} documents ({i/elapsed} docs/s, {mbs} MB/s).")
        print("Done! Now finalizing.")

    for f in fs:
        f.close()


if __name__ == '__main__':
    main()
