# This file isn't really a formal automated test, it's just a place to
# put some code used during development and manual testing of
# indexed_dataset.
import argparse
import os
import sys

import torch

import megatron.tokenizer
from megatron.data import indexed_dataset

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, "../../../"))


def test_indexed_dataset(args):
    ds = indexed_dataset.make_dataset(args.data, args.dataset_impl)
    tokenizer = megatron.tokenizer.build_tokenizer(args)
    count = args.count
    del args
    print(len(ds.doc_idx))
    print(len(ds))
    print(ds.doc_idx[-1])
    if ds.supports_prefetch:
        # just prefetch the whole thing in test (so assume it is small)
        ds.prefetch(range(len(ds)))
    if count > len(ds.doc_idx) - 1:
        count = len(ds.doc_idx) - 1

    for i in range(count):
        start = ds.doc_idx[i]
        end = ds.doc_idx[i + 1]
        ids = ds[start:end]
        print(f"Document {i}:")
        print("--------------")
        for s in ids:
            assert len(s) > 0
            l = s.data.tolist()
            text = tokenizer.detokenize(l)
            print(text)
            print("---")


def test_indexed_dataset_get(args):
    ds = indexed_dataset.make_dataset(args.data, args.dataset_impl)
    size = ds.sizes[0]
    print(f"size: {size}")
    full = ds.get(0)
    print(full)

    print("---")
    end = ds.get(0, offset=size - 10)
    print(end)

    start = ds.get(0, length=10)
    print(start)

    part = ds.get(0, offset=2, length=8)
    print(part)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='prefix to data files')
    parser.add_argument('--dataset_impl', type=str, default='infer',
                        choices=['lazy', 'cached', 'mmap', 'infer'])
    parser.add_argument('--count', type=int, default=10,
                        help='Number of samples/documents to print')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer_type', type=str, required=True,
                       choices=['BertWordPieceLowerCase',
                                'GPT2BPETokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab_file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge_file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to plan for')
    parser.add_argument('--max_num_samples', type=int, default=None,
                        help='Maximum number of samples to plan for')
    parser.add_argument('--masked_lm_prob', type=float, default=0.15,
                        help='probability of masking tokens')
    parser.add_argument('--seq_length', type=int, default=512,
                        help='maximum sequence length')
    parser.add_argument('--short_seq_prob', type=float, default=0.1,
                        help='probability of creating a short sequence')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed')
    args = parser.parse_args()
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1

    if args.dataset_impl == "infer":
        args.dataset_impl = indexed_dataset.infer_dataset_impl(args.data)

#    test_albert_dataset(args)
    test_indexed_dataset_get(args)


if __name__ == "__main__":
    main()
