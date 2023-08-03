# import sys, os
import argparse
from pathlib import Path
import logging
import json

import torch
from torch.utils.data import Dataset, Subset

from model_training.utils.utils import (
    get_dataset,
    read_yamls,
    _strtobool,
)
from model_training.custom_datasets.formatting import (
    DatasetEntryLm,
    DatasetEntrySft,
    Role,
)


from tokenizer import build_tokenizer
import indexed_dataset

logger = logging.getLogger(__name__)


class Encoder(object):
    def __init__(self, args):
        self.args = args
        self.tokenizer = build_tokenizer(self.args)

    def encode_text(self, text: str) -> list[int]:
        return self.tokenizer.tokenize(text)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.detokenize(tokens)

    @property
    def special_tokens(self) -> dict:
        return self.tokenizer._special_tokens


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="configuration")
    group.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Multiple configs can be passed to set different options.",
    )
    group.add_argument(
        "--output_dir",
        type=str,
        help="Path to binary output file without suffix",
    )

    args, remaining = parser.parse_known_args()

    # load yaml configurations
    conf = {}
    configs = read_yamls("./configs")
    conf.update(configs["defaults"])
    try:
        for name in args.configs:
            if "," in name:
                for n in name.split(","):
                    conf.update(configs[n])
            else:
                conf.update(configs[name])
    except KeyError as e:
        print(f'Error: Section "{e.args[0]}" not found in YAML configuration files.')
        exit(1)

    # override yaml args
    for k, v in vars(args).items():
        if k == "configs" or v is None:
            continue
        conf[k] = v

    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = _strtobool
        parser.add_argument(f"--{key}", type=type_, default=value)
        # Allow --no-{key}  to remove a configuration value
        parser.add_argument(f"--no-{key}", dest=key, action="store_const", const=None)

    args = parser.parse_args(remaining)
    args.keep_empty = False
    args.rank = 0
    args.vocab_extra_ids = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.new_tokens = True

    return args


class DatasetWriter:
    def __init__(
        self,
        filename_prefix: str,
        vocab_size: int,
        dataset_impl: str = "mmap",
    ):
        self.bin_filename = f"{filename_prefix}-text.bin"
        self.idx_filename = f"{filename_prefix}-text.idx"
        self.builder = indexed_dataset.make_builder(
            self.bin_filename, impl=dataset_impl, vocab_size=vocab_size
        )

    def add_item(self, tokenized_item):
        self.builder.add_item(torch.IntTensor(tokenized_item))

    def finalize(self):
        self.builder.finalize(self.idx_filename)


def format_pairs(pairs: list[str]) -> list[str]:
    assert isinstance(pairs, list)
    roles = ("user", "assistant")
    return [
        f"<|im_start|>{roles[i%2]}\n{pairs[i]}<|im_end|>\n" for i in range(len(pairs))
    ]


def format_sft_entry(entry: DatasetEntrySft) -> list[str]:
    turns = []
    if entry.system_message and len(entry.system_message) > 0:
        turns.append(f"<|im_start|>system\n{entry.system_message}<|im_end|>\n")
    for m in entry.conversation:
        if m.role == Role.prompter:
            turns.append(f"<|im_start|>user\n{m.text}<|im_end|>\n")
        elif m.role == Role.assistant:
            turns.append(f"<|im_start|>assistant\n{m.text}<|im_end|>\n")
    return turns


def format_conversation(messages) -> str:
    if isinstance(messages, DatasetEntrySft):
        return format_sft_entry(messages)
    elif isinstance(messages, DatasetEntryLm):
        messages = messages.text
    else:
        messages = list(messages)
        return format_pairs(messages)


def tokenize_dataset(
    output_dir: Path,
    filename_prefix: str,
    dataset: Dataset,
    encoder: Encoder,
    dataset_impl: str,
    max_count: int | None = None,
):
    full_prefix = str(output_dir / filename_prefix)

    train_writer = DatasetWriter(
        filename_prefix=full_prefix,
        dataset_impl=dataset_impl,
        vocab_size=encoder.tokenizer.vocab_size,
    )

    for i, messages in enumerate(dataset):
        if max_count and i >= max_count:
            break
        turns = format_conversation(messages)
        text = "".join(turns)
        x = encoder.encode_text(text)
        train_writer.add_item(x)

    train_writer.finalize()


def main():
    args = parse_args()
    print("Configuration:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    train, val = get_dataset(args)

    # show dataset stats
    print("Training dataset sizes (before sampling):")
    total = len(train)
    for d in train.datasets:
        if isinstance(d, Subset):
            name = f"Subset of {type(d.dataset).__name__}"
            if hasattr(d.dataset, "name"):
                name += f" ({d.dataset.name})"
        else:
            name = type(d).__name__
            if hasattr(d, "name"):
                name += f" ({d.name})"
        print(f"{name}: {len(d)} ({len(d) / total:.2%})")

    print("Building encoder")
    encoder = Encoder(args)

    print(f"Vocab size: {encoder.tokenizer.vocab_size}")
    print(f"Output dir: {args.output_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    fn = output_dir / "special_tokens.json"
    with fn.open("w") as f:
        json.dump(encoder.special_tokens, f)

    for split_name, ds in zip(["train", "val"], [train, val]):
        tokenize_dataset(
            output_dir=output_dir,
            filename_prefix=f"{args.filename_prefix}-{split_name}",
            dataset=ds,
            encoder=encoder,
            dataset_impl=args.dataset_impl,
        )


if __name__ == "__main__":
    main()
