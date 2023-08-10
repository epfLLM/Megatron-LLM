# import sys, os
import argparse
from pathlib import Path
import logging
import json

import torch
from torch.utils.data import Dataset, Subset, ConcatDataset


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


class DatasetWriter:
    def __init__(
        self,
        filename_prefix: str,
        vocab_size: int,
        dataset_impl: str = "mmap",
        feature: str = "text",
    ):
        self.bin_filename = f"{filename_prefix}-{feature}.bin"
        self.idx_filename = f"{filename_prefix}-{feature}.idx"
        self.builder = indexed_dataset.make_builder(self.bin_filename, impl=dataset_impl, vocab_size=vocab_size)

    def add_item(self, tokenized_item):
        self.builder.add_item(torch.IntTensor(tokenized_item))

    def finalize(self):
        self.builder.finalize(self.idx_filename)


def format_pairs(pairs: list[str] | tuple[str]) -> tuple[list[str], list[int]]:
    assert isinstance(pairs, list) or isinstance(pairs, tuple)
    role_names = ("user", "assistant")
    role_ids = (1, 2)
    return [f"<|im_start|>{role_names[i%2]}\n{pairs[i]}<|im_end|>\n" for i in range(len(pairs))], [
        role_ids[i % 2] for i in range(len(pairs))
    ]


def format_sft_entry(entry: DatasetEntrySft) -> tuple[list[str], list[int]]:
    turns = []
    roles = []
    if entry.system_message and len(entry.system_message) > 0:
        turns.append(f"<|im_start|>system\n{entry.system_message}<|im_end|>\n")
        roles.append(0)
    for m in entry.conversation:
        if m.role == Role.prompter:
            turns.append(f"<|im_start|>user\n{m.text}<|im_end|>\n")
            roles.append(1)
        elif m.role == Role.assistant:
            turns.append(f"<|im_start|>assistant\n{m.text}<|im_end|>\n")
            roles.append(2)
    return turns, roles


def format_conversation(messages) -> str:
    if isinstance(messages, DatasetEntrySft):
        return format_sft_entry(messages)
    elif isinstance(messages, DatasetEntryLm):
        return messages.text, [3]
    else:
        return format_pairs(messages)


def tokenize_dataset(
    output_dir: Path,
    filename_prefix: str,
    dataset: Dataset,
    encoder: Encoder,
    dataset_impl: str,
    max_count: int | None = None,
    check_tokenization: bool = True,
):
    full_prefix = str(output_dir / filename_prefix)

    token_writer = DatasetWriter(
        filename_prefix=full_prefix,
        dataset_impl=dataset_impl,
        vocab_size=encoder.tokenizer.vocab_size,
        feature="text",
    )

    role_writer = DatasetWriter(
        filename_prefix=full_prefix,
        dataset_impl=dataset_impl,
        vocab_size=16,
        feature="role",
    )

    jsonl_path = Path(full_prefix + ".jsonl")
    with jsonl_path.open("w", encoding="UTF-8") as jsonl_file:
        for i, messages in enumerate(dataset):
            if max_count and i >= max_count:
                break

            turns, turn_roles = format_conversation(messages)

            tokens = []
            role_lables = []
            for t, r in zip(turns, turn_roles):
                turn_tokens = encoder.encode_text(t)
                turn_role = [r] * len(turn_tokens)
                tokens.extend(turn_tokens)
                role_lables.extend(turn_role)

            if check_tokenization:
                x = encoder.encode_text("".join(turns))
                assert x == tokens and len(tokens) == len(role_lables)

            token_writer.add_item(tokens)
            role_writer.add_item(role_lables)

            json.dump({"text": "".join(turns)}, jsonl_file)
            jsonl_file.write("\n")

    token_writer.finalize()
    role_writer.finalize()


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
    parser.add_argument(
        "--max_count",
        type=int,
        help="Limit number of train/eval examples to process (debug)",
    )

    args = parser.parse_args(remaining)
    args.keep_empty = False
    args.rank = 0
    args.vocab_extra_ids = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.new_tokens = True

    return args


def main():
    """
    Example usage: `python pretokenize.py --output_dir output--configs llama_oasst_top1`
    """
    args = parse_args()
    print("Configuration:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    train, evals = get_dataset(args)

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

    output_dir = Path(args.output_dir)
    
    print(f"Vocab size: {encoder.tokenizer.vocab_size}")
    print(f"Output dir: {args.output_dir} (exists: {output_dir.exists()})")

    output_dir.mkdir(exist_ok=True)    

    fn = output_dir / "special_tokens.json"
    with fn.open("w") as f:
        json.dump(encoder.special_tokens, f)

    val = ConcatDataset(evals.values())
    for split_name, ds in zip(["train", "val"], [train, val]):
        tokenize_dataset(
            output_dir=output_dir,
            filename_prefix=f"{args.filename_prefix}-{split_name}",
            dataset=ds,
            encoder=encoder,
            dataset_impl=args.dataset_impl,
            max_count=args.max_count,
        )


if __name__ == "__main__":
    main()
