# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.


"""Fine-tune GPT"""

from functools import partial
import os
import sys
import datetime as dt
from typing import List

import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))

import megatron
from megatron import get_timers
from megatron import get_tokenizer
from megatron import print_rank_0
from megatron.core import mpu

import megatron.data.gpt_dataset
from megatron.model import LlamaModel
from megatron.model.enums import ModelType
import megatron.training
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group


def _model_provider(pre_process: bool,
                    post_process: bool):
    args = megatron.get_args()
    """Build the model."""
    print_rank_0('building Llama model ...')
    # pre_process = False
    parallel_output = True
    num_tokentypes = 0
    model = LlamaModel(
        num_tokentypes,
        parallel_output,
        pre_process,
        post_process,
        args
    )
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = megatron.get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {'lm loss': averaged_loss[0]}


def _forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    return output_tensor, partial(loss_func, loss_mask)


def _train_valid_test_datasets_provider(train_val_test_num_samples: List[int]):
    """Build train, valid, and test datasets."""
    args = megatron.get_args()
    model_name = "llama"
    assert args.data_path, "Not supporting None data_path"
    print_rank_0(f'> building train, validation, and test datasets for {model_name} ...')
    train_ds, valid_ds1, test_ds = megatron.data.gpt_dataset.build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0(f"> finished creating finetuning {model_name} datasets ...")

    _, valid_ds, _ = megatron.data.gpt_dataset.build_train_valid_test_datasets(
        data_prefix=args.data_path2,
        data_impl="mmap",
        splits_string="98,2,0",
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=4,
        seed=1234,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0(f"> finished creating pretrained {model_name} datasets ...")
    return train_ds, valid_ds, test_ds


def add_validation_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='validation set')
    group.add_argument('--data_path2', nargs='*', default=None,
                       help='Path to the validation dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')
    group.add_argument('--eval_ppl', action='store_true', default=False)
    group.add_argument('--stored_params', type=dict, default=dict())
    # group.add_argument('--padded_vocab_size', type=int, default=100)
    return parser


if __name__ == "__main__":
    megatron.training.pretrain(_train_valid_test_datasets_provider,
                               _model_provider,
                               ModelType.encoder_or_decoder,
                               _forward_step,
                               args_defaults={'tokenizer_type':
                                                  'GPT2BPETokenizer'},
                               extra_args_provider=add_validation_args)

    print(f"done {dt.datetime.now(dt.timezone.utc)}")
