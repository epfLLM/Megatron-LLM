# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron global variables."""

import os
import sys
from collections import defaultdict

from megatron import dist_signal_handler
from megatron.tokenizer import build_tokenizer
from .microbatches import build_num_microbatches_calculator
from .timers import Timers

_GLOBAL_ARGS = None
_GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
_GLOBAL_TOKENIZER = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_ADLR_AUTORESUME = None
_GLOBAL_TIMERS = None
_GLOBAL_SIGNAL_HANDLER = None
_GLOBAL_COUNTERS = None


def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
    return _GLOBAL_ARGS


def get_num_microbatches():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get()


def get_current_global_batch_size():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get_current_global_batch_size()


def update_num_microbatches(consumed_samples, consistency_check=True):
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR.update(consumed_samples,
                                               consistency_check)


def get_tokenizer():
    """Return tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    return _GLOBAL_TOKENIZER


def get_tensorboard_writer():
    """Return our wrapped tensorboard/wandb writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_TENSORBOARD_WRITER


def get_adlr_autoresume():
    """ADLR autoresume object. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ADLR_AUTORESUME


def get_timers():
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, 'timers')
    return _GLOBAL_TIMERS


def get_counters():
    """Return counters."""
    _ensure_var_is_initialized(_GLOBAL_COUNTERS, 'counters')
    return _GLOBAL_COUNTERS


def get_signal_handler():
    _ensure_var_is_initialized(_GLOBAL_SIGNAL_HANDLER, 'signal handler')
    return _GLOBAL_SIGNAL_HANDLER


def _set_signal_handler():
    global _GLOBAL_SIGNAL_HANDLER
    _ensure_var_is_not_initialized(_GLOBAL_SIGNAL_HANDLER, 'signal handler')
    _GLOBAL_SIGNAL_HANDLER = dist_signal_handler.DistributedSignalHandler().__enter__()


def _set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def set_global_variables(args):
    """Set args, tokenizer, tensorboard_writer, adlr_autoresume, and timers."""
    assert args is not None
    _ensure_var_is_not_initialized(_GLOBAL_ARGS, 'args')
    _set_args(args)

    _build_num_microbatches_calculator(args)
    if args.vocab_file or args.tokenizer_type in ["FalconTokenizer", "LlamaTokenizer", "GPT2BPETokenizer"]:
        _ = _build_tokenizer(args)
    _set_tensorboard_writer(args)
    _set_adlr_autoresume(args)
    _set_timers(args)
    _set_counters(args)

    if args.exit_signal_handler:
        _set_signal_handler()


def _build_num_microbatches_calculator(args):
    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    _ensure_var_is_not_initialized(_GLOBAL_NUM_MICROBATCHES_CALCULATOR,
                                   'num microbatches calculator')
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = build_num_microbatches_calculator(args)


def _build_tokenizer(args):
    """Initialize tokenizer."""
    global _GLOBAL_TOKENIZER
    _ensure_var_is_not_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    _GLOBAL_TOKENIZER = build_tokenizer(args)
    return _GLOBAL_TOKENIZER


def rebuild_tokenizer(args):
    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = None
    return _build_tokenizer(args)


def _set_tensorboard_writer(args):
    """Set our wrapped tensorboard/wandb writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER,
                                   'tensorboard writer')

    if getattr(args,"wandb_logger",False):
        """
        if this arg is set to True, we check the other wandb relevant arguments and
        return a shim which exposes the wandb logging via a tensorboard-y API
        """
        if args.rank == (args.world_size - 1):
            try:
                from megatron.wandb_logger import WandBConfig,WandbTBShim
                cfg=WandBConfig.from_args(args) 
                shim=WandbTBShim(cfg)
                print('> setting wandb ...')
                _GLOBAL_TENSORBOARD_WRITER=shim
            except ModuleNotFoundError:
                print('WARNING: WanDB writing requested but is not '
                      'available, '
                      'no WandB logs will be written.', flush=True)
    else:
        if hasattr(args, 'tensorboard_dir') and \
           args.tensorboard_dir and args.rank == (args.world_size - 1):
            try:
                from torch.utils.tensorboard import SummaryWriter
                print('> setting tensorboard ...')
                _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                    log_dir=args.tensorboard_dir,
                    max_queue=args.tensorboard_queue_size)
            except ModuleNotFoundError:
                print('WARNING: TensorBoard writing requested but is not '
                      'available (are you using PyTorch 1.1.0 or later?), '
                      'no TensorBoard logs will be written.', flush=True)


def _set_adlr_autoresume(args):
    """Initialize ADLR autoresume."""
    global _GLOBAL_ADLR_AUTORESUME
    _ensure_var_is_not_initialized(_GLOBAL_ADLR_AUTORESUME, 'adlr autoresume')

    if args.adlr_autoresume:
        if args.rank == 0:
            print('enabling autoresume ...', flush=True)
        sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))
        try:
            from userlib.auto_resume import AutoResume
        except BaseException:
            print('ADLR autoresume is not available, exiting ...')
            sys.exit()

        _GLOBAL_ADLR_AUTORESUME = AutoResume


def _set_timers(args):
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, 'timers')
    _GLOBAL_TIMERS = Timers(args.timing_log_level, args.timing_log_option)


def _set_counters(args):
    global _GLOBAL_COUNTERS
    _ensure_var_is_not_initialized(_GLOBAL_COUNTERS, 'counters')
    _GLOBAL_COUNTERS = defaultdict(int)


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)




