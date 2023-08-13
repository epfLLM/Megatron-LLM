"""Fine-tune gpt, llama or falcon"""

import datetime as dt
from functools import partial

import torch

from megatron import get_args, get_tokenizer, get_timers, print_rank_0
from megatron.training import pretrain
from megatron.core import tensor_parallel
from megatron.model import GPTModel, ModelType, LlamaModel, FalconModel
from megatron.utils import (
    get_ltor_masks_and_position_ids,
    average_losses_across_data_parallel_group,
)
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.data.instruction_dataset import (
    build_train_valid_test_datasets as build_instruction_datasets,
)
from megatron.initialize import initialize_megatron


def model_provider(pre_process: bool = True, post_process: bool = True):
    """Build the model."""

    print_rank_0("Building model ...")

    args = get_args()
    if args.model_name == "gpt":
        cls = GPTModel
    elif args.model_name == "falcon":
        cls = FalconModel
    elif args.model_name in {"llama", "llama2"}:
        cls = partial(LlamaModel, version=1 if args.model_name == "llama" else 2)
    else:
        raise KeyError(f"Unkown model {other}")

    if isinstance(args.model_type, ModelType):
        model_type = args.model_type
    elif args.model_type == "encoder_or_decoder":
        model_type = ModelType.encoder_or_decoder
    elif args.model_type == "encoder_and_decoder":
        model_type = ModelType.encoder_and_decoder
    else:
        raise KeyError(f"Unsupported model_type {args.model_type}")

    model = cls(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process,
        model_type=model_type,
    )
    return model


def get_batch_old(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ["text"]
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b["text"].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
    )

    return tokens, labels, loss_mask, attention_mask, position_ids


def get_attention_mask_and_position_ids(data, attention_mask):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    att_mask_batch = micro_batch_size
    attention_mask = (
        attention_mask.unsqueeze(1)
        .expand(micro_batch_size, seq_length, seq_length)
        .to(data.device)
    )
    attention_mask = torch.tril(attention_mask).view(
        att_mask_batch, 1, seq_length, seq_length
    )

    # Convert attention mask to binary, True entries will masked
    attention_mask = attention_mask < 0.5

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)

    return attention_mask, position_ids


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ["text", "attention_mask", "loss_mask"]
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens = data_b["text"]
    labels = torch.roll(tokens, shifts=-1, dims=-1)

    attention_mask = data_b["attention_mask"]

    loss_mask = data_b["loss_mask"]
    loss_mask[:, 0] = 0  # first tokens are no targets due to roll
    loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
    loss_mask = loss_mask.float().to(tokens.device)

    # Get the masks and postition ids.
    attention_mask, position_ids = get_attention_mask_and_position_ids(
        tokens, attention_mask
    )

    return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    # batch_fill_rate = loss_mask.sum()/losses.nelement()
    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {"lm loss": averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers("batch-generator").stop()

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    return output_tensor, partial(loss_func, loss_mask)


def eval_loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {"lm loss": averaged_loss[0]}


def eval_forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers("batch-generator").stop()

    # logits = model(tokens, position_ids, attention_mask)

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    return output_tensor, partial(eval_loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0("> building train, validation, and test datasets " "for GPT ...")
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path,
    )
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def instruction_train_valid_test_datasets_provider(train_val_test_num_samples):
    args = get_args()

    print_rank_0(
        "> building train, validation, and test datasets " "for instruction tuning ..."
    )
    train_ds, valid_ds, test_ds = build_instruction_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path,
    )
    print_rank_0("> finished creating Instruction datasets ...")

    return train_ds, valid_ds, test_ds


def extra_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title="validation set")
    group.add_argument(
        "--model_name", choices={"gpt", "llama", "falcon", "llama2"}, default="gpt"
    )
    group.add_argument(
        "--model_type",
        choices={"encoder_or_decoder", "encoder_and_decoder"},
        default="encoder_or_decoder",
    )
    group.add_argument("--log_learning_rate_to_tensorboard", type=bool, default=True)
    group.add_argument("--log_loss_scale_to_tensorboard", type=bool, default=True)
    group.add_argument("--variable_seq_lengths", type=bool, default=True)
    return parser


def round_to_multiple_of(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def instruction_collator(args, data):
    tokenizer = get_tokenizer()
    pad_id = tokenizer.pad
    seq_len = args.seq_length

    if args.variable_seq_lengths:
        max_sample_length = max(len(x["text"]) for x in data)
        seq_len = min(args.seq_length, round_to_multiple_of(max_sample_length, 16))

    # pad data to seq_len, create attention mask
    batch_size = len(data)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    role = torch.full_like(attention_mask, -1)
    input = torch.full_like(attention_mask, pad_id)

    for i, x in enumerate(data):
        t = x["text"]
        r = x["role"]
        l = len(t)

        if l < seq_len:
            attention_mask[i, l:] = 0
            input[i, :l] = torch.from_numpy(t)
            role[i, :l] = torch.from_numpy(r)
        else:
            input[i] = torch.from_numpy(t[:seq_len])
            role[i] = torch.from_numpy(r[:seq_len])

    loss_mask = role.eq(2).long()  # assistant tokens have role == 2

    return {"text": input, "attention_mask": attention_mask, "loss_mask": loss_mask}


if __name__ == "__main__":
    args_defaults = {"tokenizer_type": "GPT2BPETokenizer"}
    initialize_megatron(extra_args, args_defaults)
    args = get_args()
    pretrain(
        args,
        instruction_train_valid_test_datasets_provider,
        # train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        eval_forward_step_func=eval_forward_step,
        collate_fn=partial(instruction_collator, args),
    )
    print(f"Done {dt.datetime.now(dt.timezone.utc)}")
