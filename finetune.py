"""Fine-tune gpt, llama or falcon"""

import datetime as dt
from functools import partial

import torch

from megatron import get_args, get_tokenizer, get_timers, get_counters, print_rank_0
from megatron.training import pretrain
from megatron.core import tensor_parallel
from megatron.core.parallel_state import get_data_parallel_group
from megatron.model import GPTModel, ModelType, LlamaModel, FalconModel, MistralModel
from megatron.utils import get_ltor_masks_and_position_ids, average_losses_across_data_parallel_group
from megatron.data.gpt_dataset import build_train_valid_test_datasets as gpt_build_datasets
from megatron.data.instruction_dataset import instruction_collator
from megatron.data.instruction_dataset import build_train_valid_test_datasets as instruct_build_datasets
from megatron.initialize import initialize_megatron
from megatron.metrics import MetricInput, get_metric


##
# Model provider utilities
##


def model_provider(pre_process: bool = True, post_process: bool = True):
    """Build the model."""

    print_rank_0("Building model ...")

    args = get_args()
    if args.model_name == "gpt": 
        cls = GPTModel
    elif args.model_name == "falcon":
        cls = FalconModel
    elif args.model_name in {"llama", "llama2", "codellama"}:
        cls = partial(LlamaModel, version=1 if args.model_name == "llama" else 2)
    elif args.model_name == "mistral":
        cls = MistralModel
        if args.sliding_window_size != 4096:
            print_rank_0("Mistral uses sliding window attention (set sliding_window=4096)")
            args.sliding_window_size = 4096
    else:
        raise KeyError(f"Unkown model {args.model_name}")

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
        model_type=model_type
    )
    return model


##
# Dataset utilities
##

# Heavily inspired by Andreas Köpf: https://github.com/andreaskoepf/epfl-megatron/tree/local_changes/
def get_attention_mask_and_position_ids(data, attention_mask):
    """Build causal attention masks and position id for left to right model.
    Builds a (batch, 1, seq, seq)-sized binary causal attention mask from
    a (batch, seq)-sized attention mask specifying.
    If any value in the input attention_mask is < 0.5, the output
    attention mask will mask this position for every token, i.e. out[i, 0, :, j] = True
    if in[i, j] < 0.5.
    Returns attention_mask, position_ids"""

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
    datatype = torch.int64
    if args.data_type == "gpt":
        keys = ["text"]
    elif args.data_type == "instruction":
        keys = ["text", "attention_mask", "assistant_mask", "pad_mask"]
    else:
        raise KeyError(f"Unknown dataset type {args.data_type}")

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens = data_b["text"]
    labels = tokens[:, 1:].contiguous()
    tokens = tokens[:, :-1].contiguous()

    # Update tokens counter.
    counters = get_counters()
    n_tokens = torch.tensor(tokens.numel(), device=tokens.device)
    if args.data_parallel_size == 1:
        n_tokens = n_tokens.item()
    else:
        group = get_data_parallel_group()
        torch.distributed.all_reduce(
            n_tokens, op=torch.distributed.ReduceOp.SUM, group=group
        )
        n_tokens = n_tokens.item()
    counters["tokens"] += n_tokens

    if args.data_type == "gpt":
        # Get the masks and position ids.
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss
        )
        return tokens, labels, loss_mask, attention_mask, position_ids

    # Instruction dataset.
    # Heavily inspired by Andreas Köpf: https://github.com/andreaskoepf/epfl-megatron/tree/local_changes/
    attention_mask = data_b["attention_mask"][:, :-1]
    assistant_mask = data_b["assistant_mask"][:, 1:].to(tokens.device)
    pad_mask = data_b["pad_mask"][:, 1:].to(tokens.device)
    loss_mask = torch.full(labels.size(), args.scalar_loss_mask, dtype=torch.float,
                           device=tokens.device)
    loss_mask[assistant_mask == 1] = 1.0
    loss_mask[pad_mask == 1] = 0.0
    attention_mask, position_ids = get_attention_mask_and_position_ids(
        tokens, attention_mask
    )

    return tokens, labels, loss_mask, attention_mask, position_ids


def data_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    if args.data_type == "gpt":
        builder = gpt_build_datasets
    elif args.data_type == "instruction":
        builder = instruct_build_datasets

    print_rank_0("> building train, validation, and test datasets ...")
    train_ds, valid_ds, test_ds = builder(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path
    )
    print_rank_0("> finished creating datasets ...")

    return train_ds, valid_ds, test_ds


##
# Loss and forward
##


def loss_func(is_training, batch, outputs):
    loss_mask = batch[2]
    losses, logits = outputs
    losses = losses.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    out_dict = {"lm loss": averaged_loss[0]}

    # Calculate other metrics
    if not is_training:
        inputs = MetricInput(batch, logits, averaged_loss[0])
        args = get_args()
        for metric in map(get_metric, args.metrics):
            out_dict.update(metric(inputs))

    return loss, out_dict


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    batch = get_batch(data_iterator)
    tokens, labels, loss_mask, attention_mask, position_ids = batch
    timers("batch-generator").stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)
    return output_tensor, partial(loss_func, model.training, batch)


##
# Main
##


def extra_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='validation set')
    group.add_argument("--model_name",
                       choices={"gpt", "llama", "falcon", "llama2", "codellama", "mistral"},
                       default="gpt")
    group.add_argument("--model_type", choices={"encoder_or_decoder", "encoder_and_decoder"},
                       default="encoder_or_decoder")
    group.add_argument("--data_type", choices={"gpt", "instruction"},
                       default="gpt")
    group.add_argument("--log_learning_rate_to_tensorboard", type=bool, default=True)
    group.add_argument("--log_loss_scale_to_tensorboard", type=bool, default=True)
    return parser


if __name__ == "__main__":
    args_defaults = {"tokenizer_type": "GPT2BPETokenizer"}
    initialize_megatron(extra_args, args_defaults)
    args = get_args()

    if args.data_type == "gpt":
        collate_fn = None
    else:
        collate_fn = instruction_collator


    pretrain(args, data_provider, model_provider,  ModelType.encoder_or_decoder,
             forward_step, collate_fn=collate_fn)
    print(f"Done {dt.datetime.now(dt.timezone.utc)}")
