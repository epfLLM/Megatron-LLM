import json
import warnings
from pathlib import Path

import torch
import llama
from torch import nn
from transformers import AutoModelForCausalLM, LlamaForCausalLM
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

import megatron
from megatron import get_args, print_rank_0, update_num_microbatches
from megatron.model.enums import ModelType
from megatron.initialize import initialize_megatron
from megatron.training import _setup_model_and_optimizer, build_train_valid_test_data_iterators

from finetune import model_provider, extra_args, get_batch, loss_func, train_valid_test_datasets_provider


class Llama2Wrapper(nn.Module):
    def __init__(self, cache_dir):
        super().__init__()
        initialize_model_parallel(1)
        cache_dir = Path(cache_dir)
        checkpoints = sorted(cache_dir.glob("*.pth"))
        assert len(checkpoints) == 1, "Currently, only llama2 unsharded models implemented"
        with open(cache_dir/"params.json", "r") as f:
            params = json.loads(f.read())
            params["vocab_size"] = 32000

        self.model = llama.Transformer(llama.ModelArgs(
            max_seq_len=4096, max_batch_size=1, **params
        ))
        self.model.load_state_dict(torch.load(checkpoints[0]), strict=False)

    def forward(self, input_ids, position_ids=None, attention_mask=None,
                labels=None):
        if labels is not None:
            warnings.warn("Llama2 does not compute loss")
        logits = self.model(input_ids, 0)
        loss = torch.tensor(0.0).to(logits.device, logits.dtype)
        return {"logits": logits, "loss": loss}


def hf_provider():
    args = get_args()
    print_rank_0("Getting huggingface model...")
    if args.model_name == "falcon":
        model = AutoModelForCausalLM.from_pretrained(
            f"tiiuae/falcon-{args.model_size}b", cache_dir=args.huggingface_cache,
            trust_remote_code=True
        )
    elif args.model_name == "llama":
        model = LlamaForCausalLM.from_pretrained(args.huggingface_cache).float()
    elif args.model_name == "llama2":
        model = Llama2Wrapper(args.huggingface_cache).float()
    else:
        raise KeyError(f"Model {args.model_name} not implemented")
    model = model.eval().requires_grad_(False).to(args.huggingface_device)
    return model


def megatron_forward(model, batch):
    model = model[0]  # TODO: asuming no model parallelism
    tokens, labels, loss_mask, attention_mask, position_ids = batch
    assert torch.all(loss_mask)
    # we need to do two forward passes to get both the logits and the loss
    logits = model(tokens, position_ids, attention_mask, labels=None)
    losses = model(tokens, position_ids, attention_mask, labels=labels)
    loss, _ = loss_func(loss_mask, losses)
    return logits, loss


def huggingface_forward(model, batch):
    args = get_args()
    batch = [tensor.to(args.huggingface_device) for tensor in batch]
    tokens, labels, loss_mask, attention_mask, position_ids = batch
    output = model(input_ids=tokens, position_ids=position_ids, labels=tokens)
    return output["logits"], output["loss"]


def verify_step(forward1, model1, forward2, model2, iterator):
    batch = get_batch(iterator)
    logits1, loss1 = forward1(model1, batch)
    logits2, loss2 = forward2(model2, batch)
    assert logits1.size() == logits2.size(), \
            f"ours={logits1.size()}, true={logits2.size()}"
    logits1 = logits1.cpu()
    logits2 = logits2.cpu()
    abs_error = torch.max(torch.abs(logits1 - logits2))
    print(f"Max absoulute error in the logits: {abs_error:.6f}")
    assert loss1.size() == loss2.size()
    loss1 = loss1.cpu()
    loss2 = loss2.cpu()
    loss_error = torch.abs(loss1 - loss2)
    print(f"Abs loss error: {loss_error:.6f} "
          f"Our loss: {loss1:.3f}, theirs: {loss2:.3f}")


# heavily inspired from megatron.training.pretrain
def verify(args, train_valid_test_dataset_provider,
           model_provider_func, forward_step_func,
           baseline_provider_func, forward_baseline_func,
           model_type, process_non_loss_data_func=None):
    """Verifies that the `forward_step_func` gives forward outputs similar
    to the `forward_baseline_func` using the dataset provider specified"""

    # MISC INITALIZATIONS
    megatron.initialize.set_jit_fusion_options(args)
    print_rank_0("Megatron has been initialized!")
    model, optimizer, opt_param_schedulkr = _setup_model_and_optimizer(
        model_provider_func, model_type, args=args
    )
    for module in model:
        module.eval().requires_grad_(False)
    baseline_model = baseline_provider_func()
    print('==== Megatron model ====')
    print(model)
    print()
    print('==== Huggingface model ====')
    print(baseline_model)
    print_rank_0("Model has been setup")
    if args.virtual_pipeline_model_parallel_size is not None:
        all_data_iterators = [
            build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider, args)
            for _ in range(len(model))
        ]
        train_data_iterator = [di[0] for di in all_data_iterators]
        valid_data_iterator = [di[1] for di in all_data_iterators]
        test_data_iterator = [di[2] for di in all_data_iterators]
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator \
            = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider, args)
    print_rank_0("Dataloaders have been built!")

    # Now we can start the verifications
    print_rank_0("Starting verifications!")
    for iteration in range(0, 10):
        print_rank_0(f"Iteration {iteration}...")
        update_num_microbatches(args.consumed_train_samples)
        args.curr_iteration = iteration
        verify_step(forward_step_func, model,
                    forward_baseline_func, baseline_model, train_data_iterator)


def extra_extra_args(parser):
    parser = extra_args(parser)
    group = parser.add_argument_group(title="huggingface")
    group.add_argument("--huggingface_cache", default=None)
    group.add_argument("--huggingface_device", default="cuda:0")
    group.add_argument("--model_size", type=int, default=7)
    group.add_argument("--hf_weights", help="Path to llama weights")
    return parser


if __name__ == "__main__":
    # INITIALIZATION
    print("Starting megatron vs huggingface verification")
    defaults = {"micro_batch_size": 1, "use_checkpoint_args": True, "train_iters": 10,
                "lr": 1.0}
    initialize_megatron(extra_extra_args, args_defaults=defaults)
    args = get_args()

    # VERIFICATION
    verify(args, train_valid_test_datasets_provider,
           model_provider, megatron_forward,
           hf_provider, huggingface_forward,
           ModelType.encoder_or_decoder,)
    print("Verification done, we are set now woohoo! :)")
