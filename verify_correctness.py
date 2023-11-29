import os
import json
import warnings
from pathlib import Path
from typing import Optional

import torch
import llama
from torch import nn
from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, MistralForCausalLM
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from megatron import get_args, update_num_microbatches
from megatron.arguments import parse_args
from megatron.initialize import initialize_megatron, set_jit_fusion_options
from megatron.training import _setup_model_and_optimizer, build_train_valid_test_data_iterators

from finetune import model_provider, extra_args, get_batch, loss_func, data_provider


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


def is_meta_llama2_path(path: Optional[Path]) -> bool:
    return path is not None and len(list(path.glob("*.pth"))) > 0


def hf_provider(name: str, cache_dir: Optional[Path], device: str,
                size: int = 7, bf16: bool = False):
    print("Getting huggingface model...")
    extra_kwargs = {}
    if bf16:
        extra_kwargs = {"torch_dtype": torch.bfloat16}
    if name == "falcon":
        model = AutoModelForCausalLM.from_pretrained(
            f"tiiuae/falcon-{size}b", cache_dir=cache_dir,
            trust_remote_code=True,
            **extra_kwargs
        )
    elif name == "llama":
        try:
            model = LlamaForCausalLM.from_pretrained(cache_dir, **extra_kwargs)
        except OSError:
            print(f"Cache dir {cache_dir} does not look like a huggingface "
                  "checkpoint, assuming cache_dir instead")
            model = LlamaForCausalLM.from_pretrained(
                f"decapoda-research/llama-{size}b-hf", cache_dir=cache_dir,
                **extra_kwargs
            )
    elif name == "llama2" and is_meta_llama2_path(cache_dir):
        print(f"baseline path {cache_dir} does not look like a huggingface, "
              "assuming it's raw llama2 weights instead")
        model = Llama2Wrapper(cache_dir)
    elif name == "llama2":
        model = LlamaForCausalLM.from_pretrained(
            f"meta-llama/Llama-2-{size}b-hf", cache_dir=cache_dir,
            **extra_kwargs
        )
    elif name == "mistral":
        assert size == 7, "Mistral only supports 7B model"
        try:
            model = MistralForCausalLM.from_pretrained(cache_dir, **extra_kwargs)
        except OSError:
            print(f"Cache dir {cache_dir} does not look like a huggingface "
                  "checkpoint, assuming cache_dir instead")
            model = MistralForCausalLM.from_pretrained(
                f"mistralai/Mistral-{size}B-v0.1", cache_dir=cache_dir,
                **extra_kwargs
            )
    else:
        raise KeyError(f"Model {name} not implemented")
    return model.eval().requires_grad_(False).to(device)


def hf_our_provider(name: str, data_dir: Path, device: str, size: int = 7):
    if name in {"llama", "llama2"}:
        model = LlamaForCausalLM.from_pretrained(data_dir)
    else:
        raise NotImplementedError("Testing custom checkpoints supported for llama")
    return model.eval().requires_grad_(False).to(device)


def hf_forward(model, batch):
    device = next(param.device for param in model.parameters())
    batch = [tensor.to(device) for tensor in batch]
    tokens, labels, loss_mask, attention_mask, position_ids = batch
    output = model(input_ids=tokens, position_ids=position_ids, labels=tokens)
    return output["logits"], output["loss"]


def mega_provider(name: str):
    print("Getting megatron model...")
    model, _ , _ = _setup_model_and_optimizer(model_provider, name, args=get_args())
    assert len(model) == 1, "correctness verification only supported with unsharded models"
    model = model[0].eval().requires_grad_(False)
    return model


def mega_forward(model, batch):
    tokens, labels, loss_mask, attention_mask, position_ids = batch
    assert torch.all(loss_mask)
    # we need to do two forward passes to get both the logits and the loss
    _, logits = out = model(tokens, position_ids, attention_mask, labels=labels)
    loss, _ = loss_func(model.training, batch, out)
    return logits, loss


def verify_step(our_forward, our_model, base_forward, base_model, batch):
    our_logits, our_loss = our_forward(our_model, batch)
    base_logits, base_loss = base_forward(base_model, batch)
    assert our_logits.size() == base_logits.size(), \
            f"ours={our_logits.size()}, true={base_logits.size()}"
    our_logits = our_logits.cpu()
    base_logits = base_logits.cpu()
    abs_error = torch.abs(our_logits - base_logits)
    print("Max absoulute error in the logits:",
          f"max={torch.max(abs_error):.6f}, avg={torch.mean(abs_error):.6f}")
    assert our_loss.size() == base_loss.size()
    our_loss = our_loss.cpu()
    base_loss = base_loss.cpu()
    loss_error = torch.abs(our_loss - base_loss)
    print(f"Abs loss error: {loss_error:.6f} "
          f"Our loss: {our_loss:.3f}, theirs: {base_loss:.3f}")


def is_megatron_path(path: Path | str):
    path = Path(path) if isinstance(path, str) else path
    return (path/"latest_checkpointed_iteration.txt").exists()


def main():
    # Misc initializations
    print("Starting megatron vs huggingface verification")
    args = get_args()
    set_jit_fusion_options(args)

    # Determine if the provided weight is a megatron checkpoint or huggingface checkpoint
    print("Loading our model!")
    if is_megatron_path(args.load):
        our_model = mega_provider(args.model_name)
        our_forward = mega_forward
    else:
        print("NOTE: The given path does not look like a megatron checkpoint, "
              f"assuming it's a huggingface checkpoint instead (path={args.load})")
        our_model = hf_our_provider(args.model_name, args.load, "cuda:0", bf16=args.bf16)
        our_forward = hf_forward
        args.iteration = 0

    # Load baseline model
    print("Loading baseline model!")
    base_model = hf_provider(args.model_name, args.cache_dir,
                             args.baseline_device, size=args.model_size)
    base_forward = hf_forward

    # Load dataset iterator
    print("Loading dataset!")
    data_iterator, _, _ = build_train_valid_test_data_iterators(
        data_provider, args
    )

    # Now we can start the verifications
    for iteration in range(0, 10):
        print(f"Iteration {iteration}...")
        update_num_microbatches(args.consumed_train_samples)
        args.curr_iteration = iteration
        verify_step(our_forward, our_model, base_forward, base_model,
                    get_batch(data_iterator))



def extra_extra_args(parser):
    parser = extra_args(parser)
    group = parser.add_argument_group(title="huggingface")
    group.add_argument("--huggingface_cache", type=Path, default=None, dest="cache_dir", help=(
        "If falcon, optional: path to huggingface cache. "
        "If llama2, optional: either the huggingface cache path, or "
        "the raw weight directory given by meta. "
        "If llama, optional: either the path to converted huggingface weights "
        "(use convert_llama_weights_to_hf.py) or the huggingface cache dir."
    ))
    group.add_argument("--huggingface_device", default="cuda:1", dest="baseline_device",
                       help="Device to use for the baseline model")
    group.add_argument("--model_size", type=int, default=7)
    return parser


if __name__ == "__main__":
    defaults = {"micro_batch_size": 1, "use_checkpoint_args": True, "train_iters": 10,
                "lr": 1.0}
    # if not is_megatron_path(parse_args(extra_extra_args).load):
    #     defaults.update({"encoder_num_layers": 1, "hidden_size": 1, 
    #                      "num_attention_heads": 1, "seq_length": 2048,
    #                      "max_position_embeddings": 2048})
    initialize_megatron(extra_extra_args, args_defaults=defaults)
    main()
