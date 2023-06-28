from pathlib import Path
from typing import Optional
from argparse import ArgumentParser

import torch
from tqdm.auto import trange
from transformers import AutoModelForCausalLM


def convert_to_megatron(weights: dict, size: int) -> dict:
    result = {}
    if size == 7:
        n_layer = 32
    else:
        n_layer = 60

    # weights independent of layers (i.e. token embeddings and layernorms
    assert torch.allclose(weights["lm_head.weight"],
                          weights["transformer.word_embeddings.weight"])
    result["model.language_model.embedding.word_embeddings.weight"] = \
        weights["transformer.word_embeddings.weight"]
    result["model.language_model.transformer.final_layernorm.weight"] = \
        weights["transformer.ln_f.weight"]

    for layer in trange(n_layer, desc="Converting weights"):
        prefix1 = f"model.language_model.transformer.layers.{layer}"
        prefix2 = f"transformer.h.{layer}"
        # mlp
        result[f"{prefix1}.mlp.dense_h_to_4h.weight"] = \
            weights[f"{prefix2}.mlp.dense_h_to_4h.weight"]
        result[f"{prefix1}.mlp.dense_4h_to_h.weight"] = \
            weights[f"{prefix2}.mlp.dense_4h_to_h.weight"]
        # layers - TODO: no post_attention_layernorm in HF weights
        result[f"{prefix1}.input_layer_norm.weight"] = \
            weights[f"{prefix2}.input_layer_norm.weight"]
        result[f"{prefix1}.input_layer_norm.bias"] = \
            weights[f"{prefix2}.input_layer_norm.bias"]
        # qkv weights
        result[f"{prefix1}.attention.query_key_value.weight"] = \
            weights[f"{prefix2}.self_attention.query_key_value.weight"]
    return result


def main(size: int = 7, out: Optional[Path] = None,
                 cache_dir: Optional[Path] = None):
    if out is None:
        out = Path(f"falcon{size}b_megatron.pt").absolute()

    # get weights from huggingface
    print("Fetching weights from huggingface/specified directory...")
    model = AutoModelForCausalLM.from_pretrained(f"tiiuae/falcon-{size}b",
                                                         trust_remote_code=True,
                                                                                                          cache_dir=cache_dir)
    hf_weights = model.load_state_dict()
    # convert state dict to be megatron-compatible
    megatron_weights = convert_to_megatron(hf_weights, size)
    # save converted weights in specified out
    torch.save(megatron_weights, out)


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert Huggingface falcon weights to "
                                                    "megatron-compatible weights")
    parser.add_argument("--size", default=7, choices={7, 40},
                                    help="The size of the falcon model (i.e. 7 or 40)")
    parser.add_argument("--out", type=Path,
                                    help="The filename to store the megatron weights")
    parser.add_argument("--cache-dir", type=Path,
                                    help="Directory to store the huggingface weights")
    args = parser.parse_args()
    main(args.size, args.out, args.cache_dir)
