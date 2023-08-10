import os
import sys
from pathlib import Path
from typing import Optional
from argparse import ArgumentParser, Namespace

import torch
from tqdm.auto import trange
from transformers import AutoModelForCausalLM

from permute_qkv import permute_qkv
from merge_llama import merge_llama


llama_s2layer = {7: 32, 13: 40, 30: 60, 65: 80, 70: 80}
llama_s2heads = {7: 32, 13: 40, 30: 52, 65: 64, 70: 64}
llama_s2dense = {7: 11008, 13: 13824, 30: 17920, 65: 22016,
                 70: 28672}  # should be (2/3)*4*d, but it isn't exaclty that
llama_s2hidden = {7: 4096, 13: 5120, 32: 6656, 65: 8192, 70: 8192}


def falcon_to_megatron(weights: dict, size: int) -> dict:
    def permute(qkv_w):
        return permute_qkv(qkv_w, dim, n_heads, n_heads_kv)

    embedding = {}
    transformer = {}
    if size == 7:
        n_layer = 32
        dim = 4544
        n_heads = 71
        n_heads_kv = 1
    else:
        n_layer = 60
        dim = 8192
        n_heads = 128
        n_heads_kv = 8

    # weights independent of layers (i.e. token embeddings and layernorms
    assert torch.allclose(weights["lm_head.weight"],
                          weights["transformer.word_embeddings.weight"])
    embedding["word_embeddings.weight"] = weights["transformer.word_embeddings.weight"]
    transformer["final_layernorm.weight"] = weights["transformer.ln_f.weight"]
    transformer["final_layernorm.bias"] = weights["transformer.ln_f.bias"]

    # copy weights for each transformer layer
    for layer in trange(n_layer, desc="Converting weights"):
        prefix1 = f"layers.{layer}"
        prefix2 = f"transformer.h.{layer}"
        # mlp
        transformer[f"{prefix1}.mlp.dense_h_to_4h.weight"] = \
            weights[f"{prefix2}.mlp.dense_h_to_4h.weight"]
        transformer[f"{prefix1}.mlp.dense_4h_to_h.weight"] = \
            weights[f"{prefix2}.mlp.dense_4h_to_h.weight"]
        # qkv weights
        transformer[f"{prefix1}.attention.query_key_value.weight"] = \
            permute(weights[f"{prefix2}.self_attention.query_key_value.weight"])
        # dense
        transformer[f"{prefix1}.self_attention.dense.weight"] = \
            weights[f"{prefix2}.self_attention.dense.weight"]
        # falcon7 and falcon40 differ in the input layernorms
        if size == 7:
            transformer[f"{prefix1}.input_layernorm.weight"] = \
                weights[f"{prefix2}.input_layernorm.weight"]
            transformer[f"{prefix1}.input_layernorm.bias"] = \
                weights[f"{prefix2}.input_layernorm.bias"]
        else:
            transformer[f"{prefix1}.input_layernorm.weight"] = \
                weights[f"{prefix2}.ln_attn.weight"]
            transformer[f"{prefix1}.mlp_layernorm.weight"] = \
                weights[f"{prefix2}.ln_mlp.weight"]
            transformer[f"{prefix1}.input_layernorm.bias"] = \
                weights[f"{prefix2}.ln_attn.bias"]
            transformer[f"{prefix1}.mlp_layernorm.bias"] = \
                weights[f"{prefix2}.ln_mlp.bias"]
    return {"embedding": embedding, "transformer": transformer}


def llama_to_megatron(llama_config: dict, size: int, version: int = 1):
    def get_wqkv(llama_config, layer_prefix, n_heads=32):
        wq, wk, wv = llama_config[layer_prefix+'attention.wq.weight'], llama_config[layer_prefix+'attention.wk.weight'], llama_config[layer_prefix+'attention.wv.weight']
        n_hidden_per_head = wq.shape[-1] // n_heads
        if version == 1 or size <= 13:
            n_kv_heads = n_heads
        else:
            n_kv_heads = 8

        dim = wq.shape[-1]
        # wq = wq.view(n_heads, n_hidden_per_head//2, 2, dim).transpose(1, 2).reshape(dim, dim)
        # wk = wk.view(n_kv_heads, n_hidden_per_head//2, 2, dim).transpose(1, 2).reshape(n_kv_heads*n_hidden_per_head, dim)

        wq_convert = torch.split(wq, n_hidden_per_head, dim=0)
        wk_convert = torch.split(wk, n_hidden_per_head, dim=0)
        wv_convert = torch.split(wv, n_hidden_per_head, dim=0)
        assert len(wq_convert) == n_heads
        assert len(wk_convert) == n_kv_heads
        assert len(wv_convert) == n_kv_heads

    
        w_qkv = []
        n_qs_per_kv = n_heads//n_kv_heads
        for i in range(n_kv_heads):
            w_qkv += [wq_convert[i*n_qs_per_kv + j] for j in range(n_qs_per_kv)]
            w_qkv += [wk_convert[i], wv_convert[i]]
        out = torch.concat(w_qkv, dim=0)
        # out = permute_qkv(out, dim, n_heads, n_kv_heads)
        return out

    # dictionary
    scale2layer = {f"{size_}B": val for size_, val in llama_s2layer.items()}
    scale2heads = {f"{size_}B": val for size_, val in llama_s2heads.items()}
    megatron2llama = {
        'attention.query_key_value': ['attention.wq', 'attention.wk', 'attention.wv'],
        'attention.dense': ['attention.wo'],
        'post_attention_layernorm': ['ffn_norm'],
        'input_layernorm': ['attention_norm'],
        'mlp.dense_h_to_4h': ['feed_forward.w3', 'feed_forward.w1'],  # gate weights come second for us
        'mlp.dense_4h_to_h': ['feed_forward.w2'],
    }

    # copy weights
    megatron_dict = {
        'model': {
            'language_model': {
                'embedding': {},
                'transformer': {},
                'lm_head': None
                },
            }
    }
    n_layers = scale2layer[f"{size}B"]
    megatron_dict['model']['language_model']['embedding']['word_embeddings.weight'] = llama_config['tok_embeddings.weight']
    megatron_dict['model']['language_model']['transformer']['final_layernorm.weight'] = llama_config['norm.weight']
    megatron_dict['model']['language_model']['lm_head'] = llama_config['output.weight']
    for layer_idx in trange(n_layers):
        layer_prefix = f'layers.{layer_idx}.'
        for megatron_param, llama_param_list in megatron2llama.items():
            if len(llama_param_list)==1:
                megatron_dict['model']['language_model']['transformer'][layer_prefix+megatron_param+'.weight'] = llama_config[layer_prefix+llama_param_list[0]+'.weight']
            elif len(llama_param_list)==3:
                megatron_dict['model']['language_model']['transformer'][layer_prefix+megatron_param+'.weight'] = get_wqkv(llama_config, layer_prefix, n_heads=scale2heads[f"{size}B"])
            else:
                megatron_dict['model']['language_model']['transformer'][layer_prefix+megatron_param+'.weight'] = torch.concat([llama_config[layer_prefix+w+'.weight'] for w in llama_param_list], dim=0)
    return megatron_dict


def main(model_name: str = "falcon", size: int = 7, out: Optional[Path] = None,
         cache_dir: Optional[Path] = None, megatron_path: Optional[Path] = None):
    if out is None:
        out = Path(f"falcon{size}b_megatron.pt").absolute()

    # get weights from or specified directory
    if model_name == "falcon":
        print("Fetching weights from huggingface")
        model = AutoModelForCausalLM.from_pretrained(f"tiiuae/falcon-{size}b",
                                                     trust_remote_code=True,
                                                     cache_dir=cache_dir)
        hf_weights = model.state_dict()
    else:
        print("Getting llama...")
        hf_weights = merge_llama(size, cache_dir)

    # convert state dict to be megatron-compatible
    if model_name == "falcon":
        megatron_weights = falcon_to_megatron(hf_weights, size)
    else:
        megatron_weights = llama_to_megatron(hf_weights, size,
                                             version=1 if model_name == "llama" else 2)
        megatron_weights = megatron_weights["model"]["language_model"]

    # set args
    dtype = megatron_weights["embedding"]["word_embeddings.weight"].dtype
    if model_name == "falcon":
        if size == 7:
            args = {"num_layers": 32, "hidden_size": 4544,
                    "num_attention_heads": 71, "num_attention_heads_kv": 1}
        else:
            args = {"num_layers": 60, "hidden_size": 8192,
                    "num_attention_heads": 128, "num_attention_heads_kv": 8,
                    "parallel_layernorm": True}
        args.update({"tokenizer_type": "FalconTokenizer", "use_flash_attn": True,
                     "hidden_dropout": 0.0,
                     "parallel_attn": True, "max_position_embeddings": 2048,
                     "seq_length": 2048})
    else:
        args = {"num_layers": llama_s2layer[size],
                "hidden_size": llama_s2hidden[size],
                "num_attention_heads": llama_s2heads[size],
                "ffn_hidden_size": llama_s2dense[size],
                "parallel_attn": False,
                "make_vocab_size_divisible_by": 1,
                "glu_activation": "swiglu",
                "padded_vocab_size": 32000,
                "use_rms_norm": True,
                "tie_embed_logits": False,
                "tokenizer_type": "SentencePieceTokenizer"}
        if model_name == "llama":
            args.update({"max_position_embeddings": 2048, "seq_length": 2048,
                         "layernorm_epsilon": 1e-6})
        else:  # llama2
            args.update({"max_position_embeddings": 4096, "seq_length": 4096,
                         "layernorm_epsilon": 1e-5})
            if size >= 34:
                args.update({"num_attention_heads_kv": 8})
    args.update({
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "iteration": "release",
        "bias_gelu_fusion": False,
        "bias_droput_fusion": False,
        "position_embedding_type": "rotary"
    })

    # save converted weights in specified out
    (out/"release"/"mp_rank_00").mkdir(parents=True)
    with open(out/"latest_checkpointed_iteration.txt", "w+") as f:
        f.write("release")
    final_dict = {"iteration": "release", "model": {"language_model": megatron_weights},
                  "checkpoint_version": 3.0, "args": Namespace(**args)}
    torch.save(final_dict, out/"release"/"mp_rank_00"/"model_optim_rng.pt")
    print("Saved weights in", out)
    print("Done")


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert Huggingface falcon weights to "
                                        "megatron-compatible weights")
    parser.add_argument("model", choices={"falcon", "llama", "llama2"})
    parser.add_argument("--size", default=7, choices={7, 13, 30, 34, 40, 65, 70}, type=int,
                        help="The size of the model")
    parser.add_argument("--out", type=Path,
                        help="Directory to store the megatron weights (as checkpoint)")
    parser.add_argument("--cache-dir", type=Path,
                        help=("Directory to store the huggingface weights, or "
                              "in case of the llama model, where to look for "
                              "the consolidated.xx.pth"))
    parser.add_argument("--megatron-path", type=Path,
                        help="Path where to find megatron code")
    args = parser.parse_args()

    # small arg verification
    if args.model == "falcon":
        assert args.size in {7, 40}
    elif args.model == "llama":
        assert args.size in {7, 13, 30, 65}
    else:
        assert args.size in {7, 13, 70}

    main(args.model, args.size, args.out, args.cache_dir, args.megatron_path)
