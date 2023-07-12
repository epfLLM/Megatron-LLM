import os
import sys
from pathlib import Path
from typing import Optional
from argparse import ArgumentParser, Namespace

import torch
from tqdm.auto import trange
from transformers import AutoModelForCausalLM

from merge_llama import merge_llama


llama_s2layer = {7: 32, 13: 40, 30: 60, 65: 80}
llama_s2heads = {7: 32, 13: 40, 30: 52, 65: 64}
llama_s2dense = {7: 11008, 13: 13824, 30: 17920, 65: 22016}  # should be (2/3)*4*d, but it isn't exaclty that
llama_s2hidden = {7: 4096, 13: 5120, 32: 6656, 65: 8192}


def falcon_to_megatron(weights: dict, size: int) -> dict:
    embedding = {}
    transformer = {}
    if size == 7:
        n_layer = 32
    else:
        n_layer = 60

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
            weights[f"{prefix2}.self_attention.query_key_value.weight"]
        # dense
        transformer[f"{prefix1}.self_attention.dense.weight"] = \
            weights[f"{prefix2}.self_attention.dense.weight"]
        # falcon7 and falcon40 have different names in the layernorm
        if size == 7:
            transformer[f"{prefix1}.input_layernorm.weight"] = \
                weights[f"{prefix2}.input_layernorm.weight"]
            transformer[f"{prefix1}.input_layernorm.bias"] = \
                weights[f"{prefix2}.input_layernorm.bias"]
        else:
            transformer[f"{prefix1}.input_layernorm.weight"] = \
                weights[f"{prefix2}.ln_attn.weight"]
            transformer[f"{prefix1}.input_layernorm.bias"] = \
                weights[f"{prefix2}.ln_attn.bias"]
    return {"embedding": embedding, "transformer": transformer}


def llama_to_megatron(llama_config: dict, size: int):
    def get_wqkv(llama_config, layer_prefix, n_heads=32):
        wq, wk, wv = llama_config[layer_prefix+'attention.wq.weight'], llama_config[layer_prefix+'attention.wk.weight'], llama_config[layer_prefix+'attention.wv.weight']
        n_hidden_per_head = wq.shape[-1] // n_heads

        dim = wq.shape[-1]
        wq = wq.view(n_heads, n_hidden_per_head//2, 2, dim).transpose(1, 2).reshape(dim, dim)
        wk = wk.view(n_heads, n_hidden_per_head//2, 2, dim).transpose(1, 2).reshape(dim, dim)

        wq_convert = torch.split(wq, n_hidden_per_head, dim=0)
        wk_convert = torch.split(wk, n_hidden_per_head, dim=0)
        wv_convert = torch.split(wv, n_hidden_per_head, dim=0)
        assert len(wq_convert)==n_heads
    
        w_qkv = []
        for i in range(n_heads):
            w_qkv.extend([wq_convert[i], wk_convert[i], wv_convert[i]])
        out = torch.concat(w_qkv, dim=0)
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
                'lm_head': {}
                },
            }
    }
    n_layers = scale2layer[f"{size}B"]
    megatron_dict['model']['language_model']['embedding']['word_embeddings.weight'] = llama_config['tok_embeddings.weight']
    megatron_dict['model']['language_model']['transformer']['final_layernorm.weight'] = llama_config['norm.weight']
    megatron_dict['model']['language_model']['lm_head']['weight'] = llama_config['output.weight']
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

    # imports megatron
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
    if megatron_path is not None:
        sys.path.insert(0, megatron_path)
    from megatron.model.enums import PositionEmbeddingType

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
        megatron_weights = llama_to_megatron(hf_weights, size)
        megatron_weights = megatron_weights["model"]["language_model"]

    # set args
    dtype = megatron_weights["embedding"]["word_embeddings.weight"].dtype
    if model_name == "falcon":
        if size == 7:
            args = {"num_layers": 32, "hidden_size": 4544,
                    "num_attention_heads": 71, "num_attention_heads_kv": 1}
        else:
            args = {"num_layers": 60, "hidden_size": 8192,
                    "num_attention_heads": 128, "num_attention_heads_kv": 8}
        args.update({"tokenizer_type": "FalconTokenizer", "use_flash_attn": True,
                     "hidden_dropout": 0.0, "use_multiquery_attn": True,
                     "parallel_attn": True})
    else:
        args = {"num_layers": llama_s2layer[size],
                "hidden_size": llama_s2hidden[size],
                "num_attention_heads": llama_s2heads[size],
                "ffn_hidden_size": llama_s2dense[size],
                "parallel_attn": False,
                "make_vocab_size_divisible_by": 128,
                "glu_activation": "swiglu",
                "padded_vocab_size": 32000,
                "layernorm_epsilon": 1e-6,
                "use_post_ln": True,
                "use_rms_norm": True,
                "tie_embed_logits": False,
                "tokenizer_type": "SentencePieceTokenizer"}
    args.update({
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "iteration": "release",
        "params_dtype": dtype,
        "seq_length": 2048,
        "max_position_embeddings": 2048,
        "bias_gelu_fusion": False,
        "bias_droput_fusion": False,
        "position_embedding_type": PositionEmbeddingType.rotary,
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
    parser.add_argument("model", choices={"falcon", "llama"})
    parser.add_argument("--size", default=7, choices={7, 13, 30, 40, 65}, type=int,
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
    else:
        assert args.size in {7, 13, 30, 65}

    main(args.model, args.size, args.out, args.cache_dir, args.megatron_path)
