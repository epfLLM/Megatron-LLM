"""
Convert megatron checkpoints to huggingface weights.

This script will also convert the tokenizer configured.
Set the `--input_dir` to the megatron checkpoint root (i.e. where the
`latest_checkpointed_iteration.txt` file is located) and  `--output_dir` to
the directory where the huggingface weights should be stored.
"""

# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import os
import sys
import json
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from argparse import ArgumentParser, Namespace
sys.path.append(str(Path(__file__).parent.parent.absolute()))  # megatron is importable

import torch
from tqdm.auto import trange
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast, FalconConfig, FalconForCausalLM, AutoTokenizer, MistralConfig, MistralForCausalLM

from utils.permute_qkv import permute_qkv

from megatron.tokenizer import build_tokenizer


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def convert_wqkv(llama_mega, layer_idx=0, n_heads=32, n_heads_kv=8):
    qkv_w = llama_mega["transformer"][f'layers.{layer_idx}.attention.query_key_value.weight']
    n_hidden = qkv_w.size(1)
    hidden_dim = n_hidden//n_heads
    qkv_w = permute_qkv(qkv_w, n_hidden, n_heads, n_heads_kv, revert=True)

    n_qs_per_kv = n_heads//n_heads_kv
    n_groups = qkv_w.size(0)//hidden_dim//(n_qs_per_kv + 2)
    qkv_w = list(torch.split(qkv_w, hidden_dim, dim=0))

    wq, wk, wv = [], [], []
    for group in range(n_groups):
        for qs in range(n_qs_per_kv):
            wq.append(qkv_w[0])
            del qkv_w[0]
        wk.append(qkv_w[0])
        del qkv_w[0]
        wv.append(qkv_w[0])
        del qkv_w[0]
    assert len(qkv_w) == 0

    wq = torch.concat(wq, dim=0)
    wk = torch.concat(wk, dim=0)
    wv = torch.concat(wv, dim=0)
    return wq, wk, wv


def convert_ffn(llama_mega, layer_idx=0, n_dense=11008):
    mega_ffn = llama_mega["transformer"][f'layers.{layer_idx}.mlp.dense_h_to_4h.weight']
    ffn_w3, ffn_w1 = mega_ffn.split(n_dense, dim=0)
    return ffn_w1, ffn_w3


def write_llama_model(model_path,
                input_base_path,
                num_output_shards: int=2,
                norm_eps: float=1e-05,
                rope_theta: float=1e4):

    # Preliminaries
    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(input_base_path, 'latest_checkpointed_iteration.txt')) as f:
        iteration = f.read()
    if iteration != "release":
        iteration = f"iter_{int(iteration):07d}"
    print(f"Fetching iteration {iteration}")

    # Load weights
    base_path = Path(input_base_path)/iteration
    assert len(list(base_path.glob("mp_rank_*"))) == 1, "Unshard your model with checkpoint_util.py first!"
    loaded = torch.load(base_path/"mp_rank_00"/"model_optim_rng.pt", map_location="cpu")
    args = loaded['args']

    loaded = loaded['model']['language_model']
    if 'transformer' not in loaded:  # normalize key names
        loaded["transformer"] = loaded.pop("encoder")
        for key in list(loaded["transformer"].keys()):
            loaded["transformer"][key.replace("self_attention", "attention")] = loaded["transformer"].pop(key)
        loaded["embedding"]["word_embeddings.weight"] = loaded["embedding"].pop("word_embeddings")["weight"]
        args.num_layers = args.encoder_num_layers

    # Load arguments
    n_layers = args.num_layers
    n_heads = args.num_attention_heads
    n_heads_kv = getattr(args, "num_attention_heads_kv", n_heads)
    n_dense = args.ffn_hidden_size
    n_hidden = args.hidden_size
    hidden_per_head = n_hidden // n_heads
    intermediate_size = args.ffn_hidden_size
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, hidden_per_head, 2).float() / hidden_per_head))

    print('Llama-Megatron Loaded!')
    param_count = 0
    index_dict = {"weight_map": {}}
        
    # Start conversion
    with TemporaryDirectory() as tmp_model_path:
        print(f'Weighted Converting for {n_layers} layers...')
        for layer_i in range(n_layers):
            filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
            wq_proj, wk_proj, wv_proj = convert_wqkv(llama_mega=loaded, 
                                          layer_idx=layer_i, n_heads=n_heads,
                                          n_heads_kv=n_heads_kv)
            ffn_w1, ffn_w3 = convert_ffn(llama_mega=loaded, 
                                        layer_idx=layer_i, 
                                        n_dense=n_dense)
            state_dict = {
                f"model.layers.{layer_i}.self_attn.q_proj.weight": wq_proj,
                f"model.layers.{layer_i}.self_attn.k_proj.weight": wk_proj,
                f"model.layers.{layer_i}.self_attn.v_proj.weight": wv_proj,
                f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded["transformer"][f"layers.{layer_i}.attention.dense.weight"],
                f"model.layers.{layer_i}.mlp.gate_proj.weight": ffn_w1,
                f"model.layers.{layer_i}.mlp.down_proj.weight": loaded["transformer"][f"layers.{layer_i}.mlp.dense_4h_to_h.weight"],
                f"model.layers.{layer_i}.mlp.up_proj.weight": ffn_w3,
                f"model.layers.{layer_i}.input_layernorm.weight": loaded["transformer"][f"layers.{layer_i}.input_layernorm.weight"],
                f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded["transformer"][f"layers.{layer_i}.post_attention_layernorm.weight"],
                f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq": inv_freq
            }

            for k, v in state_dict.items():
                index_dict["weight_map"][k] = filename
                param_count += v.numel()
            torch.save(state_dict, os.path.join(tmp_model_path, filename))
            print(f'Sharded file saved to {filename}')

        filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
        state_dict = {
            "model.norm.weight": loaded["transformer"]['final_layernorm.weight'],
            "lm_head.weight": loaded['lm_head'],
            "model.embed_tokens.weight": loaded['embedding']["word_embeddings.weight"]
        }

        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch_dtype = state_dict["lm_head.weight"].dtype
        torch.save(state_dict, os.path.join(tmp_model_path, filename))
        print(f'Sharded file saved to {filename}')

        # Write configs and save
        index_dict["metadata"] = {"total_size": param_count * 2}
        write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))
        config = LlamaConfig(
            vocab_size=args.padded_vocab_size,
            hidden_size=n_hidden,
            intermediate_size=intermediate_size,
            num_attention_heads=n_heads,
            num_hidden_layers=n_layers,
            rms_norm_eps=norm_eps,
            num_key_value_heads=n_heads_kv,
            max_position_embeddings=args.seq_length,
        )
        config.save_pretrained(tmp_model_path)

        # Make space so we can load the model properly now.
        del state_dict
        del loaded
        gc.collect()

        print("Loading the checkpoint in a Llama model...")
        model = LlamaForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch_dtype)
        # Avoid saving this as part of the config.
        del model.config._name_or_path

    print("Saving in the Transformers format.")
    max_num_params_per_shard = param_count*2 // max(1,(num_output_shards-1))
    model.save_pretrained(model_path, max_shard_size=max_num_params_per_shard)

def write_mistral_model(
    model_path,
    input_base_path,
    num_output_shards: int=2,
    norm_eps: float=1e-5,
    rope_theta: float=10000.0,
    vocab_size: int=None,
):

    # Preliminaries
    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(input_base_path, 'latest_checkpointed_iteration.txt')) as f:
        iteration = f.read()
    if iteration != "release":
        iteration = f"iter_{int(iteration):07d}"
    print(f"Fetching iteration {iteration}")

    # Load weights
    base_path = Path(input_base_path)/iteration
    assert len(list(base_path.glob("mp_rank_*"))) == 1, "Unshard your model with checkpoint_util.py first!"
    loaded = torch.load(base_path/"mp_rank_00"/"model_optim_rng.pt", map_location="cpu")
    args = loaded['args']

    loaded = loaded['model']['language_model']
    if 'transformer' not in loaded:  # normalize key names
        loaded["transformer"] = loaded.pop("encoder")
        for key in list(loaded["transformer"].keys()):
            loaded["transformer"][key.replace("self_attention", "attention")] = loaded["transformer"].pop(key)
        loaded["embedding"]["word_embeddings.weight"] = loaded["embedding"].pop("word_embeddings")["weight"]
        args.num_layers = args.encoder_num_layers

    # Load arguments
    n_layers = args.num_layers
    n_heads = args.num_attention_heads
    n_heads_kv = getattr(args, "num_attention_heads_kv", n_heads)
    n_dense = args.ffn_hidden_size
    n_hidden = args.hidden_size
    hidden_per_head = n_hidden // n_heads
    intermediate_size = args.ffn_hidden_size
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, hidden_per_head, 2).float() / hidden_per_head))

    print('Mistral-Megatron Loaded!')
    param_count = 0
    index_dict = {"weight_map": {}}
        
    # Start conversion
    with TemporaryDirectory() as tmp_model_path:
        print(f'Weighted Converting for {n_layers} layers...')
        for layer_i in range(n_layers):
            filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
            wq_proj, wk_proj, wv_proj = convert_wqkv(llama_mega=loaded, 
                                          layer_idx=layer_i, n_heads=n_heads,
                                          n_heads_kv=n_heads_kv)
            ffn_w1, ffn_w3 = convert_ffn(llama_mega=loaded, 
                                        layer_idx=layer_i, 
                                        n_dense=n_dense)
            state_dict = {
                f"model.layers.{layer_i}.self_attn.q_proj.weight": wq_proj,
                f"model.layers.{layer_i}.self_attn.k_proj.weight": wk_proj,
                f"model.layers.{layer_i}.self_attn.v_proj.weight": wv_proj,
                f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded["transformer"][f"layers.{layer_i}.attention.dense.weight"],
                f"model.layers.{layer_i}.mlp.gate_proj.weight": ffn_w1,
                f"model.layers.{layer_i}.mlp.down_proj.weight": loaded["transformer"][f"layers.{layer_i}.mlp.dense_4h_to_h.weight"],
                f"model.layers.{layer_i}.mlp.up_proj.weight": ffn_w3,
                f"model.layers.{layer_i}.input_layernorm.weight": loaded["transformer"][f"layers.{layer_i}.input_layernorm.weight"],
                f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded["transformer"][f"layers.{layer_i}.post_attention_layernorm.weight"],
                f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq": inv_freq
            }

            for k, v in state_dict.items():
                index_dict["weight_map"][k] = filename
                param_count += v.numel()
            torch.save(state_dict, os.path.join(tmp_model_path, filename))
            print(f'Sharded file saved to {filename}')

        filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
        state_dict = {
            "model.norm.weight": loaded["transformer"]['final_layernorm.weight'],
            "lm_head.weight": loaded['lm_head'],
            "model.embed_tokens.weight": loaded['embedding']["word_embeddings.weight"]
        }

        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch_dtype = state_dict["lm_head.weight"].dtype
        torch.save(state_dict, os.path.join(tmp_model_path, filename))
        print(f'Sharded file saved to {filename}')

        # Write configs and save
        index_dict["metadata"] = {"total_size": param_count * 2}
        write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))

        # load mistral config from huggingface
        config = MistralConfig.from_pretrained(
            "mistralai/Mistral-7B-v0.1"
        )
        # assert configuration matches
        assert config.hidden_size == n_hidden
        assert config.intermediate_size == intermediate_size
        assert config.num_attention_heads == n_heads
        assert config.num_hidden_layers == n_layers
        assert config.rms_norm_eps == norm_eps
        assert config.num_key_value_heads == n_heads_kv
        # Set vocab size
        config.vocab_size = args.padded_vocab_size
        config.save_pretrained(tmp_model_path)

        # Make space so we can load the model properly now.
        del state_dict
        del loaded
        gc.collect()

        if vocab_size is None:
            vocab_size = args.padded_vocab_size
        else:
            print(f"Using vocab size {vocab_size} from tokenizer and not {args.padded_vocab_size} from args.")
            # update config
            config.vocab_size = vocab_size

        print("Loading the checkpoint in a Mistral model...")
        model = MistralForCausalLM.from_pretrained(
            tmp_model_path,
            torch_dtype=torch_dtype
        )
        model.config.vocab_size = vocab_size
        # resizes the embedding layer to the correct size
        model.resize_token_embeddings(vocab_size)
        # Avoid saving this as part of the config.
        del model.config._name_or_path

    print("Saving in the Transformers format.")
    max_num_params_per_shard = param_count*2 // max(1,(num_output_shards-1))
    model.save_pretrained(model_path, max_shard_size=max_num_params_per_shard)


def write_falcon_model(
    model_path: str,
    input_base_path: str,
    num_output_shards: int = 2,
    safe_serialization: bool = True,
):
    # Preliminaries
    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    input_base_path = Path(input_base_path)
    iteration = (input_base_path / "latest_checkpointed_iteration.txt").read_text()
    if iteration != "release":
        iteration = f"iter_{int(iteration):07d}"
    print(f"Fetching iteration {iteration}")

    # Load weights
    loaded = torch.load(
        input_base_path / iteration / "mp_rank_00" / "model_optim_rng.pt",
        map_location="cpu",
    )
    args = loaded["args"]
    loaded = loaded["model"]["language_model"]

    if "transformer" not in loaded:  # normalize key names
        loaded["transformer"] = loaded.pop("encoder")
        loaded["embedding"]["word_embeddings.weight"] = loaded["embedding"].pop(
            "word_embeddings"
        )["weight"]
        args.num_layers = args.encoder_num_layers

    # Make sure the self_attention layer is called "attention" in the megatron state dict
    for key in list(loaded["transformer"].keys()):
        loaded["transformer"][key.replace("self_attention", "attention")] = loaded[
            "transformer"
        ].pop(key)

    embedding = loaded["embedding"]
    transformer = loaded["transformer"]

    # Load arguments
    n_layers = args.num_layers
    dim = args.hidden_size
    n_heads = args.num_attention_heads
    n_heads_kv = args.num_attention_heads_kv

    def permute(qkv_w):
        return permute_qkv(qkv_w, dim, n_heads, n_heads_kv, revert=True)

    weights = {}

    # weights independent of layers (i.e. token embeddings and layernorms
    weights["transformer.word_embeddings.weight"] = embedding["word_embeddings.weight"]
    weights["lm_head.weight"] = weights["transformer.word_embeddings.weight"]
    weights["transformer.ln_f.weight"] = transformer["final_layernorm.weight"]
    weights["transformer.ln_f.bias"] = transformer["final_layernorm.bias"]

    # copy weights for each transformer layer
    for layer in trange(n_layers, desc="Converting weights"):
        prefix1 = f"layers.{layer}"
        prefix2 = f"transformer.h.{layer}"
        # mlp
        weights[f"{prefix2}.mlp.dense_h_to_4h.weight"] = transformer[
            f"{prefix1}.mlp.dense_h_to_4h.weight"
        ]
        weights[f"{prefix2}.mlp.dense_4h_to_h.weight"] = transformer[
            f"{prefix1}.mlp.dense_4h_to_h.weight"
        ]

        # qkv weights
        weights[f"{prefix2}.self_attention.query_key_value.weight"] = permute(
            transformer[f"{prefix1}.attention.query_key_value.weight"]
        )

        # dense
        weights[f"{prefix2}.self_attention.dense.weight"] = transformer[
            f"{prefix1}.attention.dense.weight"
        ]

        # falcon7 and falcon40 differ in the input layernorms
        if n_layers <= 32:  # 7B model
            weights[f"{prefix2}.input_layernorm.weight"] = transformer[
                f"{prefix1}.input_layernorm.weight"
            ]
            weights[f"{prefix2}.input_layernorm.bias"] = transformer[
                f"{prefix1}.input_layernorm.bias"
            ]
        else:
            weights[f"{prefix2}.ln_attn.weight"] = transformer[
                f"{prefix1}.input_layernorm.weight"
            ]
            weights[f"{prefix2}.ln_mlp.weight"] = transformer[
                f"{prefix1}.mlp_layernorm.weight"
            ]
            weights[f"{prefix2}.ln_attn.bias"] = transformer[
                f"{prefix1}.input_layernorm.bias"
            ]
            weights[f"{prefix2}.ln_mlp.bias"] = transformer[
                f"{prefix1}.mlp_layernorm.bias"
            ]

    print("Falcon-Megatron Loaded!")

    vocab_size = 65024  # default size for falcon
    if "padded_vocab_size" in args:
        vocab_size = args.padded_vocab_size

    # creating HF falcon model
    config = FalconConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_kv_heads=None
        if args.num_attention_heads_kv == 1
        else args.num_attention_heads_kv,
        new_decoder_architecture=args.num_layers >= 60,
    )

    print("Creating FalconForCausalLM")
    model = FalconForCausalLM(config=config)
    torch_dtype = weights["lm_head.weight"].dtype
    print(f"dtype: {torch_dtype}")
    print("Loading state dict...")
    model.to(torch_dtype)  # convert model to soucre dtype
    model.load_state_dict(weights)
    print("Done!")

    param_count = 0
    for v in weights.values():
        param_count += v.numel()
    print(f"param_count: {param_count:,}")

    # write model
    print(f"Saving in the Transformers format to: {model_path} ({torch_dtype})")
    bits_per_param = torch.finfo(torch_dtype).bits
    max_shard_size = param_count * bits_per_param // num_output_shards // 8
    print(f"max_shard_size: {max_shard_size:,} bytes")
    model.save_pretrained(
        model_path,
        max_shard_size=max_shard_size,
        safe_serialization=safe_serialization,
    )


def write_tokenizer(args: Namespace):
    if args.model in {"llama", "llama2", "codellama", "mistral"}:
        # mistral also use LlamaTokenizerFast
        args.tokenizer_type = "SentencePieceTokenizer"
        if args.vocab_file:
            # prevent "single file or url is deprecated and won't be possible anymore in v5" warning,
            # use parent directory instead
            p = Path(args.vocab_file)
            if p.suffix == ".model":
                p = p.parent
            hf_tokenizer = LlamaTokenizerFast.from_pretrained(p)
            args.vocab_file = hf_tokenizer.vocab_file
        else:
            if args.model == "codellama":
                hf_repo_name = "TheBloke/CodeLlama-13B-fp16"
            elif args.model == "mistral":
                hf_repo_name = "mistralai/Mistral-7B-v0.1"
            else:
                hf_repo_name = "meta-llama/Llama-2-7b-hf"
            try:  # try loading from huggingface
                hf_tokenizer = LlamaTokenizerFast.from_pretrained(hf_repo_name,
                                                            cache_dir=args.cache_dir)
                print("LlamaTokenizerFast loaded from huggingface")
                print("vocab_file not set, assuming same tokenizer.model used "
                      "by llama LlamaTokenizerFast")
                args.vocab_file = hf_tokenizer.vocab_file
            except OSError:
                print(f"ERROR: Could not load tokenizer from HF repo '{hf_repo_name}'. "
                      "Tokenizer processing failed.")
                return
    else:
        hf_tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40b", cache_dir=args.cache_dir)
        args.tokenizer_type = "FalconTokenizer"

    # add default args for megatron tokenizer
    args.rank = 0
    args.vocab_extra_ids = 0
    args.new_tokens = True
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    mt_tokenizer = build_tokenizer(args)

    if args.tokenizer_type == "SentencePieceTokenizer":
        if mt_tokenizer.cls is not None:
            hf_tokenizer.add_tokens("<CLS>", special_tokens=True)
            hf_tokenizer.cls_token_id = mt_tokenizer.cls
        if mt_tokenizer.sep is not None:
            hf_tokenizer.add_tokens("<SEP>", special_tokens=True)
            hf_tokenizer.sep_token_id = mt_tokenizer.sep
        if mt_tokenizer.eod is not None:
            hf_tokenizer.add_tokens("<EOD>", special_tokens=True)
        if mt_tokenizer.mask is not None:
            hf_tokenizer.add_tokens("<MASK>", special_tokens=True)
            hf_tokenizer.mask_token_id = mt_tokenizer.mask
        if mt_tokenizer.pad is not None:
            hf_tokenizer.add_tokens("<PAD>", special_tokens=True)
            hf_tokenizer.pad_token_id = mt_tokenizer.pad

        additional_special_tokens = hf_tokenizer.additional_special_tokens
        if args.vocab_extra_ids_list:
            additional_special_tokens.extend(args.vocab_extra_ids_list.split(","))

        for special_token in additional_special_tokens:
            hf_tokenizer.add_special_tokens({"additional_special_tokens": [special_token]})

        hf_vocab = hf_tokenizer.get_vocab()
        tokens_to_check = [
            v for k, v in hf_tokenizer.special_tokens_map.items() if k != "additional_special_tokens"
        ] + additional_special_tokens
        for t in tokens_to_check:
            a = mt_tokenizer.vocab.get(t)
            b = hf_vocab.get(t)
            assert a == b, f"Mismatch between megatron and huggingface tokenizer vocabularies {t}, {a}, {b}"
    elif args.tokenizer_type == "FalconTokenizer":
        hf_tokenizer = mt_tokenizer.tokenizer
    else:
        raise RuntimeError(f"Unsupported tokenizer type: {args.tokenizer_type}")

    # handle special token overrides
    for override in args.override_special_tokens:
        try:
            key, value = override.split("=")
            assert key in {"bos", "cls", "eos", "mask", "pad", "sep", "unk"}
            value = mt_tokenizer.vocab[value]
            setattr(hf_tokenizer, f"{key}_token_id", value)
        except ValueError:
            warnings.warn(f"Illegal override string {override}")
        except AssertionError:
            warnings.warn(f"Cannot override key {key}")
        except KeyError:
            warnings.warn(f"Token {value} not found in megatron tokenizer")

    print("Final HF Tokenizer configuration:")
    print(hf_tokenizer)
    hf_tokenizer.save_pretrained(args.output_dir)


def main():
    # make sure megatron is importable

    parser = ArgumentParser()
    parser.add_argument("--input_dir", help="Location of Megatron weights",
                        required=True)
    parser.add_argument("--num_output_shards", type=int, default=1)
    parser.add_argument("--model", choices={"falcon", "llama", "llama2", "codellama", "mistral"},
                         default="llama2")
    parser.add_argument("--output_dir", help="Location to write HF model and tokenizer",
                        required=True)
    parser.add_argument("--cache_dir", help="Huggingface cache_dir (optional)")
    parser.add_argument("--vocab_file", type=str, help="Path to the vocab file")
    parser.add_argument("--vocab_extra_ids_list",
                        help="comma separated list of special vocab ids to add to the tokenizer")
    parser.add_argument("--override_special_tokens", nargs="*", default=[],
                        help=("One or more arguments to override special tokens. "
                              "Syntax set as `key=value`, e.g. `eos=<|im_end|>`. "
                              "Overrides available only bos, cls, eos, mask, pad, sep, unk."))
    
    args = parser.parse_args()
    if args.model in {"llama", "llama2", "codellama"}:
        eps = 1e-6 if args.model == "llama" else 1e-5
        rope_theta = 1e6 if args.model == "codellama" else 1e4
        write_llama_model(
            model_path=args.output_dir,
            input_base_path=args.input_dir,
            num_output_shards=args.num_output_shards,
            norm_eps=eps,
            rope_theta=rope_theta,
        )
    elif args.model == "mistral":
        write_mistral_model(
            model_path=args.output_dir,
            input_base_path=args.input_dir,
            num_output_shards=args.num_output_shards,
            vocab_size=vocab_size,
        )
    elif args.model == "falcon":
        write_falcon_model(
            model_path=args.output_dir,
            input_base_path=args.input_dir,
            num_output_shards=args.num_output_shards,
            safe_serialization=True,
        )
    write_tokenizer(args)

if __name__ == "__main__":
    main()
