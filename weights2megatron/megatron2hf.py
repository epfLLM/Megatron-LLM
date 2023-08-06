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
import argparse
import gc
import json
import math
import os
import shutil
import warnings

import torch

from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer


try:
    from transformers import LlamaTokenizerFast
except ImportError as e:
    warnings.warn(e)
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )
    LlamaTokenizerFast = None

"""
Sample usage:

```
python3 /pure-mlo-scratch/sfan/model-parallel-trainer/llama2megatron/convert_llama2hf.py \
    --input_dir /pure-mlo-scratch/llama/ --model_size 7 --output_dir /pure-mlo-scratch/llama/converted_HF_7B
```

Thereafter, models can be loaded via:

```py
from transformers import LlamaForCausalLM, LlamaTokenizer

model = LlamaForCausalLM.from_pretrained("/output/path")
tokenizer = LlamaTokenizer.from_pretrained("/output/path")
```

Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
"""

llama_s2layer = {7: 32, 13: 40, 30: 60, 65: 80, 70: 80}
llama_s2heads = {7: 32, 13: 40, 30: 52, 65: 64, 70: 64}
llama_s2dense = {7: 11008, 13: 13824, 30: 17920, 65: 22016,
                 70: 28672}  # should be (2/3)*4*d, but it isn't exaclty that
llama_s2hidden = {7: 4096, 13: 5120, 32: 6656, 65: 8192, 70: 8192}


param_dict = {
    'attention.query_key_value': ['attention.wq', 'attention.wk', 'attention.wv'],
    'attention.dense': ['attention.wo'],
    'post_attention_layernorm': ['ffn_norm'],
    'input_layernorm': ['attention_norm'],
    'mlp.dense_h_to_4h': ['feed_forward.w1', 'feed_forward.w3'],
    'mlp.dense_4h_to_h': ['feed_forward.w2'],
}

def compute_intermediate_size(n):
    return int(math.ceil(n * 8 / 3) + 255) // 256 * 256


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)

def convert_wqkv(llama_mega, layer_idx=0, n_heads=32):
    mega_qkv = llama_mega['transformer'][f'layers.{layer_idx}.attention.query_key_value.weight']
    n_hidden_per_head = mega_qkv.shape[0]//n_heads//3
    mega_qkv_chunk = torch.split(mega_qkv, n_hidden_per_head, dim=0)

    wq_proj, wk_proj, wv_proj = [], [], []
    for i,chk in enumerate(mega_qkv_chunk):
        if i%3 == 0:
            wq_proj.append(chk)
        elif i%3 == 1:
            wk_proj.append(chk)
        else:
            wv_proj.append(chk)

    wq_proj = torch.concat(wq_proj, dim=0)
    wk_proj = torch.concat(wk_proj, dim=0)
    wv_proj = torch.concat(wv_proj, dim=0)

    return wq_proj, wk_proj, wv_proj

def convert_ffn(llama_mega, layer_idx=0, n_dense=11008):
    mega_ffn = llama_mega['transformer'][f'layers.{layer_idx}.mlp.dense_h_to_4h.weight']
    ffn_w3, ffn_w1 = mega_ffn.split(n_dense, dim=0)
    return ffn_w1, ffn_w3

def write_model(model_path, 
                input_base_path, 
                model_size,
                num_input_shards=1,
                num_output_shards=2,
                skip_permute=True,
                norm_eps=1e-05):
    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    num_shards = num_input_shards
    n_layers = llama_s2layer[model_size]
    n_heads = llama_s2heads[model_size]
    n_heads_per_shard = n_heads // num_shards
    n_dense = llama_s2dense[model_size]
    n_hidden = llama_s2hidden[model_size]
    hidden_per_head = n_hidden // n_heads
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, hidden_per_head, 2).float() / hidden_per_head))

    # permute for sliced rotary
    def permute(w, skip_permute=skip_permute):
        if skip_permute:
            return w
        return w.view(n_heads, n_hidden // n_heads // 2, 2, n_hidden).transpose(1, 2).reshape(n_hidden, n_hidden)

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    # Load weights
    if num_shards==1:
        # Not sharded
        # (The sharded implementation would also work, but this is simpler.)
        # /pure-mlo-scratch/alhernan/megatron-data/checkpoints/llama2-7b-tp4-pp1-optim/release/mp_rank_00/model_optim_rng.pt
        loaded = torch.load(os.path.join(input_base_path, 'release', 'mp_rank_00', 'model_optim_rng.pt'), map_location="cpu")['model']['language_model']
    else:
        # Sharded
        loaded = [
            torch.load(os.path.join(input_base_path, 'release', f'mp_rank_{i:02d}', 'model_optim_rng.pt'), map_location="cpu")['model']['language_model']
            for i in range(num_shards)
        ]
    print('Llama-Megatron Loaded!')
    param_count = 0
    index_dict = {"weight_map": {}}
    
    print(f'Weighted Converting for {n_layers} layers...')
    for layer_i in range(n_layers):
        print(layer_i)
        filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
        if num_shards == 1:
            # Unsharded
            wq_proj, wk_proj, wv_proj = convert_wqkv(llama_mega=loaded, 
                                          layer_idx=layer_i, n_heads=n_heads)
            ffn_w1, ffn_w3 = convert_ffn(llama_mega=loaded, 
                                        layer_idx=layer_i, 
                                        n_dense=n_dense)
            state_dict = {
                f"model.layers.{layer_i}.self_attn.q_proj.weight": permute(
                    wq_proj
                ),
                f"model.layers.{layer_i}.self_attn.k_proj.weight": permute(
                    wk_proj
                ),
                f"model.layers.{layer_i}.self_attn.v_proj.weight": wv_proj,
                f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded['transformer'][f"layers.{layer_i}.attention.dense.weight"],
                f"model.layers.{layer_i}.mlp.gate_proj.weight": ffn_w1,
                f"model.layers.{layer_i}.mlp.down_proj.weight": loaded['transformer'][f"layers.{layer_i}.mlp.dense_4h_to_h.weight"],
                f"model.layers.{layer_i}.mlp.up_proj.weight": ffn_w3,
                f"model.layers.{layer_i}.input_layernorm.weight": loaded['transformer'][f"layers.{layer_i}.input_layernorm.weight"],
                f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded['transformer'][f"layers.{layer_i}.post_attention_layernorm.weight"],
            }
        else:
            # Sharded
            # Note that attention.w{q,k,v,o}, feed_fordward.w[1,2,3], attention_norm.weight and ffn_norm.weight share
            # the same storage object, saving attention_norm and ffn_norm will save other weights too, which is
            # redundant as other weights will be stitched from multiple shards. To avoid that, they are cloned.

            state_dict = {
                f"model.layers.{layer_i}.input_layernorm.weight": loaded[0]['transformer'][
                    f"layers.{layer_i}.input_layernorm.weight"
                    ].clone(),
                f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[0]['transformer'][
                    f"layers.{layer_i}.post_attention_layernorm.weight"
                    ].clone(),
            }
            
            wqs, wks, wvs, ffn_w1s, ffn_w3s = [], [], [], [], []
            for shard_idx in range(num_shards):
                wq_proj, wk_proj, wv_proj = convert_wqkv(llama_mega=loaded[shard_idx], 
                                            layer_idx=layer_i, n_heads=n_heads)
                ffn_w1, ffn_w3 = convert_ffn(llama_mega=loaded[shard_idx], 
                                            layer_idx=layer_i, 
                                            n_dense=n_dense)
                wqs.append(wq_proj)
                wks.append(wk_proj)
                wvs.append(wv_proj)
                ffn_w1s.append(ffn_w1)
                ffn_w3s.append(ffn_w3)
                
            state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = permute(
                torch.cat(
                    [
                        wq.view(n_heads_per_shard, hidden_per_head, n_hidden)
                        for wq in range(wqs)
                    ],
                    dim=0,
                ).reshape(n_hidden, n_hidden)
            )
            state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = permute(
                torch.cat(
                    [
                        wk.view(n_heads_per_shard, hidden_per_head, n_hidden)
                        for wk in range(wks)
                    ],
                    dim=0,
                ).reshape(n_hidden, n_hidden)
            )
            state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = torch.cat(
                [
                    wv.view(n_heads_per_shard, hidden_per_head, n_hidden)
                        for wv in range(wvs)
                ],
                dim=0,
            ).reshape(n_hidden, n_hidden)

            state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = torch.cat(
                [loaded[i]['transformer'][f"layers.{layer_i}.attention.dense.weight"] for i in range(num_shards)], dim=1
            )
            state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = torch.cat(
                ffn_w1s, dim=0
            )
            state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"] = torch.cat(
                [loaded[i]['transformer'][f"layers.{layer_i}.mlp.dense_4h_to_h.weight"] for i in range(num_shards)], dim=1
            )
            state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"] = torch.cat(
                ffn_w3s, dim=0
            )

        state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))
        print(f'Sharded file saved to {filename}')

    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    if num_shards==1:
        # Unsharded
        state_dict = {
            "model.embed_tokens.weight": loaded['embedding']['word_embeddings.weight'],
            "model.norm.weight": loaded['transformer']['final_layernorm.weight'],
            "lm_head.weight": loaded['lm_head'],
        }
    else:
        state_dict = {
            "model.embed_tokens.weight": loaded[0]['embedding']['word_embeddings.weight'],
            "model.norm.weight": loaded[0]['transformer']['final_layernorm.weight'],
            "lm_head.weight": loaded[0]['lm_head'],
        }

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))

    config = LlamaConfig(
        hidden_size=n_hidden,
        intermediate_size=compute_intermediate_size(n_hidden),
        num_attention_heads=n_heads,
        num_hidden_layers=n_layers,
        rms_norm_eps=norm_eps,
    )
    config.save_pretrained(tmp_model_path)

    # Make space so we can load the model properly now.
    del state_dict
    del loaded
    gc.collect()

    print("Loading the checkpoint in a Llama model...")
    model = LlamaForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    # Avoid saving this as part of the config.
    del model.config._name_or_path

    print("Saving in the Transformers format.")
    
    max_num_params_per_shard = param_count*2 // max(1,(num_output_shards-1))
    model.save_pretrained(model_path, max_shard_size=max_num_params_per_shard)
    # shutil.rmtree(tmp_model_path)


def write_tokenizer(tokenizer_path, input_tokenizer_path):
    # Initialize the tokenizer based on the `spm` model
    tokenizer_class = LlamaTokenizer if LlamaTokenizerFast is None else LlamaTokenizerFast
    print(f"Saving a {tokenizer_class.__name__} to {tokenizer_path}.")
    tokenizer = tokenizer_class(input_tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of LLaMA_Megatron weights",
    )
    parser.add_argument(
        "--model_size",
        type=int,
        choices=[7, 13, 30, 65, 70],
    )
    parser.add_argument(
        "--num_input_shards",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_output_shards",
        type=int,
        default=1,
    )
    parser.add_argument('--skip_permute', action='store_true')
    
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    
    args = parser.parse_args()
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        model_size=args.model_size,
        num_input_shards=args.num_input_shards,
        num_output_shards=args.num_output_shards,
        skip_permute=args.skip_permute
    )
    

if __name__ == "__main__":
    main()
