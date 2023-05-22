import os
import torch
from collections import OrderedDict
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=False, default='13B')
parser.add_argument('--llama_config_path', type=str, required=False, default='/mlodata1/llms/llama/')
args = parser.parse_args()

LLAMA_config_PATH = os.path.join(args.llama_config_path, f'merged_{args.model_name}.pth')
# LLAMA_config_PATH = f'/mlodata1/sfan/merged_{args.model_name}.pth'
print(f'Loading {args.model_name} checkpoint...')
llama_config = torch.load(LLAMA_config_PATH)

scale2layer = {
    '7B': 32,
    '13B': 40,
    '30B': 60,
    '65B': 80,
}

scale2heads = {
    '7B': 32,
    '13B': 40,
    '30B': 52,
    '65B': 64,
}

megatron2llama = {
    'attention.query_key_value': ['attention.wq', 'attention.wk', 'attention.wv'],
    'attention.dense': ['attention.wo'],
    'post_attention_layernorm': ['ffn_norm'],
    'input_layernorm': ['attention_norm'],
    'mlp.dense_h_to_4h': ['feed_forward.w1', 'feed_forward.w3'],
    'mlp.dense_4h_to_h': ['feed_forward.w2'],
}

def get_wqkv(llama_config, layer_prefix, n_heads=32):
    wq, wk, wv = llama_config[layer_prefix+'attention.wq.weight'], llama_config[layer_prefix+'attention.wk.weight'], llama_config[layer_prefix+'attention.wv.weight']
    n_hidden_per_head = wq.shape[-1] // n_heads

    wq_convert = torch.split(wq, n_hidden_per_head, dim=0)
    wk_convert = torch.split(wk, n_hidden_per_head, dim=0)
    wv_convert = torch.split(wv, n_hidden_per_head, dim=0)
    assert len(wq_convert)==n_heads

    w_qkv = []
    for i in range(n_heads):
        w_qkv.extend([wq_convert[i], wk_convert[i], wv_convert[i]])
    out = torch.concat(w_qkv, dim=0)
    return out


if __name__ == '__main__' :
    megatron_dict = {
        'model': {
            'language_model': {
                'embedding': {},
                'transformer': {},
                },
            }
    }
    n_layers = scale2layer[args.model_name]
    bar = tqdm(total=n_layers)
    megatron_dict['model']['language_model']['embedding']['word_embeddings.weight'] = llama_config['tok_embeddings.weight']
    megatron_dict['model']['language_model']['transformer']['final_layernorm.weight'] = llama_config['norm.weight']
    megatron_dict['model']['language_model']['transformer']['proj_out.weihgt'] = llama_config['output.weight']
    for layer_idx in range(n_layers):
        layer_prefix = f'layers.{layer_idx}.'
        for megatron_param, llama_param_list in megatron2llama.items():
            if len(llama_param_list)==1:
                megatron_dict['model']['language_model']['transformer'][layer_prefix+megatron_param+'.weight'] = llama_config[layer_prefix+llama_param_list[0]+'.weight']
            elif len(llama_param_list)==3:
                megatron_dict['model']['language_model']['transformer'][layer_prefix+megatron_param+'.weight'] = get_wqkv(llama_config, layer_prefix, n_heads=scale2heads[args.model_name])
            else:
                megatron_dict['model']['language_model']['transformer'][layer_prefix+megatron_param+'.weight'] = torch.concat([llama_config[layer_prefix+w+'.weight'] for w in llama_param_list], dim=0)
        bar.update(1)

    torch.save(megatron_dict, f'convert_{args.model_name}.pth')