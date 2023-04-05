import os
import torch
from collections import OrderedDict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=False, default='13B')
parser.add_argument('--llama_config_path', type=str, required=False, default='/mlodata1/llms/llama')
args = parser.parse_args()

LLAMA_config_PATH = args.llama_config_path
# os.listdir(LLAMA_config_PATH)
LLAMA_config_PATH_65B = os.path.join(LLAMA_config_PATH, args.model_name)
print(os.listdir(LLAMA_config_PATH_65B))

scale2emb = {
    '7B': 4096,
    '13B': 5120,
    '30B': 6656,
    '65B': 8192,
}

key_to_dim = {
        "w1": 0,
        "w2": -1,
        "w3": 0,
        "wo": -1,
        "wq": 0,
        "wk": 0,
        "wv": 0,
        "output": 0,
        "tok_embeddings": -1,
        "ffn_norm": None,
        "attention_norm": None,
        "norm": None,
        "rope": None,
}

def init_merged_ckpt(pth_00, num_pth=8, emb_dim=8192):
    merged_ckpt = OrderedDict()
    for parameter_name, parameter in pth_00.items():
        short_name = parameter_name.split(".")[-2]
        if key_to_dim[short_name] is None:
            merged_ckpt[parameter_name] = parameter
            del parameter
        elif key_to_dim[short_name] == 0:
            size = parameter.shape[0]
            merged_param_shape = [ parameter.shape[0] * num_pth, parameter.shape[1] ]
            merged_ckpt[parameter_name] = torch.zeros(merged_param_shape)
            merged_ckpt[parameter_name][0 : size, :] = parameter
            del parameter
        elif key_to_dim[short_name] == -1:
            size = parameter.shape[-1]
            merged_param_shape = [ parameter.shape[0], parameter.shape[1] * num_pth]
            merged_ckpt[parameter_name] = torch.zeros(merged_param_shape)
            merged_ckpt[parameter_name][:, 0 : size] = parameter
            del parameter
    return merged_ckpt

if __name__ == '__main__' :
    num_pth = len([f for f in os.listdir(LLAMA_config_PATH_65B) if f.endswith('pth')])
    print('Model Scale:', args.model_name)
    print('Number of sharded ckpt: ', num_pth)
    i = 0
    for ckpt in os.listdir(LLAMA_config_PATH_65B):
        if ckpt.endswith('pth'):
            ckpt_path = os.path.join(LLAMA_config_PATH_65B, ckpt)
            print(f"Loading checkpoint {i}")
            llama_config = torch.load(ckpt_path, map_location=torch.device('cpu'))
            if i == 0:
                merged_ckpt = init_merged_ckpt(llama_config, num_pth=num_pth, emb_dim=scale2emb[args.model_name])
            else:
                for parameter_name, parameter in llama_config.items():
                    short_name = parameter_name.split(".")[-2]
                    if key_to_dim[short_name] == 0:
                        size = parameter.shape[0]
                        merged_param_shape = [ parameter.shape[0] * num_pth, parameter.shape[1] ]
                        merged_ckpt[parameter_name] = torch.zeros(merged_param_shape)
                        merged_ckpt[parameter_name][size * i : size * (i + 1), :] = parameter
                        del parameter
                    if key_to_dim[short_name] == -1:
                        size = parameter.shape[-1]
                        merged_param_shape = [ parameter.shape[0], parameter.shape[1] * num_pth]
                        merged_ckpt[parameter_name] = torch.zeros(merged_param_shape)
                        merged_ckpt[parameter_name][:, size * i : size * (i + 1)] = parameter
                        del parameter
            i += 1
            del llama_config

    torch.save(merged_ckpt, f'merged_{args.model_name}.pth')