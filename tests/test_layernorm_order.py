import argparse
import os

import torch

import megatron
import megatron.initialize
import megatron.model.utils
import megatron.model.language_model
import megatron.arguments

import megatron.core.tensor_parallel.random
import megatron.model.transformer

from megatron.model.enums import AttnMaskType, ModelType, LayerType


init_method_std = .02
num_layers = 2
layer_number = 1

init_method = megatron.model.utils.init_method_normal(init_method_std)
output_layer_init_method = megatron.model.utils.scaled_init_method_normal(init_method_std, num_layers)
layer_type = LayerType.encoder

"""
--use_bias
--micro_batch_size 2
--num_layers 2
--hidden_size 4
--num_attention_heads 4
--max_position_embeddings 4
--encoder_seq_length 4
--global_batch_size 128
--train_iters 2000000
--data_impl mmap
--split 80,10,10
--distributed_backend nccl
--lr_decay_style constant
--lr 0.0001
"""


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Megatron-LM Arguments',
    #                                  allow_abbrev=False)
    # extra_args_provider = get_the_parser_bro
    # extra_args_provider = None
    # args = megatron.get_args()
    base_parser = megatron.arguments.build_base_parser()
    args = base_parser.parse_args(["--micro_batch_size", "4"])
    args_defaults = {"micro_batch_size": 2,
                     "num_layers": 2,
                     "hidden_size": 4,
                     "num_attention_heads": 4,
                     "max_position_embeddings": 4,
                     "encoder_seq_length": 4
                     }

    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    _MODEL_PARALLEL_RNG_TRACKER_NAME = 'model-parallel-rng'
    megatron.core.tensor_parallel.random._CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME,
                                111)

    # megatron.initialize.initialize_megatron(extra_args_provider=None,
    #                                         args_defaults=args_defaults)
    # args = megatron.arguments.parse_args(extra_args_provider=None)
    megatron.arguments.validate_args(args, args_defaults)
    # megatron.initialize._compile_dependencies(args)
    megatron.fused_kernels.load(args)

    device = torch.device("cuda")
    world_size = 1
    # layer2 = megatron.model.transformer_matoba.ParallelTransformerLayer(init_method,
    #                                                                     output_layer_init_method,
    #                                                                     layer_number,
    #                                                                     layer_type,
    #                                                                     args=args)
    layer1 = megatron.model.transformer.ParallelTransformerLayer(init_method,
                                                                 output_layer_init_method,
                                                                 layer_number,
                                                                 layer_type,
                                                                 world_size=world_size,
                                                                 args=args).to(device)

    attention_mask = torch.tensor([[[[False, True, True, True],
                                     [False, False, True, True],
                                     [False, False, False, True],
                                     [False, False, False, False]]]]).to(device)
    hidden_states = torch.tensor([[[0.0000, 0.0334, -0.0528, -0.0357],
                                   [-0.0061, -0.0052, 0.0041, -0.0000]],
                                  [[0.0075, 0.0000, -0.0000, -0.0542],
                                   [0.0196, 0.0000, -0.0114, -0.0205]],
                                  [[0.0077, 0.0188, 0.0371, 0.0155],
                                   [0.0009, 0.0042, 0.0135, 0.0034]],
                                  [[-0.0073, -0.0129, 0.0069, 0.0060],
                                   [-0.0000, -0.0000, 0.0174, 0.0210]]]).to(device)
    y1 = layer1(hidden_states, attention_mask)
    # y2 = layer2(hidden_states, attention_mask)

    # torch.testing.assert_allclose(y1, y2)


