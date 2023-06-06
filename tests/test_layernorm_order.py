import argparse

import torch

import megatron
import megatron.initialize
import megatron.model.utils
import megatron.model.language_model
import megatron.arguments

from megatron import fused_kernels

import megatron.model.transformer
import megatron.model.transformer_matoba

from megatron.model.enums import AttnMaskType, ModelType, LayerType


init_method_std = .02
num_layers = 2
layer_number = 1

init_method = megatron.model.utils.init_method_normal(init_method_std)
output_layer_init_method = megatron.model.utils.scaled_init_method_normal(init_method_std, num_layers)
layer_type = LayerType.encoder

"""
--num_layers 2
--hidden-size 4
--num-attention-heads 2
--seq-length 5
--max-position-embeddings 5
--lr 0.0001
--lr-decay-iters 990000
--train-iters 2000000
--min-lr 0.00001
--lr-warmup-fraction 0.01
--micro-batch-size 4
--global-batch-size 8

--vocab-file /idiap/user/kmatoba/model-parallel-trainer/bert-large-uncased-vocab.txt
--split 500,300,200
--log-interval 10
--save-interval 500
--eval-interval 100
--eval-iters 10
--save checkpoints/bert_345m
--load checkpoints/bert_345m
--data-path my-bert_text_sentence
--use_bias
"""


def get_the_parser_bro(parser):
    group = parser.add_argument_group(title='')
    group.add_argument("--hidden-size", default=4)
    group.add_argument("--num-attention-heads", default=2)
    group.add_argument("--seq-length", default=5)
    group.add_argument("--max-position-embeddings", default=5)
    group.add_argument("--lr", default=0.0001)
    group.add_argument("--lr-decay-iters", default=990000)
    group.add_argument("--train-iters", default=2000000)
    group.add_argument("--min-lr", default=0.00001)
    group.add_argument("--lr-warmup-fraction", default=0.01)
    group.add_argument("--micro-batch-size", default=4)
    group.add_argument("--global-batch-size", default=8)
    return parser


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Megatron-LM Arguments',
    #                                  allow_abbrev=False)
    # extra_args_provider = get_the_parser_bro
    # extra_args_provider = None
    # args = megatron.get_args()
    base_parser = megatron.arguments.build_base_parser()
    args = base_parser.parse_args(["--micro_batch_size", "4"])
    args_defaults = {"micro_batch_size": 4,
                     "num_layers": 2,
                     "hidden_size": 3,
                     "num_attention_heads": 3,
                     "max_position_embeddings": 13,
                     "seq_length": 10}
    #
    # megatron.initialize.initialize_megatron(extra_args_provider=None,
    #                                         args_defaults=args_defaults)
    args = megatron.arguments.parse_args(extra_args_provider=None)
    megatron.arguments.validate_args(args, args_defaults)
    # megatron.initialize._compile_dependencies(args)
    fused_kernels.load(args)

    layer1 = megatron.model.transformer.ParallelTransformerLayer(init_method,
                                                                 output_layer_init_method,
                                                                 layer_number,
                                                                 layer_type,
                                                                 args=args)
    layer2 = megatron.model.transformer_matoba.ParallelTransformerLayer(init_method,
                                                                        output_layer_init_method,
                                                                        layer_number,
                                                                        layer_type,
                                                                        args=args)
    attention_mask = torch.tensor([[[[False, True, True, True],
                                     [False, False, True, True],
                                     [False, False, False, True],
                                     [False, False, False, False]]]])
    hidden_states = torch.tensor([[[0.0000, 0.0334, -0.0528, -0.0357],
                                   [-0.0061, -0.0052, 0.0041, -0.0000]],
                                  [[0.0075, 0.0000, -0.0000, -0.0542],
                                   [0.0196, 0.0000, -0.0114, -0.0205]],
                                  [[0.0077, 0.0188, 0.0371, 0.0155],
                                   [0.0009, 0.0042, 0.0135, 0.0034]],
                                  [[-0.0073, -0.0129, 0.0069, 0.0060],
                                   [-0.0000, -0.0000, 0.0174, 0.0210]]])
    y1 = layer1(hidden_states, attention_mask)
    y2 = layer2(hidden_states, attention_mask)

    torch.testing.assert_allclose(y1, y2)


