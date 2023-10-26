# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Transformer."""
import math
from contextlib import nullcontext
from typing import Callable

import torch
import flash_attn
from torch.nn import functional as F
from einops import rearrange

from megatron import core, get_num_microbatches
from .module import MegatronModule
import megatron.core
from megatron.model.enums import AttnMaskType, ModelType, LayerType, AttnType, PositionEmbeddingType
from megatron.model import LayerNorm
from megatron.model import RMSNorm
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.utils import attention_mask_func, erf_gelu

# Extracted from: https://github.com/bigscience-workshop/Megatron-DeepSpeed
from .glu_activations import GLU_ACTIVATIONS
from megatron.model.positional_embeddings import precompute_freqs_cis, apply_rotary_emb


""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
"""

class DropPath(MegatronModule):
    """Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_state):
        if self.drop_prob == 0. or not self.training:
            return hidden_state
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        # hidden_state: [s, b, h]
        shape = (1,) + (hidden_state.shape[1],) + (1,) * (hidden_state.ndim - 2)
        random_tensor = keep_prob + \
            torch.rand(shape, dtype=hidden_state.dtype, device=hidden_state.device)
        random_tensor.floor_()  # binarize
        output = hidden_state.div(keep_prob) * random_tensor
        return output


def _args_to_kwargs(args):
    common_kwargs = {
        "params_dtype": args.params_dtype,
        "use_cpu_initialization": args.use_cpu_initialization,
        "perform_initialization": args.perform_initialization,
        "gradient_accumulation_fusion": args.gradient_accumulation_fusion,
        "sequence_parallel_enabled": args.sequence_parallel,
    }
    return common_kwargs


class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self,
                 init_method,
                 output_layer_init_method,
                 args,
                 world_size):
        super(ParallelMLP, self).__init__()
        # Project to 4h.
        self.dense_h_to_4h = megatron.core.tensor_parallel.ColumnParallelLinear(
            args.hidden_size,
            # GLU is a special activation that divides the dimension by a factor 2.
            2 * args.ffn_hidden_size if args.glu_activation else args.ffn_hidden_size,
            bias=args.use_bias,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
            **_args_to_kwargs(args),
            world_size=world_size)
        self.use_bias = args.use_bias

        self.bias_gelu_fusion = args.bias_gelu_fusion

        if args.glu_activation:
            self.activation_func = GLU_ACTIVATIONS[args.glu_activation]
        elif args.onnx_safe:
            self.activation_func = erf_gelu
        else:
            self.activation_func = F.gelu

        # Project back to h.
        self.dense_4h_to_h = megatron.core.tensor_parallel.RowParallelLinear(
            args.ffn_hidden_size,
            args.hidden_size,
            bias=args.use_bias,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            **_args_to_kwargs(args),
            world_size=world_size)

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.bias_gelu_fusion:
            intermediate_parallel = \
                bias_gelu_impl(intermediate_parallel, bias_parallel)
        elif self.use_bias:
            intermediate_parallel = \
                self.activation_func(intermediate_parallel + bias_parallel)
        else:
            intermediate_parallel = \
                self.activation_func(intermediate_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class CoreAttention(MegatronModule):
    def __init__(self,
                 layer_number,
                 attn_mask_type=AttnMaskType.padding,
                 args=None,
                 world_size=None):
        super(CoreAttention, self).__init__()
        self.fp16 = args.fp16
        self.bf16 = args.bf16

        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.sequence_parallel = args.sequence_parallel

        projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = core.utils.divide(projection_size,
                                                           world_size)
        self.hidden_size_per_attention_head = core.utils.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = core.utils.divide(
            args.num_attention_heads, world_size)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            self.attn_mask_type,
            args.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(args.attention_dropout)

    def forward(self, query_layer, key_layer,
                value_layer, attention_mask):

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = megatron.core.mpu.get_global_memory_buffer().get_tensor(
            (output_size[0]*output_size[1], output_size[2], output_size[3]),
            query_layer.dtype, "mpu")

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer.transpose(0, 1),   # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0, alpha=(1.0/self.norm_factor))

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.sequence_parallel:
            with megatron.core.tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       query_layer.size(0),
                       value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self,
                 init_method,
                 output_layer_init_method,
                 layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding,
                 world_size: int=None,
                 args=None):
        super(ParallelAttention, self).__init__()
        assert world_size is not None

        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.params_dtype = args.params_dtype
        self.sequence_parallel = args.sequence_parallel
        self.use_flash_attn = args.use_flash_attn
        self.num_attention_heads_kv = args.num_attention_heads_kv
        self.num_attention_heads = args.num_attention_heads
        self.seq_length = args.seq_length
        if self.use_flash_attn:
            assert attention_type == AttnType.self_attn, ('FlashAttention code path only supports '
                                                          'self-attention for now')
            assert self.attn_mask_type == AttnMaskType.causal, ('FlashAttention code path only '
                                                                'supports causal mask for now')

        projection_size = args.kv_channels * args.num_attention_heads

        qkv_projection_size = args.kv_channels * args.num_attention_heads + 2 * args.kv_channels * args.num_attention_heads_kv

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = core.utils.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = core.utils.divide(
            args.num_attention_heads, world_size)

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.query_key_value = megatron.core.tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                qkv_projection_size,
                bias=args.use_bias,
                gather_output=False,
                init_method=init_method,
                async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
                **_args_to_kwargs(args),
                world_size=world_size)
        else:
            assert attention_type == AttnType.cross_attn
            self.query = megatron.core.tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                projection_size,
                bias=args.use_bias,
                gather_output=False,
                init_method=init_method,
                async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
                **_args_to_kwargs(args),
                world_size=world_size)

            self.key_value = megatron.core.tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                2 * projection_size,
                bias=args.use_bias,
                gather_output=False,
                init_method=init_method,
                async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
                **_args_to_kwargs(args),
                world_size=world_size,)

        self.core_attention = CoreAttention(self.layer_number, self.attn_mask_type, args, world_size)
        self.checkpoint_core_attention = args.recompute_granularity == 'selective'

        if self.use_flash_attn:
            self.core_attention_flash = flash_attn.flash_attn_func

        # Output.
        self.dense = megatron.core.tensor_parallel.RowParallelLinear(
            projection_size,
            args.hidden_size,
            bias=args.use_bias,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            **_args_to_kwargs(args),
            world_size=world_size)
        
        self.position_embedding_type = args.position_embedding_type
        if self.position_embedding_type == PositionEmbeddingType.rotary:
            self.freqs_cis = precompute_freqs_cis(
                dim=args.hidden_size // args.num_attention_heads,
                end=self.seq_length,
                theta=args.rope_theta,
                scaling_factor=args.rope_scaling_factor,
            )

    def _checkpointed_attention_forward(self,
                                        query_layer,
                                        key_layer,
                                        value_layer,
                                        attention_mask):
        """Forward method with activation checkpointing."""
        def custom_forward(*inputs):
            query_layer = inputs[0]
            key_layer = inputs[1]
            value_layer = inputs[2]
            attention_mask = inputs[3]
            output_ = self.core_attention(query_layer, key_layer,
                                          value_layer, attention_mask)
            return output_

        hidden_states = megatron.core.tensor_parallel.checkpoint(
            custom_forward,
            False, query_layer, key_layer, value_layer, attention_mask)

        return hidden_states

    def _allocate_memory(self, inference_max_sequence_len, batch_size):
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
            dtype=self.params_dtype,
            device=torch.cuda.current_device())

    def forward(self,
                hidden_states,
                attention_mask,
                encoder_output=None,
                inference_params=None,
                position_ids=None):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        if inference_params:
            if self.layer_number not in inference_params.key_value_memory_dict:
                inf_max_seq_len = inference_params.max_sequence_len
                inf_max_batch_size = inference_params.max_batch_size
                inference_key_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size)
                inference_value_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size)
                inference_params.key_value_memory_dict[self.layer_number] = (
                    inference_key_memory, inference_value_memory)
            else:
                inference_key_memory, inference_value_memory = \
                    inference_params.key_value_memory_dict[self.layer_number]

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn:
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            sq, b = mixed_x_layer.shape[:2]
            # , we simply expand smaller keys and values tensors to have the usual shapes and then
            # feed those tensor to the standard attention/flash attention
            qkv = mixed_x_layer.view(sq, b, -1, self.num_attention_heads // self.num_attention_heads_kv + 2, self.hidden_size_per_attention_head)
            query_layer = qkv[:, :, :, :-2]
            key_layer = qkv[:, :, :, [-2]]
            value_layer = qkv[:, :, :, [-1]]
            key_layer = torch.broadcast_to(key_layer, query_layer.shape)
            value_layer = torch.broadcast_to(value_layer, query_layer.shape)
            query_layer, key_layer, value_layer = [rearrange(x, "seq_len batch group num_heads head_dim -> seq_len batch (group num_heads) head_dim",
                                                             head_dim=self.hidden_size_per_attention_head,) for x in [query_layer, key_layer, value_layer]]
        else:
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 2 * self.hidden_size_per_attention_head)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer,
             value_layer) = megatron.core.tensor_parallel.split_tensor_along_last_dim(mixed_kv_layer, 2)

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head)
            query_layer = query_layer.view(*new_tensor_shape)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if inference_params:
            batch_start = inference_params.batch_size_offset
            batch_end = batch_start + key_layer.size(1)
            assert batch_end <= inference_key_memory.size(1)
            sequence_start = inference_params.sequence_len_offset
            sequence_end = sequence_start + key_layer.size(0)
            assert sequence_end <= inference_key_memory.size(0)
            # Copy key and values.
            inference_key_memory[sequence_start:sequence_end,
                                 batch_start:batch_end, ...] = key_layer
            inference_value_memory[sequence_start:sequence_end,
                                   batch_start:batch_end, ...] = value_layer
            key_layer = inference_key_memory[:sequence_end, batch_start:batch_end, ...]
            value_layer = inference_value_memory[:sequence_end, batch_start:batch_end, ...]
        
        # ==================================
        # Rotary embeddings
        # ==================================
        if self.position_embedding_type == PositionEmbeddingType.rotary:
            query_layer, key_layer = apply_rotary_emb(query_layer, key_layer, self.freqs_cis, position_ids=position_ids)

        # ==================================
        # core attention computation
        # ==================================

        if not self.use_flash_attn:
            if self.checkpoint_core_attention:
                context_layer = self._checkpointed_attention_forward(
                    query_layer, key_layer, value_layer, attention_mask)
            else:
                context_layer = self.core_attention(
                    query_layer, key_layer, value_layer, attention_mask)
        else:
            q, k, v = [rearrange(x, "s b n h -> b s n h").contiguous()
                       for x in (query_layer, key_layer, value_layer)]
            if not self.sequence_parallel:
                with megatron.core.tensor_parallel.get_cuda_rng_tracker().fork():
                    context_layer = self.core_attention_flash(q, k, v, causal=True)
            else:
                context_layer = self.core_attention_flash(q, k, v, causal=True)
            context_layer = rearrange(context_layer, 'b s n h -> s b (n h)').contiguous()

        # =================
        # Output. [sq, b, h]
        # =================
        # print(self.dense)
        output, bias = self.dense(context_layer)
        return output, bias


def dropout_add(x, residual, prob, training):
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add



def dropout_add(x, residual, prob, training):
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out


def get_dropout_add(training):
    def _dropout_add(x, residual, prob):
        return dropout_add(x, residual, prob, training)
    return _dropout_add



@torch.jit.script
def bias_dropout_add_fused_train(x: torch.Tensor,
                                 bias: torch.Tensor,
                                 residual: torch.Tensor,
                                 prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x: torch.Tensor,
                                     bias: torch.Tensor,
                                     residual: torch.Tensor,
                                     prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, False)


class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.
    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self,
                 init_method: Callable,
                 output_layer_init_method: Callable,
                 layer_number: int,
                 layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 drop_path_rate: float=0.0,
                 world_size: int=None,
                 hidden_dropout: float=0.0,
                 args=None):
        super(ParallelTransformerLayer, self).__init__()

        self.layer_number = layer_number
        self.layer_type = layer_type
        self.apply_residual_connection_post_layernorm = args.apply_residual_connection_post_layernorm
        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection
        self.parallel_layernorm = args.parallel_layernorm

        # Layernorm on the input data.
        if args.use_rms_norm:
            self.input_layernorm = RMSNorm(args.hidden_size, eps=args.layernorm_epsilon,
                                           sequence_parallel=args.sequence_parallel)
            self.output_layernorm = RMSNorm(args.hidden_size, eps=args.layernorm_epsilon,
                                           sequence_parallel=args.sequence_parallel)
            if self.parallel_layernorm:
                self.mlp_layernorm = RMSNorm(args.hidden_size, eps=args.layernorm_epsilon,
                                             sequence_parallel=args.sequence_parallel)
        else:
            self.input_layernorm = LayerNorm(args.hidden_size,
                                             eps=args.layernorm_epsilon,
                                             no_persist_layer_norm=args.no_persist_layer_norm,
                                             sequence_parallel=args.sequence_parallel)
            self.output_layernorm = LayerNorm(args.hidden_size,
                                             eps=args.layernorm_epsilon,
                                             no_persist_layer_norm=args.no_persist_layer_norm,
                                             sequence_parallel=args.sequence_parallel)
            if self.parallel_layernorm:
                self.mlp_layernorm = LayerNorm(args.hidden_size,
                                               eps=args.layernorm_epsilon,
                                               no_persist_layer_norm=args.no_persist_layer_norm,
                                               sequence_parallel=args.sequence_parallel)
        self.use_post_ln = args.use_post_ln
        if args.use_post_ln:
            self.input_layernorm = torch.nn.Identity()
        else:
            self.output_layernorm = torch.nn.Identity()

        # Self attention.
        self.self_attention = ParallelAttention(
            init_method,
            output_layer_init_method,
            layer_number,
            attention_type=AttnType.self_attn,
            attn_mask_type=self_attn_mask_type,
            world_size=world_size,
            args=args)
        self.hidden_dropout = hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None
        self.parallel_attn = args.parallel_attn
        self.use_bias = args.use_bias

        # Layernorm on the attention output
        if not args.parallel_attn:
            if not args.use_rms_norm:
                self.post_attention_layernorm = LayerNorm(
                    args.hidden_size,
                    eps=args.layernorm_epsilon,
                    no_persist_layer_norm=args.no_persist_layer_norm,
                    sequence_parallel=args.sequence_parallel)
            else:
                self.post_attention_layernorm = RMSNorm(
                    args.hidden_size,
                    eps=args.layernorm_epsilon,
                    sequence_parallel=args.sequence_parallel)

        if self.layer_type == LayerType.decoder:
            self.inter_attention = ParallelAttention(
                init_method,
                output_layer_init_method,
                layer_number,
                attention_type=AttnType.cross_attn,
                world_size=world_size,
                args=args)
            # Layernorm on the attention output.
            if not args.use_rms_norm:
                self.post_inter_attention_layernorm = LayerNorm(
                    args.hidden_size,
                    eps=args.layernorm_epsilon,
                    no_persist_layer_norm=args.no_persist_layer_norm,
                    sequence_parallel=args.sequence_parallel)
            else:
                self.post_inter_attention_layernorm = RMSNorm(
                    args.hidden_size,
                    eps=args.layernorm_epsilon,
                    sequence_parallel=args.sequence_parallel)

        self.mlp = ParallelMLP(init_method, output_layer_init_method, args, world_size)

        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = \
                nullcontext if use_nvfuser else torch.enable_grad

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                encoder_output=None,
                enc_dec_attn_mask=None,
                inference_params=None,
                position_ids=None):

        ##
        # PRELIMINARIES - utilities to compute residual + dropout
        ##

        # function to compute residual + dropout(x + bias)
        def add_dropout(x, bias, residual, prob, make_viewless=False):
            # Jit compiled function creates 'view' tensor. This tensor
            # potentially gets saved in the MPU checkpoint function context,
            # which rejects view tensors. While making a viewless tensor here
            # won't result in memory savings (like the data loader, or
            # p2p_communication), it serves to document the origin of this
            # 'view' tensor.
            if self.use_bias:
                bias = bias.expand_as(residual)
                if self.drop_path is None:
                    with self.bias_dropout_add_exec_handler():
                        output = bias_dropout_add_func(x, bias, residual, prob)
                    if make_viewless:
                        return core.utils.make_viewless_tensor(inp = output,
                                                               requires_grad = output.requires_grad,
                                                               keep_graph = True)
                    return output
                out = torch.nn.functional.dropout(x + bias, p=prob, training=self.training)
                return residual + self.drop_path(out)
            elif self.drop_path is None:
                with self.bias_dropout_add_exec_handler():
                    return dropout_add_func(x, residual, prob)
            out = torch.nn.functional.dropout(x, p=prob, training=self.training)
            return residual + self.drop_path(out)

        # determine the dropout_add_func to use in the add_dropout function
        if self.drop_path is None:
            # jit scripting for a nn.module (with dropout) is not
            # triggerring the fusion kernel. For now, we use two
            # different nn.functional routines to account for varying
            # dropout semantics during training and inference phases.
            if not self.use_bias:
                dropout_add_func = get_dropout_add(self.training)
            elif self.bias_dropout_fusion:
                if self.training:
                    bias_dropout_add_func = bias_dropout_add_fused_train
                else:
                    bias_dropout_add_func = bias_dropout_add_fused_inference
            else:
                bias_dropout_add_func = get_bias_dropout_add(self.training)

        ##
        # Transformer computation begins now.
        ##

        # hidden_states: [s, b, h]
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Get attention.
        attention_output, attention_bias = self.self_attention(layernorm_output,
                                                               attention_mask,
                                                               inference_params=inference_params,
                                                               position_ids=position_ids)

        # Determines the value of the next residual connection.
        # if not parallel_attn: used after the post_attention_layernorm,
        # else: used just before returning the output.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # dedicated mlp layernorm module
        if self.parallel_layernorm:
            layernorm_output = self.mlp_layernorm(hidden_states)

        if self.parallel_attn:
            # used only if layer is decoder and not residual_post_layernorm
            # which seems a bit strange, but it's kept just in case for now
            layernorm_input = attention_output
        else:
            layernorm_input = add_dropout(attention_output, attention_bias,
                                         residual, self.hidden_dropout)
            layernorm_output = self.post_attention_layernorm(layernorm_input)

        if self.layer_type == LayerType.decoder:
            attention_output, attention_bias = self.inter_attention(layernorm_output,
                                                                    enc_dec_attn_mask,
                                                                    encoder_output=encoder_output)
            if self.apply_residual_connection_post_layernorm:
                residual = layernorm_output
            else:
                residual = layernorm_input

            layernorm_input = add_dropout(attention_output, attention_bias,
                                          residual, self.hidden_dropout)
            # Layer norm post the decoder attention
            layernorm_output = self.post_inter_attention_layernorm(layernorm_input)
        # Compute MLP.
        # At this point, layernorm_output is:
        # if layer is decoder: the post_inter_attention_layernorm output,
        # elif parallel_layernorm: the mlp_layernorm output,
        # elif parallel_attention: the input_layernorm tensor.
        # else: the post_attention_layernorm output,
        mlp_output, mlp_bias = self.mlp(layernorm_output)

        # Second residual connection.
        if self.parallel_attn:
            mlp_output = mlp_output + attention_output
        elif self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input
        output = add_dropout(mlp_output, mlp_bias, residual, self.hidden_dropout,
                             make_viewless=True)

        # Apply final layernorm, return.
        output = self.output_layernorm(output)
        return output


class NoopTransformerLayer(MegatronModule):
    """A single 'no-op' transformer layer.

    The sole purpose of this layer is for when a standalone embedding layer
    is used (i.e., args.standalone_embedding_stage == True). In this case,
    zero transformer layers are assigned when pipeline rank == 0. Additionally,
    when virtual pipeline rank >= 1, zero total model parameters are created
    (virtual rank 0 contains the input embedding). This results in the model's
    input and output tensors being the same, which causes an error when
    performing certain memory optimiations on the output tensor (e.g.,
    deallocating it). Thus, this layer disconnects the input from the output
    via a clone. Since ranks containing a no-op layer are generally under-
    utilized (both compute and memory), there's no worry of any performance
    degredation.
    """

    def __init__(self, layer_number):
        super().__init__()
        self.layer_number = layer_number

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None):
        return hidden_states.clone()


def _get_num_layers(args, is_encoder_and_decoder_model, is_decoder=False):
    """Compute the number of transformer layers resident on the current rank."""
    if megatron.core.mpu.get_pipeline_model_parallel_world_size() > 1:
        if is_encoder_and_decoder_model:
            assert args.pipeline_model_parallel_split_rank is not None

            # When a standalone embedding stage is used, a rank is taken from
            # the encoder's ranks, to be used for the encoder's embedding
            # layer. This way, the rank referenced by the 'split rank' remains
            # the same whether or not a standalone embedding stage is used.
            num_ranks_in_encoder = (
                args.pipeline_model_parallel_split_rank - 1
                if args.standalone_embedding_stage else
                args.pipeline_model_parallel_split_rank
            )
            num_ranks_in_decoder = args.transformer_pipeline_model_parallel_size - num_ranks_in_encoder
            assert args.encoder_num_layers % num_ranks_in_encoder == 0, \
                    'encoder_num_layers (%d) must be divisible by number of ranks given to encoder (%d)' % (args.encoder_num_layers, num_ranks_in_encoder)
            assert args.decoder_num_layers % num_ranks_in_decoder == 0, \
                    'decoder_num_layers (%d) must be divisible by number of ranks given to decoder (%d)' % (args.decoder_num_layers, num_ranks_in_decoder)
            if megatron.core.mpu.is_pipeline_stage_before_split():
                num_layers = (
                    0
                    if args.standalone_embedding_stage
                       and megatron.core.mpu.get_pipeline_model_parallel_rank() == 0 else
                    args.encoder_num_layers // num_ranks_in_encoder
                )
            else:
                num_layers = args.decoder_num_layers // num_ranks_in_decoder
        else:
            assert args.num_layers == args.encoder_num_layers
            assert args.num_layers % args.transformer_pipeline_model_parallel_size == 0, \
                'num_layers must be divisible by transformer_pipeline_model_parallel_size'

            # When a standalone embedding stage is used, all transformer layers
            # are divided among pipeline rank >= 1, while on pipeline rank 0,
            # ranks either contain the input embedding layer (virtual pp rank 0),
            # or no layers at all (virtual pp rank >= 1).
            num_layers = (
                0
                if args.standalone_embedding_stage
                   and megatron.core.mpu.get_pipeline_model_parallel_rank() == 0 else
                args.num_layers // args.transformer_pipeline_model_parallel_size
            )
    else:
        if not is_decoder:
            num_layers = args.encoder_num_layers
        else:
            num_layers = args.decoder_num_layers
    return num_layers


class ParallelTransformer(MegatronModule):
    def __init__(self,
                 init_method: Callable,
                 output_layer_init_method,
                 layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 pre_process=True,
                 post_process=True,
                 drop_path_rate=0.0,
                 args=None,
                 model_type=None):
        super(ParallelTransformer, self).__init__()
        world_size = megatron.core.mpu.get_tensor_model_parallel_world_size()
        assert args is not None
        assert model_type is not None

        self.layer_type = layer_type
        self.model_type = model_type
        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.drop_path_rate = drop_path_rate
        self.transformer_impl = args.transformer_impl

        # Store activation checkpointing flag.
        self.recompute_granularity = args.recompute_granularity
        self.recompute_method = args.recompute_method
        self.recompute_num_layers = args.recompute_num_layers
        self.distribute_saved_activations = \
            args.distribute_saved_activations and not args.sequence_parallel

        self.sequence_parallel = args.sequence_parallel

        # Transformer Engine Init.
        if self.transformer_impl == 'transformer_engine':
            global transformer_engine
            import transformer_engine
        self.use_fp8 = args.fp8_e4m3 or args.fp8_hybrid
        self.fp8_recipe = None
        self.fp8_group = megatron.core.mpu.get_data_parallel_group()
        if self.use_fp8:
            if args.fp8_e4m3:
                fp8_format = transformer_engine.common.recipe.Format.E4M3
            elif args.fp8_hybrid:
                fp8_format = transformer_engine.common.recipe.Format.HYBRID
            self.fp8_recipe = transformer_engine.common.recipe.DelayedScaling(
                margin=args.fp8_margin,
                interval=args.fp8_interval,
                fp8_format=fp8_format,
                amax_history_len=args.fp8_amax_history_len,
                amax_compute_algo=args.fp8_amax_compute_algo,
                override_linear_precision=(False, False, not args.fp8_wgrad),
            )

        self.num_microbatches_in_previous_step = -1
        self.microbatch_count = 0
        self.checkpoint_core_attention = args.recompute_granularity == 'selective'

        # Number of layers.
        self.num_layers = _get_num_layers(args,
            model_type == ModelType.encoder_and_decoder,
            layer_type == LayerType.decoder)

        self.drop_path_rates = [rate.item() for rate in torch.linspace(0, self.drop_path_rate, args.num_layers)]

        if args.lima_dropout:
            # Use a layer dependent dropout probability, starting at p_d=0.0 at the bottom layer
            # and linearly raising the rate to the value specified by `args.hidden_dropout` at the last layer.
            # see "LIMA: Less Is More for Alignment", Zhou et al 2023, https://arxiv.org/abs/2305.11206
            self.hidden_dropouts = [rate.item() for rate in torch.linspace(0, args.hidden_dropout, args.num_layers)]
        else:
            # Use standard residual dropout with the same dropout probability for all layers.
            self.hidden_dropouts = [args.hidden_dropout] * args.num_layers

        # Transformer layers.
        def build_layer(layer_number: int):
            if args.transformer_impl == 'local':
                return ParallelTransformerLayer(
                    init_method,
                    output_layer_init_method,
                    layer_number,
                    layer_type=layer_type,
                    self_attn_mask_type=self_attn_mask_type,
                    drop_path_rate=self.drop_path_rates[layer_number - 1],
                    world_size=world_size,
                    hidden_dropout=self.hidden_dropouts[layer_number - 1],
                    args=args)
            else:
                return transformer_engine.pytorch.TransformerLayer(
                    args.hidden_size,
                    args.ffn_hidden_size,
                    args.num_attention_heads,
                    layernorm_epsilon=args.layernorm_epsilon,
                    hidden_dropout=self.hidden_dropouts[layer_number - 1],
                    attention_dropout=args.attention_dropout,
                    init_method=init_method,
                    output_layer_init_method=output_layer_init_method,
                    layer_number=layer_number,
                    kv_channels=args.kv_channels,
                    self_attn_mask_type=self_attn_mask_type.name,
                    tp_group=megatron.core.mpu.get_tensor_model_parallel_group(),
                    get_rng_state_tracker=megatron.core.tensor_parallel.get_cuda_rng_tracker,
                    fuse_wgrad_accumulation=args.gradient_accumulation_fusion,
                    apply_query_key_layer_scaling=args.apply_query_key_layer_scaling,
                    attention_softmax_in_fp32=args.attention_softmax_in_fp32,
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    sequence_parallel=args.sequence_parallel,
                    params_dtype=args.params_dtype,
                    apply_residual_connection_post_layernorm=args.apply_residual_connection_post_layernorm,
                    output_layernorm=False,
                    layer_type="encoder",
                    drop_path_rate=self.drop_path_rates[layer_number - 1],
                    set_parallel_mode=True,
                    fuse_qkv_params=True)

        if args.virtual_pipeline_model_parallel_size is not None:
            assert args.num_layers % args.virtual_pipeline_model_parallel_size == 0, \
                'num_layers_per_stage must be divisible by ' \
                'virtual_pipeline_model_parallel_size'
            assert args.model_type != ModelType.encoder_and_decoder
            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = self.num_layers // args.virtual_pipeline_model_parallel_size
            # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset = megatron.core.mpu.get_virtual_pipeline_model_parallel_rank() * (
                args.num_layers // args.virtual_pipeline_model_parallel_size) + \
                     (megatron.core.mpu.get_pipeline_model_parallel_rank() * self.num_layers)
        else:
            # Each stage gets a contiguous set of layers.
            if model_type == ModelType.encoder_and_decoder and \
                    megatron.core.mpu.get_pipeline_model_parallel_world_size() > 1:
                pipeline_rank = megatron.core.mpu.get_pipeline_model_parallel_rank()
                if layer_type == LayerType.encoder:
                    offset = pipeline_rank * self.num_layers
                else:
                    num_ranks_in_enc = args.pipeline_model_parallel_split_rank
                    offset = (pipeline_rank - num_ranks_in_enc) * self.num_layers
            else:
                offset = megatron.core.mpu.get_pipeline_model_parallel_rank() * self.num_layers

        if self.num_layers == 0:
            # When a standalone embedding stage is used (e.g.,
            # args.standalone_embedding_stage == True), virtual pipeline ranks
            # on pipeline rank 0 will have zero transformer layers assigned to
            # them. This results in the model's input and output tensors to be
            # the same, which will cause failure for certain output tensor
            # optimizations (e.g., pipeline output deallocation). To remedy
            # this, we assign a 'no-op' layer on these ranks, which will
            # disconnect the input tensor from the output tensor.
            self.num_layers = 1
            self.layers = torch.nn.ModuleList([NoopTransformerLayer(1)])
        else:
            self.layers = torch.nn.ModuleList(
                [build_layer(i + 1 + offset) for i in range(self.num_layers)])

        self.use_post_ln = args.use_post_ln
        if self.post_process:
            # Final layer norm before output.
            if not args.use_rms_norm:
                self.final_layernorm = LayerNorm(
                    args.hidden_size,
                    eps=args.layernorm_epsilon,
                    no_persist_layer_norm=args.no_persist_layer_norm,
                    sequence_parallel=args.sequence_parallel)
            else:
                self.final_layernorm = RMSNorm(
                    args.hidden_size,
                    eps=args.layernorm_epsilon,
                    sequence_parallel=args.sequence_parallel)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def _checkpointed_forward(self, hidden_states, attention_mask,
                              encoder_output, enc_dec_attn_mask,
                              inference_params, position_ids,
                              is_first_microbatch
                              ):
        """Forward method with activation checkpointing."""
        def custom(start, end, is_transformer_engine=False):
            def custom_forward(*args, **kwargs):
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(*args, **kwargs)
                return x_

            def custom_forward_transformer_engine(*args, **kwargs):
                return custom_forward(*args, is_first_microbatch=is_first_microbatch, **kwargs)
            if not is_transformer_engine:
                return custom_forward
            else:
                return custom_forward_transformer_engine

        if self.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            l = 0
            while l < self.num_layers:
                if self.transformer_impl == 'transformer_engine':
                    hidden_states = transformer_engine.pytorch.distributed.checkpoint(
                        custom(l, l + self.recompute_num_layers, is_transformer_engine=True),
                        self.distribute_saved_activations,
                        megatron.core.tensor_parallel.get_cuda_rng_tracker,
                        megatron.core.mpu.get_tensor_model_parallel_group(),
                        hidden_states, attention_mask, encoder_output, enc_dec_attn_mask,
                        inference_params, position_ids
                    )
                else:
                    hidden_states = megatron.core.tensor_parallel.checkpoint(
                        custom(l, l + self.recompute_num_layers),
                        self.distribute_saved_activations,
                        hidden_states, attention_mask, encoder_output, enc_dec_attn_mask,
                        inference_params, position_ids
                    )

                l += self.recompute_num_layers

        elif self.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            for l in range(self.num_layers):
                if l < self.recompute_num_layers:
                    if self.transformer_impl == 'transformer_engine':
                        hidden_states = transformer_engine.pytorch.distributed.checkpoint(
                            custom(l, l + 1, is_transformer_engine=True),
                            self.distribute_saved_activations,
                            megatron.core.tensor_parallel.get_cuda_rng_tracker,
                            megatron.core.mpu.get_tensor_model_parallel_group(),
                            hidden_states, attention_mask, encoder_output, enc_dec_attn_mask,
                            inference_params, position_ids
                            )
                    else:
                        hidden_states = megatron.core.tensor_parallel.checkpoint(
                            custom(l, l + 1),
                            self.distribute_saved_activations,
                            hidden_states, attention_mask, encoder_output, enc_dec_attn_mask,
                            inference_params, position_ids
                        )
                else:
                    if self.transformer_impl == 'transformer_engine':
                        hidden_states = custom(l, l + 1, is_transformer_engine=True)(
                            hidden_states, attention_mask, encoder_output, enc_dec_attn_mask,
                            inference_params, position_ids
                        )
                    else:
                        hidden_states = custom(l, l + 1)(
                            hidden_states, attention_mask, encoder_output, enc_dec_attn_mask,
                            inference_params, position_ids
                        )
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                encoder_output=None,
                enc_dec_attn_mask=None,
                inference_params=None,
                position_ids=None):

        # hidden_states: [s, b, h]

        # Checks.
        if inference_params:
            assert self.recompute_granularity is None, \
                'inference does not work with activation checkpointing'

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = core.utils.make_viewless_tensor(
            hidden_states,
            requires_grad=True,
            keep_graph=True,
        )

        if self.sequence_parallel:
            rng_context = megatron.core.tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        with rng_context:
            # The fp8_autocast context manager is a no-op when enabled=True
            # The if...else serves to short circuit name resolution for fp8_autocast
            with transformer_engine.pytorch.fp8_autocast(
                enabled=self.use_fp8,
                fp8_recipe=self.fp8_recipe,
                fp8_group=self.fp8_group
            ) if self.use_fp8 else nullcontext():
                # Determine if the current iteration is first microbatch
                if self.num_microbatches_in_previous_step != get_num_microbatches():
                    self.microbatch_count = 0 # Reset count on new batch size rampup interval
                self.num_microbatches_in_previous_step = get_num_microbatches()
                is_first_microbatch = self.microbatch_count % get_num_microbatches() == 0

                # Forward pass.
                if self.recompute_granularity == 'full':
                    hidden_states = self._checkpointed_forward(hidden_states,
                                                               attention_mask,
                                                               encoder_output,
                                                               enc_dec_attn_mask,
                                                               inference_params,
                                                               position_ids,
                                                               is_first_microbatch)
                else:
                    forward_kwargs = {
                        'encoder_output': encoder_output,
                        'enc_dec_attn_mask': enc_dec_attn_mask,
                        'inference_params': inference_params,
                        'position_ids': position_ids,
                    }

                    if self.transformer_impl == 'transformer_engine':
                        forward_kwargs['is_first_microbatch'] = is_first_microbatch
                        forward_kwargs['checkpoint_core_attention'] = self.checkpoint_core_attention

                    for index in range(self.num_layers):
                        layer = self._get_layer(index)

                        hidden_states = layer(
                            hidden_states,
                            attention_mask,
                            **forward_kwargs)

                # Skip counter update for eval and activation checkpointing
                if torch.is_grad_enabled() and self.training:
                    self.microbatch_count += 1

        # Final layer norm.
        # not done for the "post_ln" convention https://sh-tsang.medium.com/review-pre-ln-transformer-on-layer-normalization-in-the-transformer-architecture-b6c91a89e9ab
        if self.post_process and (not self.use_post_ln):
            hidden_states = self.final_layernorm(hidden_states)
        return hidden_states
