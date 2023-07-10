import functools
import math
import warnings
from typing import Callable, Optional, Tuple, Union


from torch.profiler import profile, record_function, ProfilerActivity
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import einops

device = torch.device("cuda")

import megatron.model.positional_embeddings

torch.manual_seed(11)


class RotaryEmbedding(torch.nn.Module):
    """
    Implementation of RotaryEmbedding from GPT-NeoX.
    This implementation is design to operate on queries and keys that are compatible with
    [batch_size, n_heads_per_partition, seq_len, head_dim] (e.g. MinGPTAttention format).
    """
    def __init__(
        self,
        head_dim: int,
        base=10000,
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim
        self.seq_len_cached = None
        self.batch_size_cached = None
        self.cos_cached: torch.Tensor | None = None
        self.sin_cached: torch.Tensor | None = None

    def cos_sin(
        self,
        seq_len: int,
        device="cuda",
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(device)

            if dtype in [torch.float16, torch.bfloat16]:
                emb = emb.float()

            self.cos_cached = emb.cos()[None, :, :]
            self.sin_cached = emb.sin()[None, :, :]

            self.cos_cached = self.cos_cached.type(dtype)
            self.sin_cached = self.sin_cached.type(dtype)
        return self.cos_cached, self.sin_cached

    def forward(self, q, k):
        batch, seq_len, head_dim = q.shape
        cos, sin = self.cos_sin(seq_len, q.device, q.dtype)
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def _make_causal_mask(
    input_ids_shape: torch.Size, device: torch.device, past_key_values_length: int
) -> torch.BoolTensor:
    batch_size, target_length = input_ids_shape
    mask = torch.empty((target_length, target_length + past_key_values_length), dtype=torch.bool, device=device)
    # ONNX doesn't support `torch.Tensor.triu` properly, thus we use this workaround
    seq_ids = torch.arange(target_length, device=device)
    mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]

    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False

    expanded_mask = mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)
    return expanded_mask


def _expand_mask(mask: torch.Tensor, tgt_length: int) -> torch.BoolTensor:
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = ~(mask[:, None, None, :].to(torch.bool))
    return expanded_mask.expand(batch_size, 1, tgt_length, src_length)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.rotary_embedding = RotaryEmbedding(config.head_dim)

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = self.inv_norm_factor

        self.query_key_value = Linear(
            self.hidden_size,
            (config.n_head_kv * 2 + config.n_head) * self.head_dim,
            bias=config.bias,
        )
        self.dense = Linear(self.hidden_size, self.hidden_size, bias=config.bias)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.num_kv = config.n_head_kv

    def _split_heads(self, fused_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim), results share same memory
        storage as `fused_qkv`
        Args:
            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]
        Returns:
            query: [batch_size, seq_length, num_heads, head_dim]
            key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        batch, seq_len, _ = fused_qkv.shape
        qkv = fused_qkv.view(batch, seq_len, -1, self.num_heads // self.num_kv + 2, 64)
        q = qkv[:, :, :, :-2]
        k = qkv[:, :, :, [-2]]
        v = qkv[:, :, :, [-1]]
        k = torch.broadcast_to(k, q.shape)
        v = torch.broadcast_to(v, q.shape)

        q, k, v = [
            rearrange(
                _,
                "batch seq_len group num_heads head_dim ->\
                batch seq_len (group num_heads) head_dim"
            )
            for _ in [q, k, v]
        ]
        return q, k, v

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, q_length, _, _ = query_layer.shape
        query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        key_layer = key_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)

        query_layer, key_layer = self.rotary_embedding(query_layer, key_layer)

        query_layer_ = query_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
        key_layer_ = key_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
        value_layer_ = value_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)

        attn_output = F.scaled_dot_product_attention(query_layer_, key_layer_, value_layer_, None, 0.0, is_causal=True)

        x = attn_output.view(batch_size, self.num_heads, q_length, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        attn_output = x.reshape(batch_size, q_length, self.num_heads * self.head_dim)
        output_tensor = self.dense(attn_output)
        return output_tensor


class MatobaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.rotary_embedding = RotaryEmbeddingMatoba(self.head_dim, self.num_heads)

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = self.inv_norm_factor

        self.query_key_value = Linear(
            self.hidden_size,
            (config.n_head_kv * 2 + config.n_head) * self.head_dim,
            bias=config.bias,
        )
        self.dense = Linear(self.hidden_size, self.hidden_size, bias=config.bias)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.num_kv = config.n_head_kv

    def _split_heads(self, fused_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim), results share same memory
        storage as `fused_qkv`
        Args:
            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]
        Returns:
            query: [batch_size, seq_length, num_heads, head_dim]
            key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        batch, seq_len, _ = fused_qkv.shape
        qkv = fused_qkv.view(batch, seq_len, -1, self.num_heads // self.num_kv + 2, 64)
        q = qkv[:, :, :, :-2]
        k = qkv[:, :, :, [-2]]
        v = qkv[:, :, :, [-1]]
        k = torch.broadcast_to(k, q.shape)
        v = torch.broadcast_to(v, q.shape)

        q, k, v = [
            rearrange(
                x,
                "batch seq_len group num_heads head_dim ->\
                batch seq_len (group num_heads) head_dim"
            )
            for x in [q, k, v]
        ]
        return q, k, v

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)
        batch_size, q_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        key_layer = key_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        query_layer, key_layer = self.rotary_embedding(query_layer, key_layer)
        # query_layer
        query_layer = rearrange(query_layer, "i j k l -> (j k) i l")
        key_layer = rearrange(key_layer, "i j k l -> (j k) i l")

        query_layer_ = query_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
        key_layer_ = key_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
        value_layer_ = value_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)

        attn_output = F.scaled_dot_product_attention(query_layer_, key_layer_, value_layer_, None, 0.0, is_causal=True)
        x = attn_output.view(batch_size, self.num_heads, q_length, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        attn_output = x.reshape(batch_size, q_length, self.num_heads * self.head_dim)

        output_tensor = self.dense(attn_output)
        return output_tensor


def prepare_attn_mask(attention_mask: torch.Tensor,
                      input_shape: Tuple[int, int],
                      past_key_values_length: int) -> torch.BoolTensor:
    # create causal mask
    # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
    combined_attention_mask = None
    device = attention_mask.device
    _, src_length = input_shape

    if src_length > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape, device=device, past_key_values_length=past_key_values_length
        )

    # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
    expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
    combined_attention_mask = (
        expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
    )

    return combined_attention_mask


class Linear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        ret = input @ self.weight.T
        if self.bias is None:
            return ret
        else:
            return ret + self.bias


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 4 == ndim
    assert freqs_cis.shape == (x.shape[0], x.shape[-1])
    shape = [d if i == 0 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    __q = q.float().reshape(*q.shape[:-1], 2, -1).transpose(-1, -2)
    __k = k.float().reshape(*k.shape[:-1], 2, -1).transpose(-1, -2)
    xq_ = torch.view_as_complex(__q.contiguous())
    xk_ = torch.view_as_complex(__k.contiguous())
    # xq_ = torch.view_as_complex(__q)
    # xk_ = torch.view_as_complex(__k)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_).to(q.device)

    xq_out = torch.view_as_real(xq_ * freqs_cis).transpose(-1, -2).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).transpose(-1, -2).flatten(3)
    return xq_out.type_as(q), xk_out.type_as(k)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in torch < 1.8.0


@functools.lru_cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def _do_it(z: torch.Tensor,
           num_head: int,
           freqs_cis: torch.Tensor) -> torch.Tensor:
    # _ = torch.einsum("ijkl,il->ijkl",
    #     torch.view_as_complex(
    #     rearrange(z,
    #           "(batch num_heads) seq_len (ri half_head_dim) -> seq_len batch num_heads half_head_dim ri",
    #           num_heads=num_head, ri=2)
    #         .contiguous()
    # ), freqs_cis)
    # _ = torch.view_as_real(_)
    # # _ = torch.view_as_real(torch.einsum("ijkl,il->ijkl", _, freqs_cis))
    # # _ = _.transpose(-1, -2).flatten(3)
    # _ = rearrange(_, "i j k l m -> i j k (m l)")

    _ = rearrange(z,
              "(batch num_heads) seq_len (ri half_head_dim) -> seq_len batch num_heads half_head_dim ri",
              num_heads=num_head, ri=2)
    _ = torch.view_as_complex(_.contiguous())
    _ = torch.view_as_real(torch.einsum("ijkl,il->ijkl", _, freqs_cis))
    # _ = _.transpose(-1, -2).flatten(3)
    _ = rearrange(_, "i j k l m -> i j k (m l)")
    _ = _.type_as(z)
    return _


class RotaryEmbeddingMatoba(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        num_head: int,
        theta=10000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_head = num_head
        self.theta = theta

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        seq_length = q.shape[1]
        head_dim = q.shape[-1]
        freqs_cis = precompute_freqs_cis(self.head_dim, seq_length, self.theta).to(device)
        # new_method = True
        # new_method = False
        # if new_method:
        #     xq_out = _do_it(q, self.num_head, freqs_cis)
        #     xk_out = _do_it(k, self.num_head, freqs_cis)
        # else:
        new_method = True

        if new_method:
            k4 = rearrange(k,
                           "(batch num_heads) seq_len (ri half_head_dim) -> seq_len (batch num_heads half_head_dim) ri",
                           num_heads=self.num_head, ri=2)
            q4 = rearrange(q,
                           "(batch num_heads) seq_len (ri half_head_dim) -> seq_len (batch num_heads half_head_dim) ri",
                           num_heads=self.num_head, ri=2)
            xq_new = rearrange(torch.view_as_complex(q4), "seq_len (batch num_heads half_head_dim) ->  seq_len batch num_heads half_head_dim",
                          seq_len=seq_length, num_heads=self.num_head, half_head_dim = self.head_dim // 2)
            xk_new = rearrange(torch.view_as_complex(k4),  "seq_len (batch num_heads half_head_dim) ->  seq_len batch num_heads half_head_dim",
                          seq_len=seq_length, num_heads=self.num_head, half_head_dim = self.head_dim // 2)
            xq_ = xq_new
            xk_ = xk_new
        else:

            k3 = rearrange(k, "(batch num_heads) seq_len (ri half_head_dim) -> seq_len batch num_heads half_head_dim ri",
                                       num_heads=self.num_head, ri=2)
            q3 = rearrange(q, "(batch num_heads) seq_len (ri half_head_dim) -> seq_len batch num_heads half_head_dim ri",
                                       num_heads=self.num_head, ri=2)
            xq_ = torch.complex(q3[..., 0], q3[..., 1])
            xk_ = torch.complex(k3[..., 0], k3[..., 1])
        
        # torch.testing.assert_close(xk_, xk_new)
        # torch.testing.assert_close(xq_, xq_new)
        # k4_2 = rearrange(k4c, "seq_len (batch num_heads half_head_dim) ->  seq_len batch num_heads half_head_dim",
        #                  seq_len=seq_length, num_heads=self.num_head, half_head_dim = self.head_dim // 2)
        # torch.testing.assert_close(k4_2, xk_)
        # k4_2 = torch.einsum("ik,ki->ik", k4c, freqs_cis)

        xq_out_new = torch.view_as_real(torch.einsum("ijkl,il->ijkl", xq_, freqs_cis))
        xk_out_new = torch.view_as_real(torch.einsum("ijkl,il->ijkl", xk_, freqs_cis))

        xq_out = rearrange(xq_out_new, "i j k l m -> i j k (m l)").type_as(q)
        xk_out = rearrange(xk_out_new, "i j k l m -> i j k (m l)").type_as(k)
        return xq_out, xk_out


class RotaryEmbeddingMatoba2(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        num_head: int,
        theta=10000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_head = num_head
        self.theta = theta

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        qk = torch.stack((q, k), 0)
        seq_length = qk.shape[2]
        freqs_cis = precompute_freqs_cis(self.head_dim, seq_length, self.theta)
        qk3 = rearrange(qk, "i (batch num_heads) seq_len (ri half_head_dim) -> i seq_len batch num_heads half_head_dim ri",
                               num_heads=self.num_head, ri=2)
        xqk_ = torch.view_as_complex(qk3.contiguous())

        xqk_out_new = torch.view_as_real(torch.einsum("Aijkl,il->Aijkl", xqk_, freqs_cis))

        xqk_out = xqk_out_new.transpose(-1, -2).flatten(4).type_as(qk)
        return xqk_out[0], xqk_out[1]


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base=10000,
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim
        self.seq_len_cached = None
        self.batch_size_cached = None
        self.cos_cached: torch.Tensor | None = None
        self.sin_cached: torch.Tensor | None = None

    def cos_sin(
        self,
        seq_len: int,
        device="cuda",
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(device)

            if dtype in [torch.float16, torch.bfloat16]:
                emb = emb.float()

            self.cos_cached = emb.cos()[None, :, :]
            self.sin_cached = emb.sin()[None, :, :]

            self.cos_cached = self.cos_cached.type(dtype)
            self.sin_cached = self.sin_cached.type(dtype)

        return self.cos_cached, self.sin_cached

    def forward(self, q, k):
        batch, seq_len, head_dim = q.shape
        cos, sin = self.cos_sin(seq_len, q.device, q.dtype)
        return (q * cos) + (rotate_half(q) * sin), \
               (k * cos) + (rotate_half(k) * sin)


class DummyConf:
    n_head = 12
    # head_dim = 64
    # head_dim = 128
    head_dim = 512  # 256
    hidden_size = head_dim * n_head
    split_size = 9
    hidden_dropout = 0.0
    attention_dropout = 0.0
    n_head_kv = n_head
    bias = False


def method1(batch_size: int,
            q: torch.Tensor,
            k: torch.Tensor):
    nb, seq_length, head_dim = q.shape
    num_head = nb // batch_size
    falcon_rotary = RotaryEmbedding(head_dim)

    q_falcon, k_falcon = falcon_rotary(q, k)
    pattern = "batch num_heads seq_len head_dim -> seq_len batch num_heads head_dim"
    q_out = rearrange(q_falcon.view(batch_size, num_head, seq_length, head_dim), pattern)
    k_out = rearrange(k_falcon.view(batch_size, num_head, seq_length, head_dim), pattern)
    return q_out, k_out


def method2(batch_size: int,
            q: torch.Tensor,
            k: torch.Tensor):
    nb, seq_length, head_dim = q.shape
    num_head = nb // batch_size
    freqs_cis = precompute_freqs_cis(head_dim, seq_length)

    pattern = "batch num_heads seq_len head_dim -> seq_len batch num_heads head_dim"
    q = rearrange(q.view(batch_size, num_head, seq_length, head_dim), pattern)
    k = rearrange(k.view(batch_size, num_head, seq_length, head_dim), pattern)
    q_out, k_out = apply_rotary_emb(q, k, freqs_cis)
    return q_out, k_out


def method3(batch_size: int,
            q: torch.Tensor,
            k: torch.Tensor):
    nb, seq_length, head_dim = q.shape
    num_head = nb // batch_size
    matoba_rotary = RotaryEmbeddingMatoba(head_dim, num_head)
    q_out, k_out = matoba_rotary(q, k)
    return q_out, k_out


def method4(batch_size: int,
            q: torch.Tensor,
            k: torch.Tensor):
    nb, seq_length, head_dim = q.shape
    num_head = nb // batch_size
    matoba_rotary2 = RotaryEmbeddingMatoba2(head_dim, num_head)
    q_out, k_out = matoba_rotary2(q, k)
    return q_out, k_out


def benchmark_rotary_embeddings():
    # seq_length = 1024
    # seq_length = 2048
    seq_length = 128
    batch_size = 8
    conf = DummyConf()

    nb = batch_size * conf.n_head
    q = torch.randn(nb, seq_length, conf.head_dim, device=device)  # batch_size * self.num_heads, q_length, self.head_dim)
    k = torch.randn(nb, seq_length, conf.head_dim, device=device)

    methods = [method3, method2]
    # methods = [method1, method3, method2]
    # methods = [method1, method2, method3]
    # methods = [method1, method3, method2, method4]
    # methods = list(reversed([method1, method2, method3]))
    num_methods = len(methods)

    qs, ks = [None] * num_methods, [None] * num_methods
    for idx, method in enumerate(methods):
        bef = torch.cuda.Event(enable_timing=True)
        aft = torch.cuda.Event(enable_timing=True)

        q_, k_ = method(batch_size, torch.rand_like(q), torch.rand_like(k))

        # activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with profile(activities=activities,
                     profile_memory=True, record_shapes=True) as prof:
            bef.record()
            precompute_freqs_cis.cache_clear()
            q_, k_ = method(batch_size, q, k)
            aft.record()

        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

        # Waits for everything to finish running
        torch.cuda.synchronize()
        qs[idx] = q_
        ks[idx] = k_
        print(idx, method, bef.elapsed_time(aft))

    for idx in range(1, num_methods):
        torch.testing.assert_close(ks[0], ks[idx], rtol=0.001, atol=0.0001)
        torch.testing.assert_close(qs[0], qs[idx], rtol=0.001, atol=0.0001)


def benchmark_rotary_attention():
    seq_length = 1028
    # batch_size = 32
    batch_size = 64
    conf = DummyConf()

    attention_mask = torch.ones((batch_size, seq_length), device='cpu')

    causal_mask = prepare_attn_mask(
        attention_mask,
        input_shape=(batch_size, seq_length),
        past_key_values_length=0,
    )
    attn_orig = Attention(conf)
    attn_mato = MatobaAttention(conf)

    attn_mato.query_key_value = attn_orig.query_key_value
    attn_mato.dense = attn_orig.dense

    x = torch.randn((batch_size, seq_length, conf.hidden_size))

    f1 = functools.partial(attn_orig, x, causal_mask)
    f2 = functools.partial(attn_mato, x, causal_mask)

    y1, t1, res1 = profile_call(f1)
    y2, t2, res2 = profile_call(f2)
    torch.testing.assert_close(y1, y2)

    print(f"Falcon {t1}")
    print(f"Falcon {res1}")

    print(f"New {t2}")
    print(f"New {res2}")


def profile_call(f: Callable):
    bef = torch.cuda.Event(enable_timing=True)
    aft = torch.cuda.Event(enable_timing=True)
    f()  # warmup
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with profile(activities=activities,
                 profile_memory=True, record_shapes=True) as prof:
        bef.record()
        precompute_freqs_cis.cache_clear()
        y = f()
        aft.record()

    torch.cuda.synchronize()
    prof_results = prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10)
    t = bef.elapsed_time(aft)
    return y, t, prof_results


# def test_rotary_embeddings_regression():
#     seq_length = 104
#     batch_size = 32
#
#     conf = DummyConf()
#
#     nb = batch_size * conf.n_head
#     q = torch.randn(nb, seq_length, conf.head_dim)  # batch_size * self.num_heads, q_length, self.head_dim)
#     k = torch.randn(nb, seq_length, conf.head_dim)
#     hidden_size = conf.hidden_size
#     num_attention_heads = conf.n_head
#
#     freq_cis = megatron.model.positional_embeddings.precompute_freqs_cis(
#         # self.params.dim // self.params.n_heads, self.params.max_seq_len * 2 # NOTE: LLaMA version
#         hidden_size // num_attention_heads, seq_length * 2
#     )
#     query_layer, key_layer = megatron.model.positional_embeddings.apply_rotary_emb(q, k, freq_cis)


if __name__ == "__main__":
    # test_rotary_embeddings_regression()
    benchmark_rotary_embeddings()

    # benchmark_rotary_attention()
