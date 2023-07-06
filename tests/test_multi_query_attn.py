import functools
import math
import warnings
from typing import Optional, Tuple, Union


from torch.profiler import profile, record_function, ProfilerActivity
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
torch.manual_seed(111)

from einops import rearrange


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

        self.maybe_rotary = RotaryEmbedding(config.head_dim) if config.rotary else lambda q, k: (q, k)

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
            # rearrange(
            #     x,
            #     "batch seq_len group num_heads head_dim ->\
            #     batch seq_len (group num_heads) head_dim",
            #     head_dim=self.head_dim,
            # )
            rearrange(
                x,
                "batch seq_len group num_heads head_dim ->\
                batch seq_len (group num_heads) head_dim"
            )
            for x in [q, k, v]
        ]
        return q, k, v

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge heads together over the last dimenstion
        Args:
            x: (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]
        Returns:
            torch.tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # What we want to achieve is:
        # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads

        # First view to decompose the batch size
        # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)

        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        x = x.permute(0, 2, 1, 3)

        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
        # print("fused_qkv", fused_qkv.shape)

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)
        # print("qkv", query_layer.shape, key_layer.shape, value_layer.shape)

        batch_size, q_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        key_layer = key_layer.transpose(1, 2).reshape(
            batch_size * self.num_heads,
            q_length,
            self.head_dim,
        )
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        query_layer, key_layer = self.maybe_rotary(query_layer, key_layer)

        _, kv_length, _ = key_layer.shape

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
    assert 0 <= 1 < ndim
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


class RotaryEmbeddingMatoba(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        seq_length: int,
        num_head: int,
        theta=10000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.seq_length = seq_length
        self.num_head = num_head
        self.freqs_cis = precompute_freqs_cis(head_dim, seq_length, theta)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        k3 = rearrange(k, "(batch num_heads) seq_len (ri half_head_dim) -> seq_len batch num_heads half_head_dim ri",
                               num_heads=self.num_head, ri=2)
        q3 = rearrange(q, "(batch num_heads) seq_len (ri half_head_dim) -> seq_len batch num_heads half_head_dim ri",
                               num_heads=self.num_head, ri=2)
        # xq_ = torch.view_as_complex(q3.contiguous())
        # xk_ = torch.view_as_complex(k3.contiguous())
        xq_ = torch.view_as_complex(q3.contiguous())
        xk_ = torch.view_as_complex(k3.contiguous())

        # freqs_cis = reshape_for_broadcast(self.freqs_cis, xq_).to(q.device)

        # xq_o1 = torch.view_as_real(xq_ * freqs_cis)
        # xk_o1 = torch.view_as_real(xk_ * freqs_cis)
        xq_out_new = torch.view_as_real(torch.einsum("ijkl,il->ijkl", xq_, self.freqs_cis))
        xk_out_new = torch.view_as_real(torch.einsum("ijkl,il->ijkl", xk_, self.freqs_cis))
        # torch.testing.assert_close(xq_o1, xq_out_new)

        xq_out = xq_out_new.transpose(-1, -2).flatten(3).type_as(q)
        xk_out = xk_out_new.transpose(-1, -2).flatten(3).type_as(k)
        if False:
            # xq_out_new = torch.einsum("ijkl,ab->kijl", q1, freqs_cis)
            freqs_cis_r = torch.view_as_real(self.freqs_cis)
            # xq_out_new = torch.einsum("ijklm,il->ijklm", q3, self.freqs_cis).transpose(-1, -2).flatten(3)
            # xq_out_new = torch.einsum("ijklm,ilm->ijklm", q3, freqs_cis_r).transpose(-1, -2).flatten(3)
            xq_out_new = torch.einsum("ijkl,il->ijklm", xq_, self.freqs_cis).transpose(-1, -2).flatten(3)
            torch.testing.assert_close(xq_out, xq_out_new)
        return xq_out, xk_out


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base=10000,
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # print(inv_freq)
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
    n_head = 128
    # head_dim = 64
    head_dim = 128

    hidden_size = head_dim * n_head
    split_size = 9
    hidden_dropout = 0.0
    attention_dropout = 0.0
    rotary = True
    n_head_kv = 8
    bias = False


def method1(batch_size: int,
            q: torch.Tensor,
            k: torch.Tensor):
    nb, seq_length, head_dim = q.shape
    num_head = nb // batch_size
    falcon_rotary = RotaryEmbedding(conf.head_dim)

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
    # q = rearrange(q.view(batch_size, num_head, seq_length, head_dim), pattern).contiguous()
    # k = rearrange(k.view(batch_size, num_head, seq_length, head_dim), pattern).contiguous()
    q = rearrange(q.view(batch_size, num_head, seq_length, head_dim), pattern)
    k = rearrange(k.view(batch_size, num_head, seq_length, head_dim), pattern)
    # ---------- current ----------
    q_out, k_out = apply_rotary_emb(q, k, freqs_cis)
    return q_out, k_out


def method3(batch_size: int,
            q: torch.Tensor,
            k: torch.Tensor):
    nb, seq_length, head_dim = q.shape
    num_head = nb // batch_size
    matoba_rotary = RotaryEmbeddingMatoba(head_dim, seq_length, num_head)
    q_out, k_out = matoba_rotary(q, k)
    return q_out, k_out


if __name__ == "__main__":
    seq_length = 199
    batch_size = 32

    conf = DummyConf()
    attention_mask = torch.ones((batch_size, seq_length), device='cpu')

    causal_mask = prepare_attn_mask(
        attention_mask,
        input_shape=(batch_size, seq_length),
        past_key_values_length=0,
    )
    attn = Attention(conf)

    x = torch.randn((batch_size, seq_length, conf.hidden_size))
    y = attn(x, causal_mask)

    falcon_rotary = RotaryEmbedding(conf.head_dim)
    matoba_rotary = RotaryEmbeddingMatoba(conf.head_dim, seq_length, conf.n_head)

    # ---------- falcon ----------
    nb = batch_size * conf.n_head
    q = torch.randn(nb, seq_length, conf.head_dim)  # batch_size * self.num_heads, q_length, self.head_dim)
    k = torch.randn(nb, seq_length, conf.head_dim)

    # methods = [method1, method2, method3]
    methods = [method1, method3, method2]
    # methods = list(reversed([method1, method2, method3]))
    num_methods = len(methods)

    qs, ks = [None] * num_methods, [None] * num_methods
    for idx, method in enumerate(methods):

        bef = torch.cuda.Event(enable_timing=True)
        aft = torch.cuda.Event(enable_timing=True)

        q_, k_ = method(batch_size, q, k)

        # activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with profile(activities=activities,
                     profile_memory=True, record_shapes=True) as prof:
            bef.record()
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
    #
    # q1, k1 = method1(batch_size, q, k)
    # q2, k2 = method2(batch_size, q, k)
    # q3, k3 = method3(batch_size, q, k)
    #
    # #
    # # ---------- new ----------
    #
    # torch.testing.assert_close(k1, k2, rtol=0.001, atol=0.0001)
    # torch.testing.assert_close(q1, q2, rtol=0.001, atol=0.0001)
    #
    # torch.testing.assert_close(k1, k3, rtol=0.001, atol=0.0001)
    # torch.testing.assert_close(q1, q3, rtol=0.001, atol=0.0001)
    #
