import time
import functools
import math
import argparse
import warnings
import logging
from typing import Callable, Optional, Tuple, Union


from torch.profiler import profile, record_function, ProfilerActivity
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from einops import rearrange
#
# import megatron.initialize

logging_level = logging.INFO
# Logging commands
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)

standard_streamhandler = logging.StreamHandler()

logger.addHandler(standard_streamhandler)
# End Logging

torch.manual_seed(111)


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
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_).to(q.device)

    xq_out = torch.view_as_real(xq_ * freqs_cis).transpose(-1, -2).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).transpose(-1, -2).flatten(3)
    return xq_out.type_as(q), xk_out.type_as(k)


@functools.lru_cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def _do_it(z: torch.Tensor,
           num_head: int,
           seq_length,
           head_dim,
           freqs_cis: torch.Tensor) -> torch.Tensor:
    dtype = z.dtype
    z = rearrange(z,
                   "(batch num_heads) seq_len (ri half_head_dim) -> seq_len (batch num_heads half_head_dim) ri",
                   num_heads=num_head, ri=2)
    z = rearrange(torch.view_as_complex(z),
                    "seq_len (batch num_heads half_head_dim) ->  seq_len batch num_heads half_head_dim",
                    seq_len=seq_length, num_heads=num_head,
                    half_head_dim=head_dim // 2)
    z = torch.view_as_real(
        torch.einsum("ijkl,il->ijkl", z, freqs_cis))
    z = rearrange(z, "i j k l m -> i j k (m l)").to(dtype)
    return z


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
        freqs_cis = precompute_freqs_cis(self.head_dim, seq_length, self.theta).to(device).reshape(seq_length, 1, 1, -1)
        # xq_out = _do_it(q, self.num_head, seq_length, self.head_dim, freqs_cis)
        # xk_out = _do_it(k, self.num_head, seq_length, self.head_dim, freqs_cis)
        pattern1 = "(batch num_heads) seq_len (ri half_head_dim) -> seq_len (batch num_heads half_head_dim) ri"
        pattern2 = "seq_len (batch num_heads half_head_dim) -> seq_len batch num_heads half_head_dim"
        k4 = rearrange(k, pattern1, num_heads=self.num_head, ri=2)
        q4 = rearrange(q, pattern1, num_heads=self.num_head, ri=2)
        xq_ = rearrange(torch.view_as_complex(q4), pattern2,
                      seq_len=seq_length, num_heads=self.num_head, half_head_dim=self.head_dim // 2)
        xk_ = rearrange(torch.view_as_complex(k4), pattern2,
                      seq_len=seq_length, num_heads=self.num_head, half_head_dim=self.head_dim // 2)

        # xq_out_new = torch.view_as_real(torch.einsum("ijkl,il->ijkl", xq_, freqs_cis))
        # xk_out_new = torch.view_as_real(torch.einsum("ijkl,il->ijkl", xk_, freqs_cis))
        #
        # xq_out_new = torch.view_as_real(xq_ * freqs_cis.reshape(seq_length, 1, 1, -1))
        # xk_out_new = torch.view_as_real(xk_ * freqs_cis.reshape(seq_length, 1, 1, -1))
        #
        # xq_out_new = torch.view_as_real(xq_ * freqs_cis)
        # xk_out_new = torch.view_as_real(xk_ * freqs_cis)
        xq_out = rearrange(torch.view_as_real(xq_ * freqs_cis), "i j k l m -> i j k (m l)").type_as(q)
        xk_out = rearrange(torch.view_as_real(xk_ * freqs_cis), "i j k l m -> i j k (m l)").type_as(k)
        return xq_out, xk_out


class DummyConf:
    # n_head = 12
    n_head = 32
    # head_dim = 64
    head_dim = 128
    # head_dim = 256
    # head_dim = 512
    hidden_size = head_dim * n_head
    split_size = 9
    hidden_dropout = 0.0
    attention_dropout = 0.0
    n_head_kv = n_head
    bias = False


def method2(batch_size: int,
            q: torch.Tensor,
            k: torch.Tensor):
    nb, seq_length, head_dim = q.shape
    num_head = nb // batch_size
    freqs_cis = precompute_freqs_cis(head_dim, seq_length).to(device)

    pattern = "batch num_heads seq_len head_dim -> seq_len batch num_heads head_dim"
    q = rearrange(q.view(batch_size, num_head, seq_length, head_dim), pattern)  # .contiguous()
    k = rearrange(k.view(batch_size, num_head, seq_length, head_dim), pattern)  # .contiguous()
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


def benchmark_rotary_embeddings():
    # seq_length = 1024
    # seq_length = 2048
    # seq_length = 4096
    # seq_length = 4096
    # seq_length = 2048
    seq_length = 4096
    # seq_length = 8192
    # seq_length = 128
    # seq_length = 1024
    # batch_size = 8
    # batch_size = 4
    # batch_size = 16
    # batch_size = 32
    batch_size = 24
    conf = DummyConf()

    nb = batch_size * conf.n_head
    q = torch.randn(nb, seq_length, conf.head_dim, device=device)
    k = torch.randn(nb, seq_length, conf.head_dim, device=device)

    # methods = [method2, method3]
    methods = [method2, method3]
    # methods = [method1, method3, method2]
    # methods = [method1, method2, method3]
    # methods = [method1, method3, method2, method4]
    # methods = list(reversed([method1, method2, method3]))
    num_methods = len(methods)
    # [<FunctionEventAvg key=cudaGetDeviceCount self_cpu_time=802.000us cpu_time=401.000us  self_cuda_time=0.000us cuda_time=0.000us input_shapes= cpu_memory_usage=0 cuda_memory_usage=0>, <FunctionEventAvg key=cudaGetDeviceProperties self_cpu_time=59.000us cpu_time=59.000us  self_cuda_time=0.000us cuda_time=0.000us input_shapes= cpu_memory_usage=0 cuda_memory_usage=0>, <FunctionEventAvg key=aten::arange self_cpu_time=41.000us cpu_time=18.250us  self_cuda_time=0.000us cuda_time=0.000us input_shapes= cpu_memory_usage=67584 cuda_memory_usage=0>, <FunctionEventAvg key=aten::empty self_cpu_time=72.000us cpu_time=9.000us  self_cuda_time=0.000us cuda_time=0.000us input_shapes= cpu_memory_usage=1073741824 cuda_memory_usage=0>, <FunctionEventAvg key=aten::resize_ self_cpu_time=5.000us cpu_time=2.500us  self_cuda_time=0.000us cuda_time=0.000us input_shapes= cpu_memory_usage=1024 cuda_memory_usage=0>, <FunctionEventAvg key=aten::slice self_cpu_time=10.000us cpu_time=12.000us  self_cuda_time=0.000us cud...
    # sort_by = "self_cpu_memory_usage"
    sort_by = "cuda_time"
    qs, ks = [None] * num_methods, [None] * num_methods
    for idx, method in enumerate(methods):
        bef = torch.cuda.Event(enable_timing=True)
        aft = torch.cuda.Event(enable_timing=True)

        # activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with profile(activities=activities,
                     profile_memory=True, record_shapes=True) as prof:
            # q_, k_ = method(batch_size, torch.rand_like(q), torch.rand_like(k))

            precompute_freqs_cis.cache_clear()
            bef.record()
            q_, k_ = method(batch_size, q, k)
            aft.record()

        logger.info(prof.key_averages().table(sort_by=sort_by, row_limit=10))

        # Waits for everything to finish running
        torch.cuda.synchronize()
        qs[idx] = q_
        ks[idx] = k_
        logger.info(str(method) + " " + str(bef.elapsed_time(aft)))

    for idx in range(1, num_methods):
        torch.testing.assert_close(ks[0], ks[idx], rtol=0.001, atol=0.0001)
        torch.testing.assert_close(qs[0], qs[idx], rtol=0.001, atol=0.0001)


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


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device_name", type=str, default="cuda")

    parser.add_argument("--mode", type=str, default="", help="Swallow PyCharm args")
    parser.add_argument("--port", type=str, default="", help="Swallow PyCharm args")
    parser.add_argument("-f", type=str, default="", help="Swallow IPython arg")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    # device = torch.device("cuda")
    device = torch.device(args.device_name)
    if device == torch.device("cuda"):
        logger.info(torch.cuda.get_device_properties(device))
    benchmark_rotary_embeddings()
