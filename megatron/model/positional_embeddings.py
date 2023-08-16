# Extracted from: https://github.com/facebookresearch/llama

from typing import Optional
import torch


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, scaling_factor: float = 1.0
) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device).float() / scaling_factor  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[0], x.shape[-1])
    shape = [d if i == 0 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = freqs_cis.to(xq.device)
    if position_ids is None:
        # we assume position_ids to be torch.arange(seq_len)
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
        # freqs_cis: [seq_len, 1, 1, head_dim//2] (complex64)
    else:
        # use specified position_ids, possibly not monotonically increasing
        # tensor shapes & tpyes:
        # xq_: [seq_len, batch_size, heads, head_dim//2] (complex64)
        # position_ids: [batch_size, seq_len] (long)
        position_ids = position_ids.to(xq.device)   # normally already on correct device
        assert position_ids.shape == (xq_.shape[1], xq_.shape[0])
        assert (freqs_cis.shape[1] == xq_.shape[-1])
        freqs_cis = freqs_cis[position_ids].transpose(0, 1).unsqueeze(-2)
        # freqs_cis: [seq_len, batch_size, 1, head_dim//2] (complex64)

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
