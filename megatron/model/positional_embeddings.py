
# Extracted from: https://github.com/facebookresearch/llama

import torch
from typing import Optional, Tuple


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    inv_freq = 1.0 / (theta**(torch.arange(0, dim, 2).float()/dim))
    t = torch.arange(end)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return torch.stack([cos, sin], dim=0)


def rotate_half(x):
    x1 = x[..., :x.size(-1)//2]
    x2 = x[..., x.size(-1)//2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,  # [seq_len, batch, heads, dim]
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    freqs_cis = freqs_cis.to(xq.device)
    cos, sin = freqs_cis  # [seq_len, dim] both
    cos = cos[:, None, None, :]
    sin = sin[:, None, None, :]
    xq = (xq*cos) + (rotate_half(xq)*sin)
    xk = (xk*cos) + (rotate_half(xk)*sin)
    return xq, xk
