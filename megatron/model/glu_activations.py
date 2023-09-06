# Adapted from: https://github.com/bigscience-workshop/Megatron-DeepSpeed

import torch
from torch import nn
from torch.nn import functional as F


class _GLUBaseModule(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return x1 * self.activation_fn(x2)


class LiGLU(_GLUBaseModule):
    def __init__(self):
        super().__init__(nn.Identity())


class GEGLU(_GLUBaseModule):
    def __init__(self):
        super().__init__(F.gelu)


class ReGLU(_GLUBaseModule):
    def __init__(self):
        super().__init__(F.relu)


class SwiGLU(_GLUBaseModule):
    def __init__(self):
        super().__init__(F.silu)


liglu = LiGLU()
geglu = GEGLU()
reglu = ReGLU()
swiglu = SwiGLU()


GLU_ACTIVATIONS = {
    "geglu": geglu,
    "liglu": liglu,
    "reglu": reglu,
    "swiglu": swiglu,
}
