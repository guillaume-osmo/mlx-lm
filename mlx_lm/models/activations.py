import mlx.nn as nn


def swiglu(gate, up):
    return nn.silu(gate) * up
