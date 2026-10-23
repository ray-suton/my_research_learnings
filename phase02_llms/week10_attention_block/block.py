"""Week 10 Session 4: layer norm, the position-wise MLP, and the block.

LEARNER SCAFFOLD. The signatures and docstrings below define the contract in
`session4.md`; the bodies are yours to implement. Sessions 2 and 3 are closed -
`attention.py` and `multihead.py` stay untouched, and this module wraps them.

`torch.nn.Linear` and `torch.nn.GELU` are allowed. `torch.nn.LayerNorm` is not:
writing layer norm by hand is part of this session's exercise.

Run from the repository root once implemented:

    .venv/bin/python phase02_llms/week10_attention_block/block.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from attention import build_causal_mask
from multihead import MultiHeadAttention

# Smoke-run sizes, from EXPERIMENT_PLAN.md "Working Defaults To Freeze".
# Smoke-run only, as in the earlier modules: the classes take their sizes as
# constructor arguments and must never read these.
BATCH_SIZE = 2
SEQ_LEN = 8
D_MODEL = 32
N_HEADS = 4
D_FF = 128
EPS = 1e-5


class LayerNorm(nn.Module):
    """Hand-written layer norm over the **last** axis.

    `__init__(d_model, eps=EPS)` must:

    - store `self.eps`;
    - create exactly two learnable parameters, with these names, so the fixture
      spec can reason about them:

          self.gamma   (d_model,)  initialised to ones
          self.beta    (d_model,)  initialised to zeros

      both `nn.Parameter`, so they are registered and trainable.

    `forward(x)` normalises over the **last** axis, per position and per sample:
    subtract that position's own mean, divide by its own standard deviation
    (plus `eps` inside the square root), then apply `gamma` and `beta`.

    Do not use `torch.nn.LayerNorm`. Shapes are preserved: `(..., d_model)` in,
    `(..., d_model)` out.
    """

    def __init__(self, d_model: int, eps: float = EPS) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones((d_model,), requires_grad = True))
        self.beta = nn.Parameter(torch.zeros((d_model,), requires_grad = True))
        self.eps = eps 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mea = torch.mean(x, dim = -1, keepdim = True)
        var = torch.mean((x-mea)**2, dim = -1, keepdim = True)

        normalized = (x - mea) / torch.sqrt(var + self.eps)
        output = self.gamma * normalized + self.beta 

        return output



class FeedForward(nn.Module):
    """Position-wise MLP: `d_model -> d_ff -> d_model`.

    `__init__(d_model, d_ff)` must create, with these names:

        self.fc_in    nn.Linear(d_model, d_ff)
        self.act      nn.GELU()
        self.fc_out   nn.Linear(d_ff, d_model)

    `forward(x)` applies the same small network to every position
    independently - it never looks across positions. Input and output are both
    `(B, T, d_model)`.
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc_in = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc_out = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        x = self.act(x)
        return self.fc_out(x)

class TransformerBlock(nn.Module):
    """Pre-norm block: attention and MLP, each wrapped in a residual add.

    `__init__(d_model, n_heads, d_ff)` must create, with these names:

        self.ln1     LayerNorm(d_model)
        self.attn    MultiHeadAttention(d_model, n_heads)
        self.ln2     LayerNorm(d_model)
        self.mlp     FeedForward(d_model, d_ff)

    `forward(x, mask=None)` is exactly these two lines, in this order:

        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x

    The layer norm sits **inside** the branch and the residual path is an
    untouched identity. The mask is forwarded to attention unchanged; the block
    never builds one. Input and output are both `(B, T, d_model)`.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_ff)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x
        


def main() -> None:
    """Smoke-run once implemented.

    Build the block at the smoke sizes, run it on a random `(B, T, d_model)`
    input with a causal mask, and print the output shape, the total parameter
    count, and how much the output moved away from its input (a residual block
    should change its input, not replace it). Use a fixed seed.
    """
    torch.manual_seed(0)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    mask = build_causal_mask(SEQ_LEN)
    block = TransformerBlock(D_MODEL, N_HEADS, D_FF)
    out = block(x, mask)
    parameter_count = sum(parameter.numel() for parameter in block.parameters())
    mean_absolute_move = (out - x).abs().mean().item()

    print(f"x               {tuple(x.shape)}")
    print(f"out             {tuple(out.shape)}")
    print(f"parameters      {parameter_count}")
    print(f"mean |out - x|  {mean_absolute_move:.6f}")


if __name__ == "__main__":
    main()
