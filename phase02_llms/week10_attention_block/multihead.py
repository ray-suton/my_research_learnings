"""Week 10 Session 3: multi-head attention.

LEARNER SCAFFOLD. The signatures and docstrings below define the contract
recorded in `session3.md`; the bodies are yours to implement. Week 10 Session 2's
`attention.py` stays untouched - this module wraps it.

`torch.nn.Linear` is allowed: it is a matmul with parameters and hides nothing.
`torch.nn.MultiheadAttention` is not - it is the thing being written.

Run from the repository root once implemented:

    .venv/bin/python phase02_llms/week10_attention_block/multihead.py
"""

from __future__ import annotations

import torch

import torch.nn as nn

from attention import build_causal_mask, scaled_dot_product_attention

# Smoke-run sizes, from EXPERIMENT_PLAN.md "Working Defaults To Freeze".
#
# Smoke-run only, exactly as in attention.py: the layer takes `d_model` and
# `n_heads` as constructor arguments and must never read these.
BATCH_SIZE = 2
SEQ_LEN = 8
D_MODEL = 32
N_HEADS = 4


class MultiHeadAttention(torch.nn.Module):
    """Self-attention over `(B, T, d_model)`, returning `(B, T, d_model)`.

    `__init__(d_model, n_heads)` must:

    - validate `d_model % n_heads == 0` and raise a clear error otherwise;
    - store `self.n_heads` and `self.d_head = d_model // n_heads`;
    - create exactly four projections, with these names, so the fixture spec can
      build an independent per-head reference from the same weights:

          self.w_q, self.w_k, self.w_v, self.w_out
              each `torch.nn.Linear(d_model, d_model)`

    `forward(x, mask=None)` does the five steps from `session3.md`:

    1. project `x` three ways -> `q`, `k`, `v`, each `(B, T, d_model)`
    2. split the last axis into heads -> `(B, T, H, d_head)`
    3. transpose to `(B, H, T, d_head)` and call
       `scaled_dot_product_attention(q, k, v, mask)`
    4. transpose back, merge to `(B, T, d_model)`
    5. apply `self.w_out`

    Pass `mask` through unchanged: this layer never builds a mask.

    Shapes:
        x     `(B, T, d_model)`
        mask  additive, broadcastable to `(B, H, T, T)`, or None
        out   `(B, T, d_model)`
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("Must be divided")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        B, T, d_model = q.shape
        q = torch.reshape(q, (B, T, self.n_heads, self.d_head)).transpose(-3, -2)
        k = torch.reshape(k, (B, T, self.n_heads, self.d_head)).transpose(-3, -2)
        v = torch.reshape(v, (B, T, self.n_heads, self.d_head)).transpose(-3, -2)
        
        output, _ = scaled_dot_product_attention(q, k, v, mask)
        output = torch.reshape(output.transpose(-3,-2), (B,T, d_model))
        return self.w_out(output)




    
        


def main() -> None:
    """Smoke-run once implemented.

    Build the layer at the smoke sizes, run it on a random `(B, T, d_model)`
    input with a causal mask, and print the output shape and the total parameter
    count. Use a fixed seed so the numbers reproduce.
    """
    torch.manual_seed(0)
    layer = MultiHeadAttention(D_MODEL, N_HEADS)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    mask = build_causal_mask(SEQ_LEN)

    out = layer(x, mask=mask)
    parameters = sum(p.numel() for p in layer.parameters())

    print(f"x           {tuple(x.shape)}")
    print(f"mask        {tuple(mask.shape)}")
    print(f"out         {tuple(out.shape)}")
    print(f"H, d_head   {layer.n_heads}, {layer.d_head}")
    print(f"parameters  {parameters}")

if __name__ == "__main__":
    main()
