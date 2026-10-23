"""Week 10 Session 2: scaled dot-product attention and the additive masks.

LEARNER SCAFFOLD. The constants, signatures, and docstrings below define the
contract recorded in `session2.md` and `EXPERIMENT_PLAN.md`; the bodies are
yours to implement. Week 9 Sessions 2 and 4 started the same way. No solution
is provided here.

Hand-written only: do not call `torch.nn.functional.scaled_dot_product_attention`,
`torch.nn.MultiheadAttention`, or `torch.nn.TransformerEncoderLayer`. Reading
their documentation is expected; calling them defeats the exercise.

Run from the repository root once implemented:

    .venv/bin/python phase02_llms/week10_attention_block/attention.py
"""

from __future__ import annotations

import math

import torch

# Smoke-run sizes, from EXPERIMENT_PLAN.md "Working Defaults To Freeze".
#
# These describe the shapes `main()` uses for its printed summary. They are NOT
# the shapes your functions receive: tests/test_attention.py deliberately calls
# the functions with smaller tensors (T = 4, H = 2, d_head = 2) so the printed
# mask grids stay readable. Every function here must take its shapes from its
# arguments - `x.shape`, `attention_mask.shape` - and never from these
# constants, or it will only work for one fixed batch.
BATCH_SIZE = 2
SEQ_LEN = 8
D_MODEL = 32
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS
D_FF = 128

NEG_INF = float("-inf")


def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax along `dim`, subtracting the maximum first.

    Subtract the maximum along `dim` before exponentiating so large logits do
    not overflow. The subtraction is mathematically a no-op and numerically
    essential.

    Return a tensor of the same shape whose entries along `dim` are
    non-negative and sum to 1.
    """

    maxs = torch.amax(x, dim, keepdim = True)

    ex = torch.exp(x-maxs)
    e_sum = ex.sum(dim = dim, keepdim = True)

    return ex/ e_sum 



def build_causal_mask(seq_len: int) -> torch.Tensor:
    """Return an additive causal mask of shape `(1, 1, seq_len, seq_len)`.

    Entry `[..., t, s]` is `0.0` when position `t` may attend to position `s`
    (that is, `s <= t`) and negative infinity when it may not (`s > t`).

    The leading two axes are broadcast axes: the same mask applies to every
    batch row and every head.
    """
    causal_mask = torch.zeros((1,1, seq_len, seq_len))

    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            causal_mask[0, 0, i, j] = -float('inf')
        
    
    return causal_mask 




def padding_to_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Convert a 0/1 padding mask of shape `(B, T)` into an additive mask.

    Input convention: 1 marks a real token, 0 marks padding.

    Return shape `(B, 1, 1, T)` with `0.0` at real positions and negative
    infinity at padding positions, so the softmax sends padding weight to zero.

    Keep this distinct from the causal mask. They answer different questions
    and are combined by addition, never by one overwriting the other.
    """
    additive = torch.where(attention_mask > 0, 0.0, float("-inf"))
    rows,length = attention_mask.shape

    return additive.view(rows, 1,1, length)
    

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend over `k` and `v` from `q`, returning `(out, weights)`.

    Shapes:

        q, k, v  `(B, H, T, d_head)`
        mask     additive, broadcastable to `(B, H, T, T)`, or None
        out      `(B, H, T, d_head)`
        weights  `(B, H, T, T)`

    Compute `weights = stable_softmax(q @ k.transpose(-2, -1) / sqrt(d_head) + mask)`
    and `out = weights @ v`. Divide by the square root of the head width, not
    by the head width.

    A row that is entirely masked must not produce `NaN`: handle it explicitly
    and describe the choice in a comment. A real position must still receive its
    normal weight.
    """


    d_head = q.shape[-1]

    # Divide by sqrt(d_head), measured from the tensor, never by d_head and
    # never by the module constant - the callers pass their own head width.
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_head)

    if mask is not None:
        scores = scores + mask

    weights = stable_softmax(scores, dim=-1)

    if mask is not None:
        # A row whose every entry is -inf is fully blocked: no source position
        # may be read, and the softmax returns NaN for that row. Repair it here,
        # because the caller is the only place that holds the mask. Zeroing the
        # weights makes `out` come out as zeros at those positions and leaves
        # every other row untouched.
        dead_rows = torch.isneginf(mask).all(dim=-1, keepdim=True)
        weights = torch.where(dead_rows, torch.zeros_like(weights), weights)

    out = weights @ v
    return out, weights


def main() -> None:
    """Smoke-run the pieces on the fixture sizes once they are implemented.

    Build random `q`, `k`, `v` of shape `(B, H, T, d_head)`, combine a causal
    mask with a padding mask additively, run attention, and print the output
    shape, the row-sum range, and the count of exactly-zero masked weights. Use
    a fixed seed so the numbers reproduce.
    """
    torch.manual_seed(0)

    shape = (BATCH_SIZE, N_HEADS, SEQ_LEN, D_HEAD)
    q = torch.randn(shape)
    k = torch.randn(shape)
    v = torch.randn(shape)

    # Last two positions of every row are filler, so the padding mask has
    # something to block. Combined additively with the causal mask.
    attention_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.int64)
    attention_mask[:, -2:] = 0
    mask = build_causal_mask(SEQ_LEN) + padding_to_mask(attention_mask)

    out, weights = scaled_dot_product_attention(q, k, v, mask=mask)

    row_sums = weights.sum(dim=-1)
    exact_zeros = (weights == 0.0).sum().item()

    print(f"q, k, v          {tuple(q.shape)}")
    print(f"mask             {tuple(mask.shape)}")
    print(f"out              {tuple(out.shape)}")
    print(f"weights          {tuple(weights.shape)}")
    print(f"row-sum range    {row_sums.min().item():.6f} .. {row_sums.max().item():.6f}")
    print(f"exact-zero weights  {exact_zeros} of {weights.numel()}")


if __name__ == "__main__":
    main()
