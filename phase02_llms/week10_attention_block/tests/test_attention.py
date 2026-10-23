"""Hand-written fixture checks for Session 2 (`attention.py`).

These fixtures are authored in `session2.md` and need no corpus and no network.
They are the executable form of the attention contract. Run them with:

    .venv/bin/python phase02_llms/week10_attention_block/tests/test_attention.py

The checks are currently **red by design**: the bodies in `attention.py` are
unimplemented stubs. PyTorch is required; pytest is not.

Note on sizes: these local fixtures are deliberately smaller than the
smoke-run constants at the top of `attention.py` (`SEQ_LEN = 8` there). `T = 4`
and `H = 2` keep the printed mask grids readable. The functions under test must
take their shapes from their arguments, so they work at either size.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

WEEK10_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK10_DIR))

import attention  # noqa: E402  (path insertion must happen first)

B = 2
H = 2
T = 4
D_HEAD = 2


def close(a: object, b: object, tol: float = 1e-5) -> bool:
    """Compare two values as float32 tensors."""
    return torch.allclose(
        torch.as_tensor(a, dtype=torch.float32),
        torch.as_tensor(b, dtype=torch.float32),
        atol=tol,
        rtol=0.0,
    )


def rand_qkv(b: int = B, h: int = H, t: int = T, d: int = D_HEAD, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    shape = (b, h, t, d)
    return (
        torch.randn(shape, generator=generator),
        torch.randn(shape, generator=generator),
        torch.randn(shape, generator=generator),
    )


# --- softmax -------------------------------------------------------------


def test_stable_softmax_rows_sum_to_one() -> None:
    x = torch.randn(3, 5, generator=torch.Generator().manual_seed(1))
    out = attention.stable_softmax(x, dim=-1)
    assert out.shape == x.shape
    assert close(out.sum(dim=-1), torch.ones(3)), "rows must sum to 1"
    assert (out >= 0).all(), "probabilities cannot be negative"


def test_stable_softmax_survives_large_logits() -> None:
    x = torch.tensor([[0.0, 1000.0, -1000.0]])
    out = attention.stable_softmax(x, dim=-1)
    assert torch.isfinite(out).all(), "max-subtraction must prevent overflow"
    assert close(out, [[0.0, 1.0, 0.0]], tol=1e-6)


def test_stable_softmax_matches_naive_reference() -> None:
    x = torch.tensor([[0.4, -1.2, 2.1], [1.0, 1.0, 1.0]])
    reference = torch.exp(x - x.max(dim=-1, keepdim=True).values)
    reference = reference / reference.sum(dim=-1, keepdim=True)
    assert close(attention.stable_softmax(x, dim=-1), reference)


# --- the two masks -------------------------------------------------------


def test_causal_mask_shape_and_orientation() -> None:
    mask = attention.build_causal_mask(3)
    assert mask.shape == (1, 1, 3, 3), f"expected (1, 1, 3, 3), got {tuple(mask.shape)}"
    flat = mask.reshape(3, 3)
    for t in range(3):
        for s in range(3):
            if s <= t:
                assert flat[t, s] == 0.0, f"({t}, {s}) must be attendable"
            else:
                assert flat[t, s] == float("-inf"), f"({t}, {s}) must be blocked"


def test_padding_mask_shape_and_orientation() -> None:
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    mask = attention.padding_to_mask(attention_mask)
    assert mask.shape == (2, 1, 1, 4), f"expected (2, 1, 1, 4), got {tuple(mask.shape)}"
    flat = mask.reshape(2, 4)
    assert (flat[0, :3] == 0.0).all(), "real positions must be 0.0"
    assert flat[0, 3] == float("-inf"), "the padded position must be -inf"
    assert (flat[1, :2] == 0.0).all(), "real positions must be 0.0"
    assert (flat[1, 2:] == float("-inf")).all(), "padded positions must be -inf"


# --- attention -----------------------------------------------------------


def test_attention_shapes() -> None:
    q, k, v = rand_qkv()
    out, weights = attention.scaled_dot_product_attention(q, k, v)
    assert out.shape == (B, H, T, D_HEAD), f"out shape {tuple(out.shape)}"
    assert weights.shape == (B, H, T, T), f"weights shape {tuple(weights.shape)}"


def test_attention_rows_are_a_distribution() -> None:
    q, k, v = rand_qkv()
    _, weights = attention.scaled_dot_product_attention(q, k, v)
    assert (weights >= 0).all()
    assert close(weights.sum(dim=-1), torch.ones(B, H, T))


def test_attention_blocked_positions_get_exactly_zero_weight() -> None:
    q, k, v = rand_qkv()
    causal = attention.build_causal_mask(T)
    _, weights = attention.scaled_dot_product_attention(q, k, v, mask=causal)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    future = future.expand(B, H, T, T)
    assert (weights[future] == 0.0).all(), "future positions must get exactly 0"


def test_hand_computed_uniform_example() -> None:
    """q = k = 0 gives uniform weights, so the output is the mean of the values."""
    q = torch.zeros(1, 1, 2, 1)
    k = torch.zeros(1, 1, 2, 1)
    v = torch.tensor([[[[2.0], [4.0]]]])
    out, weights = attention.scaled_dot_product_attention(q, k, v)
    assert close(weights, [[[[0.5, 0.5], [0.5, 0.5]]]])
    assert close(out, [[[[3.0], [3.0]]]]), "uniform attention averages the values"


def test_hand_computed_causal_example() -> None:
    """With the causal mask, position 0 sees only itself; position 1 sees both."""
    q = torch.zeros(1, 1, 2, 1)
    k = torch.zeros(1, 1, 2, 1)
    v = torch.tensor([[[[2.0], [4.0]]]])
    causal = attention.build_causal_mask(2)
    out, weights = attention.scaled_dot_product_attention(q, k, v, mask=causal)
    assert close(weights, [[[[1.0, 0.0], [0.5, 0.5]]]])
    assert close(out, [[[[2.0], [3.0]]]])


def test_future_token_perturbation_leaves_earlier_positions_unchanged() -> None:
    """The direct causality test: editing a later value cannot move an earlier output."""
    q, k, v = rand_qkv(t=T, seed=7)
    causal = attention.build_causal_mask(T)
    baseline, _ = attention.scaled_dot_product_attention(q, k, v, mask=causal)

    perturbed_v = v.clone()
    perturbed_v[:, :, -1, :] += 100.0
    after, _ = attention.scaled_dot_product_attention(q, k, perturbed_v, mask=causal)

    assert close(after[:, :, : T - 1, :], baseline[:, :, : T - 1, :]), (
        "positions 0..T-2 must not see the last token's value"
    )
    assert not close(after[:, :, -1, :], baseline[:, :, -1, :]), (
        "the perturbed position itself must change, or the test is vacuous"
    )


def test_masks_must_be_summed_not_substituted() -> None:
    """Dropping the padding mask re-opens padded positions to attention."""
    q, k, v = rand_qkv()
    causal = attention.build_causal_mask(T)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    padding = attention.padding_to_mask(attention_mask)
    combined = causal + padding

    _, with_both = attention.scaled_dot_product_attention(q, k, v, mask=combined)
    _, causal_only = attention.scaled_dot_product_attention(q, k, v, mask=causal)

    pad_cols = (attention_mask == 0).view(B, 1, 1, T).expand(B, H, T, T)
    assert (with_both[pad_cols] == 0.0).all(), (
        "padding must receive exactly 0 weight when both masks are applied"
    )
    assert (causal_only[pad_cols] > 0.0).any(), (
        "with the padding mask dropped, row 0 should wrongly attend to the pad token"
    )


def test_fully_masked_row_does_not_produce_nan() -> None:
    """A row whose every entry is blocked must still yield finite numbers."""
    q, k, v = rand_qkv()
    causal = attention.build_causal_mask(T)
    # Row 0 is padding, so its own row is entirely blocked: causal allows only
    # s <= 0, and the padding mask kills column 0 as well.
    attention_mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
    combined = causal + attention.padding_to_mask(attention_mask)
    out, weights = attention.scaled_dot_product_attention(q, k, v, mask=combined)
    assert torch.isfinite(out).all(), "a fully masked row must not produce NaN"
    assert torch.isfinite(weights).all(), "weights must stay finite"


def test_scaling_prevents_saturation_at_large_d_head() -> None:
    """At d_head = 64, dividing by sqrt(d_head) must not collapse to a hard max."""
    q, k, v = rand_qkv(b=1, h=1, t=8, d=64, seed=3)
    _, weights = attention.scaled_dot_product_attention(q, k, v)
    peak = weights.max().item()
    assert peak < 0.999, (
        f"largest weight {peak:.3f} suggests the head width was not rescaled; "
        "dividing by d_head instead of sqrt(d_head) collapses the softmax"
    )


def test_scale_comes_from_the_tensor_not_a_constant() -> None:
    """The divisor must be sqrt(q.shape[-1]), not a module-level constant.

    The saturation check above is not tight enough on its own: dividing by
    sqrt(D_HEAD) - the module constant, 8 - instead of sqrt(16) still keeps the
    peak under its threshold and passes. This check pins the actual scaling by
    comparing against an independent reference built from the tensor's width.
    """
    q, k, v = rand_qkv(b=1, h=1, t=6, d=16, seed=11)
    _, weights = attention.scaled_dot_product_attention(q, k, v)

    d_head = q.shape[-1]
    reference = torch.softmax(q @ k.transpose(-2, -1) / d_head**0.5, dim=-1)

    assert close(weights, reference, tol=1e-6), (
        "weights must equal softmax(q @ kᵀ / sqrt(d_head)) with d_head read "
        f"from the tensor ({d_head}); a module constant gives a different scale"
    )


FIXTURE_CHECKS = [
    # Ordered to match the recommended build order, so the failure you see
    # always points at the next function to write.
    test_causal_mask_shape_and_orientation,
    test_stable_softmax_rows_sum_to_one,
    test_stable_softmax_survives_large_logits,
    test_stable_softmax_matches_naive_reference,
    test_padding_mask_shape_and_orientation,
    test_attention_shapes,
    test_attention_rows_are_a_distribution,
    test_attention_blocked_positions_get_exactly_zero_weight,
    test_hand_computed_uniform_example,
    test_hand_computed_causal_example,
    test_future_token_perturbation_leaves_earlier_positions_unchanged,
    test_masks_must_be_summed_not_substituted,
    test_fully_masked_row_does_not_produce_nan,
    test_scaling_prevents_saturation_at_large_d_head,
    test_scale_comes_from_the_tensor_not_a_constant,
]


def main() -> None:
    passed = 0
    for check in FIXTURE_CHECKS:
        check()
        passed += 1
        print(f"PASS {check.__name__}")
    print(f"\n{passed}/{len(FIXTURE_CHECKS)} fixture checks passed")


if __name__ == "__main__":
    main()
