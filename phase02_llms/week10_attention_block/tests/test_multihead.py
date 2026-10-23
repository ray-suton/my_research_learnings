"""Hand-written fixture checks for Session 3 (`multihead.py`).

These fixtures are authored in `session3.md` and need no corpus and no network.
Run them with:

    .venv/bin/python phase02_llms/week10_attention_block/tests/test_multihead.py

The centrepiece is `test_matches_manual_per_head_reference`, which builds the
per-head computation itself with explicit slicing. It deliberately does NOT
reuse the implementation's split/merge machinery - a reference that shares the
code under test cannot catch a bug in it.

PyTorch is required; pytest is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

WEEK10_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK10_DIR))

import attention  # noqa: E402
import multihead  # noqa: E402

B = 2
T = 6
D_MODEL = 8  # divisible by both 1 and 2, which the head-count checks use
H = 2
SEED = 0


def close(a: object, b: object, tol: float = 1e-5) -> bool:
    return torch.allclose(
        torch.as_tensor(a, dtype=torch.float32),
        torch.as_tensor(b, dtype=torch.float32),
        atol=tol,
        rtol=0.0,
    )


def make_layer(d_model: int = D_MODEL, n_heads: int = H, seed: int = SEED):
    torch.manual_seed(seed)
    return multihead.MultiHeadAttention(d_model, n_heads)


def random_input(seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, D_MODEL, generator=generator)


def manual_per_head_reference(layer, x: torch.Tensor, mask=None) -> torch.Tensor:
    """Per-head attention with explicit slicing, using the layer's own weights.

    The head split here is written by hand - `q[..., i*d_head:(i+1)*d_head]` -
    rather than by calling any helper from the implementation.
    """
    d_head = layer.d_head
    q = layer.w_q(x)
    k = layer.w_k(x)
    v = layer.w_v(x)

    heads = []
    for i in range(layer.n_heads):
        sl = slice(i * d_head, (i + 1) * d_head)
        qi = q[..., sl][:, None, :, :]  # (B, 1, T, d_head)
        ki = k[..., sl][:, None, :, :]
        vi = v[..., sl][:, None, :, :]
        out_i, _ = attention.scaled_dot_product_attention(qi, ki, vi, mask)
        heads.append(out_i[:, 0, :, :])

    merged = torch.cat(heads, dim=-1)  # (B, T, d_model)
    return layer.w_out(merged)


# --- shapes and configuration -------------------------------------------


def test_output_shape_and_dtype() -> None:
    layer = make_layer()
    x = random_input()
    out = layer(x)
    assert out.shape == (B, T, D_MODEL), f"expected {(B, T, D_MODEL)}, got {tuple(out.shape)}"
    assert out.dtype == x.dtype, "the layer must preserve dtype"


def test_head_count_must_divide_d_model() -> None:
    torch.manual_seed(SEED)
    try:
        multihead.MultiHeadAttention(8, 3)
    except Exception as exc:  # noqa: BLE001 - any clear error is acceptable
        assert not isinstance(exc, NotImplementedError), (
            "constructing a valid-looking layer with a bad head count must raise "
            "a real error, not NotImplementedError"
        )
    else:
        raise AssertionError("d_model=8 with n_heads=3 must raise an error")


def test_single_head_is_a_valid_configuration() -> None:
    layer = make_layer(n_heads=1)
    out = layer(random_input())
    assert out.shape == (B, T, D_MODEL)


# --- the centrepiece: equivalence with an independent reference ----------


def test_matches_manual_per_head_reference() -> None:
    layer = make_layer()
    x = random_input()
    reference = manual_per_head_reference(layer, x)
    assert close(layer(x), reference), (
        "multi-head output must match a hand-written per-head loop; a mismatch "
        "means the head split or merge is inconsistent or transposed"
    )


def test_matches_reference_with_causal_and_padding_mask() -> None:
    layer = make_layer()
    x = random_input()
    attention_mask = torch.ones(B, T, dtype=torch.int64)
    attention_mask[:, -2:] = 0
    mask = attention.build_causal_mask(T) + attention.padding_to_mask(attention_mask)
    reference = manual_per_head_reference(layer, x, mask=mask)
    assert close(layer(x, mask=mask), reference), (
        "the mask must be passed through unchanged and broadcast across heads"
    )


# --- mask behaviour through the layer ------------------------------------


def test_causal_mask_blocks_future_information() -> None:
    """Editing the last token must not move the first token's output."""
    layer = make_layer()
    x = random_input()
    mask = attention.build_causal_mask(T)
    baseline = layer(x, mask=mask)

    perturbed = x.clone()
    perturbed[:, -1, :] += 100.0
    after = layer(perturbed, mask=mask)

    assert close(after[:, 0, :], baseline[:, 0, :]), (
        "position 0 must not see the last token; the causal mask is not reaching "
        "the attention computation"
    )
    assert not close(after[:, -1, :], baseline[:, -1, :]), (
        "the perturbed position itself must change, or the test is vacuous"
    )


def test_padded_position_is_not_read() -> None:
    """A padding position's keys and values must never enter another row."""
    layer = make_layer()
    x = random_input()
    attention_mask = torch.ones(B, T, dtype=torch.int64)
    attention_mask[:, -1] = 0
    mask = attention.padding_to_mask(attention_mask)
    baseline = layer(x, mask=mask)

    perturbed = x.clone()
    perturbed[:, -1, :] += 100.0
    after = layer(perturbed, mask=mask)

    assert close(after[:, :-1, :], baseline[:, :-1, :]), (
        "a padded column must be unreadable; other positions' outputs moved"
    )


# --- independence and repeatability --------------------------------------


def test_batch_rows_are_independent() -> None:
    layer = make_layer()
    x = random_input()
    x[1] = x[0]
    out = layer(x)
    assert close(out[0], out[1]), "identical rows must produce identical outputs"


def test_repeated_calls_agree() -> None:
    layer = make_layer()
    x = random_input()
    assert close(layer(x), layer(x)), "the layer must be deterministic at eval"


FIXTURE_CHECKS = [
    test_output_shape_and_dtype,
    test_head_count_must_divide_d_model,
    test_single_head_is_a_valid_configuration,
    test_matches_manual_per_head_reference,
    test_matches_reference_with_causal_and_padding_mask,
    test_causal_mask_blocks_future_information,
    test_padded_position_is_not_read,
    test_batch_rows_are_independent,
    test_repeated_calls_agree,
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
