"""Hand-written fixture checks for Session 4 (`block.py`).

Run with:

    .venv/bin/python phase02_llms/week10_attention_block/tests/test_block.py

The centrepiece is `test_block_matches_pre_norm_composition`, which rebuilds the
block's forward from its own submodules in the frozen order. It reuses the parts
but encodes the *composition* independently, so a post-norm ordering or a
missing residual add fails it.

PyTorch is required; pytest is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

WEEK10_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK10_DIR))

import attention  # noqa: E402
import block as block_module  # noqa: E402

B = 2
T = 6
D_MODEL = 8
N_HEADS = 2
D_FF = 16
SEED = 0


def close(a: object, b: object, tol: float = 1e-5) -> bool:
    return torch.allclose(
        torch.as_tensor(a, dtype=torch.float32),
        torch.as_tensor(b, dtype=torch.float32),
        atol=tol,
        rtol=0.0,
    )


def make_block(seed: int = SEED):
    torch.manual_seed(seed)
    return block_module.TransformerBlock(D_MODEL, N_HEADS, D_FF)


def random_input(seed: int = 1, b: int = B, t: int = T) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(b, t, D_MODEL, generator=generator)


def make_causal_mask(t: int = T) -> torch.Tensor:
    return attention.build_causal_mask(t)


# --- layer norm -----------------------------------------------------------


def test_layer_norm_shape_is_preserved() -> None:
    torch.manual_seed(SEED)
    ln = block_module.LayerNorm(D_MODEL)
    x = random_input()
    out = ln(x)
    assert out.shape == x.shape, f"expected {tuple(x.shape)}, got {tuple(out.shape)}"


def test_layer_norm_normalises_over_features() -> None:
    """With gamma=1 and beta=0, each position should come out ~N(0, 1)."""
    torch.manual_seed(SEED)
    ln = block_module.LayerNorm(D_MODEL)
    out = ln(random_input())

    means = out.mean(dim=-1)
    stds = out.std(dim=-1, unbiased=False)
    assert close(means, torch.zeros_like(means), tol=1e-4), (
        f"per-position means should be ~0, got max |mean| {means.abs().max().item():.2e}"
    )
    assert close(stds, torch.ones_like(stds), tol=1e-3), (
        f"per-position stds should be ~1, got range "
        f"{stds.min().item():.4f}..{stds.max().item():.4f}"
    )


def test_layer_norm_is_stable_when_variance_is_tiny() -> None:
    """eps must prevent a divide-by-zero on a near-constant position."""
    torch.manual_seed(SEED)
    ln = block_module.LayerNorm(D_MODEL)
    x = torch.full((1, 2, D_MODEL), 3.0)  # zero variance everywhere
    out = ln(x)
    assert torch.isfinite(out).all(), "a constant input must not produce inf or nan"


def test_layer_norm_parameters_are_learnable() -> None:
    torch.manual_seed(SEED)
    ln = block_module.LayerNorm(D_MODEL)
    assert ln.gamma.shape == (D_MODEL,), f"gamma shape {tuple(ln.gamma.shape)}"
    assert ln.beta.shape == (D_MODEL,), f"beta shape {tuple(ln.beta.shape)}"
    assert ln.gamma.requires_grad and ln.beta.requires_grad, (
        "gamma and beta must be nn.Parameter, or they will never train"
    )


# --- feed forward ---------------------------------------------------------


def test_feed_forward_shape_and_parameter_count() -> None:
    torch.manual_seed(SEED)
    ff = block_module.FeedForward(D_MODEL, D_FF)
    x = random_input()
    out = ff(x)
    assert out.shape == x.shape, f"expected {tuple(x.shape)}, got {tuple(out.shape)}"

    expected = (D_MODEL * D_FF + D_FF) + (D_FF * D_MODEL + D_MODEL)
    actual = sum(p.numel() for p in ff.parameters())
    assert actual == expected, f"expected {expected} parameters, got {actual}"


def test_feed_forward_is_not_affine() -> None:
    """A position-wise MLP must apply a nonlinearity, not just two linears."""
    torch.manual_seed(SEED)
    ff = block_module.FeedForward(D_MODEL, D_FF)
    x = random_input(seed=5)
    zero = torch.zeros(1, 1, D_MODEL)

    f0 = ff(zero)
    lhs = ff(2.0 * x) - f0
    rhs = 2.0 * (ff(x) - f0)

    assert not close(lhs, rhs, tol=1e-4), (
        "the MLP behaves affinely, so no nonlinearity is applied"
    )


# --- the assembled block --------------------------------------------------


def test_block_shape_and_dtype() -> None:
    block = make_block()
    x = random_input()
    out = block(x, mask=make_causal_mask())
    assert out.shape == x.shape, f"expected {tuple(x.shape)}, got {tuple(out.shape)}"
    assert out.dtype == x.dtype


def test_block_matches_pre_norm_composition() -> None:
    """Rebuild the forward from the block's own parts, in the frozen order."""
    block = make_block()
    x = random_input()
    mask = make_causal_mask()

    reference = x + block.attn(block.ln1(x), mask)
    reference = reference + block.mlp(block.ln2(reference))

    assert close(block(x, mask=mask), reference), (
        "the block must be x = x + attn(ln1(x)); x = x + mlp(ln2(x)). A mismatch "
        "means the norm is on the wrong side of the residual, or a residual add "
        "is missing"
    )


def test_block_changes_its_input_rather_than_replacing_it() -> None:
    """The residual path must survive: the output is not just the sublayers."""
    block = make_block()
    x = random_input()

    out = block(x, mask=make_causal_mask())
    sublayers_only = block.attn(block.ln1(x), make_causal_mask())

    assert not close(out, sublayers_only), (
        "output equals the attention branch alone, so the residual add is missing"
    )


def test_block_gradients_reach_every_parameter() -> None:
    block = make_block()
    x = random_input()
    out = block(x, mask=make_causal_mask())
    out.sum().backward()

    missing = [name for name, p in block.named_parameters() if p.grad is None]
    assert not missing, f"these parameters received no gradient: {missing}"


def test_block_causal_mask_passes_through() -> None:
    block = make_block()
    x = random_input()
    mask = make_causal_mask()
    baseline = block(x, mask=mask)

    perturbed = x.clone()
    perturbed[:, -1, :] += 100.0
    after = block(perturbed, mask=mask)

    assert close(after[:, 0, :], baseline[:, 0, :]), (
        "position 0 must not see the last token; the mask is not reaching "
        "attention through the block"
    )


FIXTURE_CHECKS = [
    test_layer_norm_shape_is_preserved,
    test_layer_norm_normalises_over_features,
    test_layer_norm_is_stable_when_variance_is_tiny,
    test_layer_norm_parameters_are_learnable,
    test_feed_forward_shape_and_parameter_count,
    test_feed_forward_is_not_affine,
    test_block_shape_and_dtype,
    test_block_matches_pre_norm_composition,
    test_block_changes_its_input_rather_than_replacing_it,
    test_block_gradients_reach_every_parameter,
    test_block_causal_mask_passes_through,
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
