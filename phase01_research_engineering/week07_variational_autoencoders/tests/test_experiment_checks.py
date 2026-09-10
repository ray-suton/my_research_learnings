"""Check epoch metrics and validation on tiny synthetic CPU batches."""

import argparse
import math
from pathlib import Path
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from model import vae
from train import evaluate, train_one_epoch


class MetricModel(torch.nn.Module):
    """Make the expected losses easy to calculate by hand."""

    def __init__(self):
        super().__init__()
        self.shift = torch.nn.Parameter(torch.tensor(0.0))

    def encode(self, X):
        mu = X.flatten(1)[:, :2] + self.shift
        return mu, torch.zeros_like(mu)

    def decoder(self, z):
        return z[:, :1].expand(-1, 4) + self.shift

    def forward(self, X):
        mu, logvar = self.encode(X)
        preds = torch.zeros_like(X) + self.shift
        return preds, mu, logvar


def make_batches():
    # The final batch has one image with much larger reconstruction error.
    X = torch.tensor([0.0, 0.25, 1.0]).reshape(3, 1, 1, 1)
    X = X.expand(-1, 1, 2, 2).clone()
    y = torch.zeros(3, dtype=torch.long)
    return [(X[:2], y[:2]), (X[2:], y[2:])]


def check_expected_metrics(metrics, beta):
    # Every pixel is either 0, 0.25, or 1, and predictions are all zero.
    expected_mse = (0.0 + 0.25**2 + 1.0) / 3
    # Two latent means equal the pixel value; both variances are one.
    expected_kl = expected_mse
    assert metrics["n_examples"] == 3
    expected = {
        "reconstruction_mse": expected_mse,
        "reconstruction_term": 4 * expected_mse,
        "kl": expected_kl,
        "total": 4 * expected_mse + beta * expected_kl,
    }
    for name, value in expected.items():
        assert math.isfinite(metrics[name]), name
        assert math.isclose(metrics[name], value, rel_tol=1e-6), name


def check_weighted_training():
    model = MetricModel()
    model.eval()
    # Freeze parameter updates so both batches have known predictions.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    metrics = train_one_epoch(
        make_batches(), model, torch.nn.MSELoss(), 0.25, optimizer, "cpu"
    )
    assert model.training
    check_expected_metrics(metrics, beta=0.25)


def check_weighted_evaluation():
    model = MetricModel()
    metrics = evaluate(
        make_batches(), model, torch.nn.MSELoss(), 0.25, "cpu", seed=123
    )
    check_expected_metrics(metrics, beta=0.25)
    expected_per_dimension = torch.full((2,), (0.25**2 + 1.0) / 6)
    torch.testing.assert_close(
        torch.as_tensor(metrics["kl_per_dimension"]), expected_per_dimension
    )
    assert math.isclose(
        sum(metrics["kl_per_dimension"]), metrics["kl"], rel_tol=1e-6
    )
    assert 0 <= metrics["active_latent_dimensions"] <= 2
    # Decoding mu exactly reconstructs these constant images. Forward does not.
    assert metrics["mean_latent_mse"] == 0.0
    assert metrics["reconstruction_mse"] > metrics["mean_latent_mse"]


def check_evaluation_preserves_state():
    model = vae(4, 8, 2)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.25)

    observed_modes = []

    def observe(module, inputs):
        observed_modes.append((module.training, torch.is_grad_enabled()))

    model_hook = model.register_forward_pre_hook(observe)
    decoder_hook = model.decoder.register_forward_pre_hook(observe)
    try:
        for training in (True, False):
            model.train(training)
            observed_modes.clear()
            random_state = torch.get_rng_state().clone()
            evaluate(make_batches(), model, torch.nn.MSELoss(), 1.0, "cpu")

            assert model.training == training
            assert torch.equal(torch.get_rng_state(), random_state)
            assert observed_modes
            assert all(mode == (False, False) for mode in observed_modes)
            for name, value in model.state_dict().items():
                assert torch.equal(value, before[name]), name
            for parameter in model.parameters():
                assert torch.equal(parameter.grad, torch.full_like(parameter, 0.25))
    finally:
        model_hook.remove()
        decoder_hook.remove()


def check_repeatable_evaluation():
    torch.manual_seed(42)
    model = vae(4, 8, 2)
    first = evaluate(make_batches(), model, torch.nn.MSELoss(), 1.0, "cpu", seed=123)
    torch.randn(17)
    second = evaluate(make_batches(), model, torch.nn.MSELoss(), 1.0, "cpu", seed=123)
    assert first.keys() == second.keys()
    for name in first:
        torch.testing.assert_close(
            torch.as_tensor(first[name]), torch.as_tensor(second[name]), rtol=0, atol=0
        )

    changed = evaluate(make_batches(), model, torch.nn.MSELoss(), 1.0, "cpu", seed=124)
    assert first["reconstruction_mse"] != changed["reconstruction_mse"]
    assert first["mean_latent_mse"] == changed["mean_latent_mse"]
    assert first["kl"] == changed["kl"]


@torch.no_grad()
def check_stochastic_forward():
    torch.manual_seed(42)
    model = vae(4, 8, 2)
    model.eval()
    X, _ = make_batches()[0]
    first_preds, first_mu, first_logvar = model(X)
    second_preds, second_mu, second_logvar = model(X)

    assert torch.equal(first_mu, second_mu)
    assert torch.equal(first_logvar, second_logvar)
    assert not torch.equal(first_preds, second_preds)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not __debug__:
        raise RuntimeError("Run without -O so the assertions execute.")
    torch.set_num_threads(1)
    torch.manual_seed(42)

    check_weighted_training()
    print("sample-weighted training metrics check passed")
    check_weighted_evaluation()
    print("sample-weighted validation metrics and mean decoding check passed")
    check_evaluation_preserves_state()
    print("validation mode, gradients, weights, and RNG check passed")
    check_repeatable_evaluation()
    print("repeatable validation check passed")
    check_stochastic_forward()
    print("stochastic forward check passed")
    print("5 experiment checks passed")


if __name__ == "__main__":
    main()
