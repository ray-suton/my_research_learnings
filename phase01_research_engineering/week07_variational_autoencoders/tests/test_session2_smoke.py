"""Run the eight Session 2 checks on CPU using cached Fashion-MNIST."""

from contextlib import redirect_stdout
from io import StringIO
import math
from pathlib import Path
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from loader import get_dataloaders
from loss_helper import vae_loss
from model import vae
from train import train_one_epoch


def check_data(train_loader, val_loader, test_loader, X, y):
    assert len(train_loader.dataset) == 55000
    assert len(val_loader.dataset) == 5000
    assert len(test_loader.dataset) == 10000

    train_indices = set(train_loader.dataset.indices)
    val_indices = set(val_loader.dataset.indices)
    assert not train_indices.intersection(val_indices)
    assert len(train_indices.union(val_indices)) == 60000
    assert X.shape == (8, 1, 28, 28)
    assert y.shape == (8,)
    assert torch.isfinite(X).all()
    assert ((X >= 0) & (X <= 1)).all()


@torch.no_grad()
def check_shapes(model, X):
    for batch_size in (1, 8):
        images = X[:batch_size]
        assert images.flatten(1).shape == (batch_size, 784)
        hidden = model.encoder(images)
        assert hidden.shape == (batch_size, 128)

        mu, logvar = model.encode(images)
        assert mu.shape == (batch_size, 32)
        assert logvar.shape == mu.shape
        z = model.reparameterize(mu, logvar)
        assert z.shape == mu.shape
        decoded = model.decoder(z)
        assert decoded.shape == (batch_size, 784)

        preds, forward_mu, forward_logvar = model(images)
        assert preds.shape == images.shape
        assert forward_mu.shape == mu.shape
        assert forward_logvar.shape == logvar.shape
        assert torch.allclose(forward_mu, mu)
        assert torch.allclose(forward_logvar, logvar)
        for value in (hidden, mu, logvar, z, decoded, preds):
            assert torch.isfinite(value).all()
        assert ((preds >= 0) & (preds <= 1)).all()


def check_reparameterization(model):
    mu = torch.zeros(2, 32, requires_grad=True)
    logvar = torch.zeros_like(mu, requires_grad=True)
    z = model.reparameterize(mu, logvar)
    z.sum().backward()

    assert torch.allclose(mu.grad, torch.ones_like(mu))
    assert logvar.grad is not None
    assert torch.isfinite(logvar.grad).all()
    assert torch.count_nonzero(logvar.grad) > 0


def check_zero_kl():
    mu = torch.zeros(8, 32)
    logvar = torch.zeros_like(mu)
    mse_loss, total_loss, kl_loss = vae_loss(
        torch.tensor(0.02), mu, logvar, beta=1.0, constant=784
    )

    for loss in (mse_loss, total_loss, kl_loss):
        assert loss.ndim == 0
        assert torch.isfinite(loss)
    assert torch.allclose(kl_loss, torch.tensor(0.0))
    assert torch.allclose(total_loss, torch.tensor(15.68))


def check_kl_values():
    mu = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    logvar = torch.zeros_like(mu)
    _, total_loss, kl_loss = vae_loss(torch.tensor(2.0), mu, logvar, 0.5, 4)
    assert total_loss.ndim == kl_loss.ndim == 0
    assert torch.allclose(kl_loss, torch.tensor(1.25))
    assert torch.allclose(total_loss, torch.tensor(8.625))

    mu = torch.zeros(1, 1)
    logvar = torch.tensor([[math.log(4.0)]])
    _, _, kl_loss = vae_loss(torch.tensor(0.0), mu, logvar, 1.0, 4)
    assert kl_loss.ndim == 0
    expected_kl = torch.tensor(0.5 * (3.0 - math.log(4.0)))
    assert torch.allclose(kl_loss, expected_kl)


def check_loss_gradients():
    loss = torch.tensor(2.0, requires_grad=True)
    mu = torch.tensor([[1.0, 0.0], [0.0, 2.0]], requires_grad=True)
    logvar = torch.zeros_like(mu, requires_grad=True)
    mse_loss, total_loss, kl_loss = vae_loss(loss, mu, logvar, 0.5, 4)

    assert loss.item() == 2.0
    assert mse_loss.ndim == total_loss.ndim == kl_loss.ndim == 0
    assert torch.allclose(mse_loss, torch.tensor(2.0))
    assert total_loss is not loss
    total_loss.backward()
    assert torch.allclose(loss.grad, torch.tensor(4.0))
    assert torch.allclose(mu.grad, torch.tensor([[0.25, 0.0], [0.0, 0.5]]))
    assert logvar.grad is not None
    assert torch.isfinite(logvar.grad).all()
    assert torch.isfinite(kl_loss)


def check_train_step(X, y):
    torch.manual_seed(42)
    model = vae(784, 128, 32)
    reference = vae(784, 128, 32)
    reference.load_state_dict(model.state_dict())
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    random_state = torch.get_rng_state()

    model.eval()
    output = StringIO()
    with redirect_stdout(output):
        train_one_epoch(
            [(X, y)], model, torch.nn.MSELoss(), 1.0, optimizer, torch.device("cpu")
        )
    assert output.getvalue().strip()
    print(output.getvalue().strip())
    assert model.training

    # Use the same sampling noise and a separate loss calculation as a check.
    torch.set_rng_state(random_state)
    preds, mu, logvar = reference(X)
    reconstruction_loss = (preds - X).square().flatten(1).sum(1).mean()
    posterior = torch.distributions.Normal(mu, (0.5 * logvar).exp())
    prior = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(mu))
    kl_loss = torch.distributions.kl_divergence(posterior, prior).sum(1).mean()
    total_loss = reconstruction_loss + kl_loss
    assert torch.isfinite(total_loss)
    total_loss.backward()

    expected = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.isfinite(parameter).all(), name
        assert not torch.equal(before[name], parameter), name
        assert torch.allclose(
            parameter.grad, expected[name].grad, rtol=1e-4, atol=1e-6
        ), name


@torch.no_grad()
def check_prior_samples(model):
    z = torch.randn(8, 32)
    decoded = model.decoder(z)
    assert decoded.shape == (8, 784)
    images = decoded.reshape(8, 1, 28, 28)
    assert torch.isfinite(images).all()
    assert ((images >= 0) & (images <= 1)).all()


def main():
    if not __debug__:
        raise RuntimeError("Run without -O so the smoke assertions execute.")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=8, download=False, seed=42
    )
    X, y = next(iter(train_loader))
    model = vae(784, 128, 32)

    check_data(train_loader, val_loader, test_loader, X, y)
    print("data check passed")
    check_shapes(model, X)
    print("shape check passed")
    check_reparameterization(model)
    print("reparameterization check passed")
    check_zero_kl()
    print("zero KL check passed")
    check_kl_values()
    print("KL values check passed")
    check_loss_gradients()
    print("loss gradients check passed")
    check_train_step(X, y)
    print("training step check passed")
    check_prior_samples(model)
    print("prior samples check passed")
    print("8 smoke checks passed")


if __name__ == "__main__":
    main()
