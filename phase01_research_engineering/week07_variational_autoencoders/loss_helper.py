import torch

def vae_loss(loss, mu, logvar, beta, constant):
    kl_per_dim = 0.5 * (
    mu.square() + logvar.exp() - 1 - logvar
    )

    kl_per_input = kl_per_dim.sum(dim=-1)
    kl_loss = kl_per_input.mean()

    total_loss = constant * loss + beta * kl_loss

    return loss, total_loss, kl_loss