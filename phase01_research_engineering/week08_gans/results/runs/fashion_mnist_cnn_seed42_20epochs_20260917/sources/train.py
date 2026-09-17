"""Alternating GAN updates using the learner's sigmoid/BCE convention."""

import torch


def train_batch(
    generator, discriminator, real_images, g_optimizer, d_optimizer,
    loss_fn, noise_dim=64, noise_scale=1.0,
):
    """Update D once, then G once, reusing the original connected fake image."""
    g_optimizer.zero_grad(set_to_none=True)
    d_optimizer.zero_grad(set_to_none=True)
    noise = noise_scale * torch.randn(
        real_images.shape[0], noise_dim,
        device=real_images.device, dtype=real_images.dtype,
    )
    fake_image = generator(noise)
    real_prediction = discriminator(real_images)
    fake_prediction = discriminator(fake_image.detach())
    real_loss = loss_fn(real_prediction, torch.ones_like(real_prediction))
    fake_loss = loss_fn(fake_prediction, torch.zeros_like(fake_prediction))
    d_loss = real_loss + fake_loss
    if not torch.isfinite(d_loss):
        raise FloatingPointError("Non-finite discriminator loss")
    d_loss.backward()
    d_optimizer.step()

    # D's first backward never entered G's graph, so this image can be reused.
    parameters = list(discriminator.parameters())
    previous_flags = [parameter.requires_grad for parameter in parameters]
    discriminator.zero_grad(set_to_none=True)
    for parameter in parameters:
        parameter.requires_grad_(False)
    try:
        updated_prediction = discriminator(fake_image)
        g_loss = loss_fn(
            updated_prediction, torch.ones_like(updated_prediction)
        )
        if not torch.isfinite(g_loss):
            raise FloatingPointError("Non-finite generator loss")
        g_optimizer.zero_grad(set_to_none=True)
        g_loss.backward()
        g_optimizer.step()
    finally:
        for parameter, flag in zip(parameters, previous_flags):
            parameter.requires_grad_(flag)

    return {
        "d_real_loss": real_loss.item(),
        "d_fake_loss": fake_loss.item(),
        "d_loss": d_loss.item(),
        "g_loss": g_loss.item(),
        "d_real_score": real_prediction.detach().mean().item(),
        "d_fake_score_before": fake_prediction.detach().mean().item(),
        "d_fake_score_after": updated_prediction.detach().mean().item(),
    }


def train_one_epoch(
    generator, scale_const, discriminator, real, loss_fn, input_size,
    d_optimizer, g_optimizer,
):
    """Train over a loader of normalized image/label batches; ignore labels.

    Retains the learner's argument order. real is the training DataLoader,
    and input_size is the noise-vector width. Models must share a device.
    """
    generator.train()
    discriminator.train()
    device = next(generator.parameters()).device
    totals: dict[str, float] = {}
    samples = 0
    batches = 0
    for real_images, _ in real:
        real_images = real_images.to(device)
        count = real_images.shape[0]
        values = train_batch(
            generator, discriminator, real_images, g_optimizer, d_optimizer,
            loss_fn, noise_dim=input_size, noise_scale=scale_const,
        )
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + value * count
        samples += count
        batches += 1
    if samples == 0:
        raise ValueError("Training loader contains no images")
    result = {name: value / samples for name, value in totals.items()}
    result.update(samples=samples, batches=batches)
    return result
    
