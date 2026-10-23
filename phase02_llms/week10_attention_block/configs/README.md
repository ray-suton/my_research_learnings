# Configuration Record

No standalone config file exists: the fixture sizes are frozen in the
[experiment plan](../EXPERIMENT_PLAN.md) and applied directly in the test
modules. Record resolved settings beside each actual run before reporting
measurements.

Required fields:

- Run ID, Python and PyTorch versions, hardware, dtype, and seed.
- Fixture sizes: `B`, `T`, `d_model`, `n_heads`, `d_head`, and `d_ff`.
- Norm placement (pre-norm or post-norm) and the layer-norm epsilon.
- Attention scaling factor and the softmax stabilization method.
- Mask convention: boolean or additive; how the causal and padding masks are
  combined; the identity of the padding-mask source tensor.
- Parameter counts and measured forward/backward times.
- Test command, exit status, and the failing fixture when a check fails.

Reuse the Week 9 configuration style once the learner implementation is chosen.
Avoid maintaining a second independent copy of the same defaults.
