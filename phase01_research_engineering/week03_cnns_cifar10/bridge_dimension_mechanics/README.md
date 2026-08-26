# Bridge Mini-Lab - Dimension Mechanics

## Purpose

This bridge sits between Week 3 and Week 4. Week 3 used PyTorch CNNs and a
mini-ResNet; this mini-lab makes the tensor dimensions concrete before moving
into Week 4 experiment discipline.

The goal is not to rebuild PyTorch. The goal is to understand exactly how dense
layers and convolutions move shapes around.

## Scope

Keep this small:

- hand-code one dense-layer forward pass
- hand-code one tiny Conv2D forward pass
- compare the Conv2D result against PyTorch
- write down the shape calculations

Do not implement training, autograd, backpropagation, optimizers, or a full CNN.

## Task 1 - Hand-Coded Dense Forward

Create a tiny dense-network forward pass.

Starter file: `simple_nn_forward.py`

Required shapes:

```text
x:      (batch, input_dim)
W1:     (input_dim, hidden_dim)
b1:     (hidden_dim,)
h:      (batch, hidden_dim)
W2:     (hidden_dim, output_dim)
b2:     (output_dim,)
logits: (batch, output_dim)
```

Checklist:

- [ ] Create a small input tensor by hand.
- [ ] Create `W1`, `b1`, `W2`, and `b2`.
- [ ] Compute `h = relu(x @ W1 + b1)`.
- [ ] Compute `logits = h @ W2 + b2`.
- [ ] Print every intermediate shape.
- [ ] Explain why each matrix multiplication is valid.

Completion criterion: you can predict each output shape before running the
code.

## Task 2 - Hand-Coded Conv2D Forward

Create one tiny Conv2D forward pass using loops.

Starter file: `conv2d_forward.py`

Suggested starting shapes:

```text
x:      (batch=1, in_channels=1, height=4, width=4)
weight: (out_channels=1, in_channels=1, kernel_h=3, kernel_w=3)
bias:   (out_channels=1,)
output: (batch=1, out_channels=1, out_h=2, out_w=2)
```

Use:

```text
stride = 1
padding = 0
```

Shape formula:

```text
out_h = floor((height + 2 * padding - kernel_h) / stride) + 1
out_w = floor((width + 2 * padding - kernel_w) / stride) + 1
```

Checklist:

- [ ] Create a `1 x 1 x 4 x 4` input tensor.
- [ ] Create a `1 x 1 x 3 x 3` kernel.
- [ ] Compute the output height and width manually.
- [ ] Loop over batch, output channel, output row, and output column.
- [ ] For each output position, multiply the input patch by the kernel and sum.
- [ ] Add the bias.
- [ ] Print the output tensor and shape.

Completion criterion: you can point to one output cell and explain exactly
which input patch produced it.

## Task 3 - Compare Against PyTorch

Use PyTorch only as a correctness check.

Checklist:

- [ ] Run the same input, weight, and bias through `torch.nn.functional.conv2d`.
- [ ] Compare the hand-coded output with the PyTorch output.
- [ ] Confirm the maximum absolute difference is near zero.
- [ ] If they disagree, debug the indexing before changing the math.

Completion criterion: your hand-coded Conv2D output matches PyTorch for the
same input, weight, bias, stride, and padding.

## Task 4 - Dimension Reflection

Write a short note answering:

Starter file: `reflection.md`

- What does the output-channel dimension mean?
- Why is the Conv2D weight shape `(out_channels, in_channels, kernel_h, kernel_w)`?
- Why does padding preserve spatial size when `kernel_size=3`, `stride=1`, and
  `padding=1`?
- Why does global average pooling reduce `(batch, channels, height, width)` to
  `(batch, channels, 1, 1)`?

## Done Definition

This bridge is done when:

- the dense forward pass runs
- the Conv2D forward pass runs
- the Conv2D output matches PyTorch
- the shape formulas are written in your own words
- no training loop or full model was added

After this, start Week 4 config-driven training and ablations.
