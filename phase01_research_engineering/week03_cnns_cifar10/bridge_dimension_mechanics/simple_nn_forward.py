"""Task 1: hand-coded dense neural-network forward pass.

Fill this in yourself. Keep it to tensor creation, matrix multiplication,
bias addition, ReLU, and shape prints.
"""

import torch


def main():
    #hidden dim is 4
    # TODO: Create x with shape (2, 3).
    x = torch.tensor([[1,3,4],[2,3,4]], dtype = float)
    # TODO: Create W1 with the shape that makes x @ W1 valid.
    w1 = torch.randint(0,10,(3,4), dtype = float)
    # TODO: Create b1 with the hidden dimension shape.
    b1 = torch.randint(0,10,(4,), dtype = float)
    # TODO: Compute hidden pre-activation.
    z1 = x @ w1 + b1
    # TODO: Apply ReLU.
    a1 = torch.clamp(z1, min = 0)
    # TODO: Create W2 and b2 for output_dim = 2.
    w2 = torch.randint(0,10,(4,2), dtype = float)
    b2 = torch.randint(0,10,(2,), dtype = float)
    # TODO: Compute logits.
    logits = a1 @ w2 + b2 
    # TODO: Print x, W1, b1, hidden, W2, b2, and logits shapes.
    stable_logits = logits - torch.max(logits, dim = 1, keepdim = True).values
    softmaxed = torch.exp(stable_logits) / torch.sum(torch.exp(stable_logits), dim = 1, keepdim = True)
    print("x:", x, x.shape)
    print("w1:", w1, w1.shape)
    print("b1:", b1, b1.shape)
    print("a1:", a1, a1.shape)
    print("w2:", w2, w2.shape)
    print("b2:", b2, b2.shape)
    print("logits:", logits, logits.shape)
    print("softmaxed:", softmaxed, softmaxed.shape)
    print("softmax row sums:", softmaxed.sum(dim=1))
if __name__ == "__main__":
    main()

