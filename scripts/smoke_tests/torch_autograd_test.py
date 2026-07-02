import torch

a = torch.tensor(5.0, requires_grad = True)
b = 5 * a.square() + 2
print(b)
b.backward()

print(a.grad)

if a.grad == 50 and b == 127:
    print("Success")