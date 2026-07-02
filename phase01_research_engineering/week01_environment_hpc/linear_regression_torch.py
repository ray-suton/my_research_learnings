import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt




torch.manual_seed(0)

N = 100
x = torch.rand(N, 1) * 10
w = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
noise = torch.rand(N,1)

y = 3*x + 2 + noise

epsilon = 1e-7

def loss_function(y_pred, y_true):
    return  1/2 * torch.mean((y_pred - y_true) ** 2)

def linear_regression_forward(w,b,x):
    return torch.matmul(x,w) + b


def update(w, b, alpha):
    with torch.no_grad():
        if w.grad is None or b.grad is None:
            return
        w -= alpha * w.grad
        b -= alpha * b.grad 

    w.grad.zero_()
    b.grad.zero_()

dif = float('inf')
losses = []
alpha = 0.01
prev_loss = None
epoch = 0
while dif > epsilon:
    epoch += 1
    y_pred = linear_regression_forward(w,b,x)
    loss = loss_function(y_pred,y)
    loss.backward()
    update(w,b,alpha)
    if prev_loss is not None:
        dif = abs(loss.item() - prev_loss)
    prev_loss = loss.item()
    losses.append(prev_loss)

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print(f"Final w: {w.item()} Final b: {b.item()} Final loss: {loss.item()}")

with open("phase01_research_engineering/week01_environment_hpc/results/logs/linear_reg_training_log.txt", "w") as f:
    f.write("Week 1 linear regression run\n")
    f.write(f"Final Loss: {losses[-1]}\n")
    f.write(f"Learned Weight: {w.item()}\n")
    f.write(f"Learned Bias: {b.item()}\n")
    f.write(f"Learning Rate: {alpha}\n")
    f.write(f"Epochs Run: {epoch}\n")
    f.write("Torch Manual Seed: 0\n")
    f.write(f"Sample Count: {N}\n")
    f.write("Loss Definition: 1/2 * torch.mean((y_pred - y_true) ** 2)\n")
    f.write("Synthetic Data Rule: y = 3x + 2 + noise, where noise = torch.rand(N, 1)\n")
    f.write("Loss Plot: phase01_research_engineering/week01_environment_hpc/results/plots/linear_reg_loss.png\n")
    f.write("Prediction Plot: phase01_research_engineering/week01_environment_hpc/results/plots/linear_reg_pred_vs_actual.png\n")


plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("0.5 * MSE Loss")
plt.savefig("phase01_research_engineering/week01_environment_hpc/results/plots/linear_reg_loss.png")
plt.close()



with torch.no_grad():
    y_pred_final = linear_regression_forward(w, b, x)

sorted_idx = torch.argsort(x.squeeze())

plt.figure(figsize=(7, 5))
plt.scatter(
    x.squeeze().detach().numpy(),
    y.squeeze().detach().numpy(),
    s=22,
    alpha=0.7,
    label="Actual",
)
plt.plot(
    x.squeeze()[sorted_idx].detach().numpy(),
    y_pred_final.squeeze()[sorted_idx].detach().numpy(),
    color="red",
    linewidth=2,
    label="Predicted",
)
plt.title("Linear Regression: Predicted vs Actual")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig("phase01_research_engineering/week01_environment_hpc/results/plots/linear_reg_pred_vs_actual.png")
plt.close()
