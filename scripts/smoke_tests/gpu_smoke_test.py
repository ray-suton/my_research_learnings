import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU name", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("GPU memory GB:", round(props.total_memory / 1024**3, 2))
else:
    device = torch.device("cpu")
    print("Torch: Using CPU")

x = torch.randn(2048,2048,device = device)
y = x @ x
print("Result mean:", y.mean().item())
print("Smoke test passed")