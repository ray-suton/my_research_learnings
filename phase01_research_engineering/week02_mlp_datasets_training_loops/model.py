import torch
from torch import nn


class FashionMNISTMLP(nn.Module):
    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10,) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
def build_model(hidden_size: int = 128) -> FashionMNISTMLP:
    return FashionMNISTMLP(hidden_size= hidden_size)

def main():
    model = build_model()
    sample_images = torch.randn(64, 1, 28, 28)
    logits = model(sample_images)

    print(f"logits_shape={tuple(logits.shape)}")
    
if __name__ == "__main__":
    main()