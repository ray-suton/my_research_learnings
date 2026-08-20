import torch
from torch import nn
class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def build_model(model_name: str = "small_cnn", **model_kwargs):
    if model_name in ("small_cnn", "cnn", "baseline"):
        return SmallCNN()
    if model_name in ("mini_resnet", "resnet"):
        return MiniResNet(**model_kwargs)

    raise ValueError(f"Unknown model_name: {model_name}")

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding) -> None:
        super().__init__()
        self.input_channel = in_channel
        self.output_channel = out_channel
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size= kernel_size, stride = stride, padding = padding )
        self.batch1 = nn.BatchNorm2d(out_channel)
        self.batch2 = nn.BatchNorm2d(out_channel)
        self.batch3 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size = kernel_size, stride = 1, padding = padding)
        self.idenconv = nn.Conv2d(in_channel, out_channel, kernel_size= 1, stride = stride)
    def forward(self, x):
        if self.input_channel == self.output_channel and self.stride == 1:
            identity = x
        else:
            ident = self.idenconv(x)
            identity = self.batch3(ident)


        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = x + identity 
        x = self.relu(x)

        return x 


class MiniResNet(nn.Module):
    def __init__(self, in_channel = 3, mid_out_channel1 = 32, mid_out_channel2 = 64, final_out_channel = 128, stride = 2, padding = 1, kernel_size = 3, label_num = 10):
        super().__init__()
        self.stemconv = nn.Sequential(
            nn.Conv2d(in_channel, mid_out_channel1, kernel_size = kernel_size, stride = 1, padding = padding),
            nn.BatchNorm2d(mid_out_channel1),
            nn.ReLU(),
            )
        self.resblock1 = ResidualBlock(mid_out_channel1, mid_out_channel1, kernel_size, stride = 1, padding = padding)
        self.resblock2 = ResidualBlock(mid_out_channel1, mid_out_channel2, kernel_size, stride, padding)
        self.resblock3 = ResidualBlock(mid_out_channel2, final_out_channel, kernel_size, stride, padding)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(final_out_channel, label_num)
    
    def forward(self, x):
        x = self.stemconv(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x