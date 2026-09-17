import torch

import torch.nn as nn

class Generator(nn.Module):
    
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.batch_size = 0
        self.layers = nn.Sequential(
            nn.Linear(input_shape,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256, output_shape),
            nn.Tanh()
        )
        self.output_shape = output_shape
        
    def forward(self, x):
        self.batch_size = x.shape[0]
        x = self.layers(x)
        output = torch.tensor(self.output_shape ** 0.5, dtype = int)
        x = torch.reshape(x, (self.batch_size, 1, output, output))
        return x 


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size ** 2, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.layers(x)
        return x 

class CnnGenerator(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.batch_size = 0
        self.layers = nn.Sequential(
            nn.Linear(input_shape, 128 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(dim = 1, unflattened_size= (128, 7, 7)),
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 3, padding = 1),
            nn.Tanh(),
        )
        
    def forward(self, x):
        self.batch_size = x.shape[0]
        x = self.layers(x)
        return x 


class CNNDiscriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding = 1, stride = 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1, stride = 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1, stride = 2),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(2048, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.layers(x)
        return x 