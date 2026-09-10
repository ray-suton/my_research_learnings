import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_size = 784, hidden_size = 128, latent_size = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(1,-1),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        x = torch.reshape(x,(x.shape[0],1,28,28))
        return x
