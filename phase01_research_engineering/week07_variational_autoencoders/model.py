import torch 
import torch.nn as nn

class vae(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = self.encoder = nn.Sequential(
            nn.Flatten(1,-1),
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        lovar = self.fc_logvar(h)
        
        return mu, lovar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z).reshape_as(x)
        return reconstruction, mu, logvar

