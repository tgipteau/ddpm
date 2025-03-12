import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from transformers import PreTrainedModel, PretrainedConfig

import os
from tqdm import tqdm

# Hyperparamètres
latent_dim = 16
batch_size = 128
epochs = 100
lr = 1e-3

# Prétraitement des données
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_loader = DataLoader(datasets.MNIST(root="./data", train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)

# Définition du VAE
class VAEConfig(PretrainedConfig):
    def __init__(self, latent_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

class VAE(PreTrainedModel):
    config_class = VAEConfig

    def __init__(self, config):
        super().__init__(config)
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        self.mu = nn.Linear(128, config.latent_dim)
        self.log_var = nn.Linear(128, config.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Sigmoid()  # Génération d'images normalisées entre 0 et 1
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

# Entraînement du modèle
device = torch.device("mps")
vae = VAE(VAEConfig(latent_dim)).to(device)
optimizer = optim.AdamW(vae.parameters(), lr=lr)

def loss_function(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KL = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KL

def train():
    vae.train()
    for epoch in tqdm(range(epochs)):
        train_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon_x, mu, log_var = vae(x)
            loss = loss_function(recon_x, x, mu, log_var)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.4f}")
        torch.save(vae.state_dict(), "./vae.pt")

# Génération d'images
def generate(n_samples=100):
    vae.load_state_dict(torch.load("vae.pt"))
    vae.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim).to(device)  # Bruit aléatoire
        generated_imgs = vae.decode(z).cpu().view(n_samples, 1, 28, 28)  # Génération d'une image
    
    """
    for generated_img in generated_imgs:
        generated_img = generated_img.detach().numpy()
        min = generated_img.min()
        max = generated_img.max()
        generated_img = (generated_img - min) / (max - min)
    """
    
    for i, img in enumerate(generated_imgs):
        save_image(img, os.path.join("./VAE_generated", f"vae_{i + 1}.png"))

### train()
generate()