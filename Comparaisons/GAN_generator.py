import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import save_image

import os
from tqdm import tqdm

# Hyperparamètres
latent_dim = 64
batch_size = 128
epochs = 100
lr = 2e-4

# Prétraitement des données
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_loader = DataLoader(datasets.MNIST(root="./data", train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)

# Générateur
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Tanh()  # Sortie entre -1 et 1
        )

    def forward(self, z):
        return self.model(z)

# Discriminateur
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Probabilité d’être réel
        )

    def forward(self, x):
        return self.model(x)

# Instanciation des modèles
device = torch.device("mps")
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# Optimiseurs
optim_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optim_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Perte
loss_fn = nn.BCELoss()

def train():
    # Entraînement du GAN
    generator.train()
    discriminator.train()
    
    for epoch in range(epochs):
        for real_imgs, _ in train_loader:
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)
    
            # Labels réels et faux
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
    
            # Entraînement du Discriminateur
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z)
            real_loss = loss_fn(discriminator(real_imgs), real_labels)
            fake_loss = loss_fn(discriminator(fake_imgs.detach()), fake_labels)
            loss_D = real_loss + fake_loss
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()
    
            # Entraînement du Générateur
            fake_imgs = generator(z)
            loss_G = loss_fn(discriminator(fake_imgs), real_labels)
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()
    
        print(f"Epoch {epoch+1}, Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")

    torch.save(generator.state_dict(), "GAN_generator.pt")


# Génération d'une image

def generate(n_samples=100):
    generator.load_state_dict(torch.load("GAN_generator.pt"))
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim).to(device)  # Bruit aléatoire
        generated_imgs = generator(z).cpu().view(n_samples, 28, 28)  # Génération d'une image
        
    """
    for generated_img in generated_imgs:
        generated_img = generated_img.detach().numpy()
        min = generated_img.min()
        max = generated_img.max()
        generated_img = (generated_img - min) / (max - min)
    """
    
    for i, img in enumerate(generated_imgs):
        save_image(img, os.path.join("./GAN_generated", f"gan_{i+1}.png"))

### train()
generate()
