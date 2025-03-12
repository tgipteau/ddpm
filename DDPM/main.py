## IMPORTS
import random
import imageio
import numpy as np
import yaml
from torchvision.utils import save_image
import os

from tqdm import tqdm
import matplotlib.pyplot as plt

import einops
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor, Lambda, Normalize
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import CIFAR10


# chargement de la configuration
config = yaml.safe_load(open('config_ddpm.yaml', 'r'))

# sélection du modèle unet et du dataset
unet_model = config['unet_model']
if unet_model == 'unet_cifar':
    from unet_cifar import MyUNet
    c, h, w = 3, 32, 32
    # Pipeline de transformations CIFAR
    transform = Compose([
        ToTensor(),  # Convertit en tensor et normalise les pixels dans [0, 1]
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalisation RGB pour CIFAR-10
    ])
    dataset = CIFAR10(root='./datasets', download=True, train=True, transform=transform)

elif unet_model == 'unet_mnist':
    from unet_mnist import MyUNet
    c, h, w = 1, 28, 28
    # Pipeline de tranformations MNIST
    transform = Compose([
        ToTensor(),
        Lambda(lambda x: (x - 0.5) * 2)]
    )
    dataset = MNIST("./datasets", download=True, train=True, transform=transform)
    
    
# paramètres de reproductibilité
SEED = config['seed']
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# paramètres d'entraînement
batch_size = config['batch_size']
n_epochs = config['n_epochs']
lr = config['lr']
device = config['device']
store_path = config['store_path']

# paramètres du modèle unet
n_steps = config["n_steps"]
min_beta = config["min_beta"]
max_beta = config["max_beta"]


# DDPM class
class MyDDPM(nn.Module):
    
    def __init__(self, network, image_chw=(1, 28, 28)):
        
        super(MyDDPM, self).__init__()
        self.device = device
        self.network = network.to(device)
        self.image_chw = image_chw

        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)
    
    def forward(self, x0, t, eta=None):
        # transforme l'input (x0) en image bruitée au temps t (passé en argument), avec le bruit eta
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]
        
        if eta is None:
            eta = torch.randn(n, c, h, w).to(device)
        
        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy
    
    def backward(self, x, t):
        return self.network(x, t)  # envoie x, t dans le UNET. En sortie, le bruit estimé sur x au temps t


def show_first_batch(loader):
    for batch in loader:
        show_images(batch[0], "Images in the first batch")
        break

def show_forward(ddpm, loader):
    for batch in loader:
        imgs = batch[0]

        show_images(imgs, "Original images")

        for percent in [0.25, 0.5, 0.75, 1]:
            show_images(
                ddpm(imgs.to(device),
                     [int(percent * n_steps) - 1 for _ in range(len(imgs))]),
                f"DDPM Noisy images {int(percent * 100)}%"
            )
        break
        

def training_loop(ddpm, loader, optim, display=False):
    mse = nn.MSELoss()
    best_loss = float("inf")

    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, leave=False, position=0,
                                          desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
            # Loading data
            x0 = batch[0].to(device)
            n = len(x0) # taille effective du batch

            # fabrication du bruit eta pour chaque image x0 du batch (forward)
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)

            # appel à forward pour obtenir le batch bruité
            noisy_imgs = ddpm(x0, t, eta)

            # estimation du bruit par le modèle (backward)
            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))

            # loss : mse entre bruit prédit et bruit réel
            loss = mse(eta_theta, eta)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset) # loss moyenne de l'epoch

        # affichage de l'image générée à cet epoch
        if display:
            show_images(generate_new_images(ddpm), f"Généré à l'epoch {epoch + 1}")

        log_string = f"Loss epoch {epoch + 1}: {epoch_loss:.3f}"

        # si le modèle est le meilleur jusque ici (loss minimale)
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Meilleur modèle actuel ; sauvegardé."

        print(log_string)
        
def generate_for_comparison(n_samples=100) :
    
    with torch.no_grad():
        
        # Starting from random noise
        x = torch.randn(n_samples, c, h, w).to(device)
        
        for idx, t in enumerate(list(range(n_steps))[::-1]):
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)
            
            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]
            
            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)
            
            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)
                
                # sigma_t au carré = beta_t
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()
                
                # ajout de bruit "à la Langevin"
                x = x + sigma_t * z
            
    
        for i, img in enumerate(x.detach().cpu().view(n_samples, h, w)):
            # print(img.shape)
            # plt.imshow(img, cmap="gray")
            # plt.show()
            img = (img - img.min()) / (img.max() - img.min())
            save_image(img, os.path.join(store_path, f"ddpm_{i + 1}.png"))


### MAIN
if __name__ == "__main__":
    
    loader = DataLoader(dataset, batch_size, shuffle=True)
    
    # construction du modèle
    ddpm = MyDDPM(MyUNet())
    
    # quelques fonctions de visualisation, désactivées par défaut
    """
    show_first_batch(loader) # voir le premier batch
    show_forward(ddpm, loader) # voir le processus de bruitage
    generated = generate_new_images(ddpm, gif_name="before_training.gif") # voir les images générées par le modèle neuf
    show_images(generated, "Images generated before training")
    """
    
    # chargement du dernier modèle enregistré
    if config["old_model"]:
        # chargement du meilleur modèle d'avant
        ddpm.load_state_dict(torch.load(store_path, map_location=device))
        ddpm.eval()
        print("Modèle précédent rechargé")

    # entraînement
    if config["train"]:
        training_loop(ddpm, loader, optim=Adam(ddpm.parameters(), lr), display=config['train_display'])
        
        best_model = MyDDPM(MyUNet())
        best_model.load_state_dict(torch.load(store_path, map_location=device))
        best_model.eval()
        print("Entraînement terminé")
    
    else :
        best_model = ddpm
    
    for name, param in ddpm.named_parameters():
        print(f"{name}: {param.numel()} paramètres")
    
    total_params = sum(p.numel() for p in ddpm.parameters())
    print(f"Nombre total de paramètres : {total_params}")
    
    generate_for_comparison()
    
