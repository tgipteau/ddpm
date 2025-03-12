# 4 blocs down, 4 up . Group norm dynamique. Pas d'attention heads.
# adapté à MNIST

import torch
from torch import nn
import yaml

# Charger la configuration
config = yaml.safe_load(open('config_ddpm.yaml', 'r'))


def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:, ::2] = torch.sin(t * wk[:, ::2])
    embedding[:, 1::2] = torch.cos(t * wk[:, ::2])
    
    return embedding


class MyBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyBlock, self).__init__()
        self.norm = nn.GroupNorm(num_groups=self._get_num_groups(in_c), num_channels=in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize
    
    def forward(self, x):
        out = self.norm(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out
    
    def _get_num_groups(self, num_channels):
        """ Retourne le plus grand nombre de groupes possible tout en restant <= 8 et divisible par num_channels """
        for g in range(min(num_channels, 8), 0, -1):
            if num_channels % g == 0:
                return g
        return 1  # Fallback pour éviter les erreurs


class MyUNet(nn.Module):
    # adapté à MNIST seulement
    def __init__(self, im_size=28, channels=1, n_steps=config['n_steps'],
                 time_emb_dim=config['time_emb_dim']):
        super(MyUNet, self).__init__()
        
        self.im_size = im_size  # Taille de l'image (hauteur et largeur)
        self.channels = channels  # Nombre de canaux d'entrée (par exemple 1 pour MNIST)
        
        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)
        
        # Calcul de la taille de l'image à chaque downsampling
        self.height, self.width = im_size, im_size
        self.downsampling_factor = [2, 2, 2, 2]  # Facteur de downsampling
        
        # Premier half
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            MyBlock(channels, 10),
            MyBlock(10, 10),
            MyBlock(10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)
        
        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            MyBlock(10, 20),
            MyBlock(20, 20),
            MyBlock(20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)
        
        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            MyBlock(20, 40),
            MyBlock(40, 40),
            MyBlock(40, 40)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 1),
            nn.SiLU(),
            nn.Conv2d(40, 40, 4, 2, 1)
        )
        
        # Nouveau bloc Down
        self.te4 = self._make_te(time_emb_dim, 40)
        self.b4 = nn.Sequential(
            MyBlock(40, 80),
            MyBlock(80, 80),
            MyBlock(80, 80)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(80, 80, 2, 1),
            nn.SiLU(),
            nn.Conv2d(80, 80, 4, 2, 1)
        )
        
        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 80)
        self.b_mid = nn.Sequential(
            MyBlock(80, 40),
            MyBlock(40, 40),
            MyBlock(40, 80)
        )
        
        # Second half (Upsampling)
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(80, 80, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(80, 80, 2, 1)
        )
        
        self.te5 = self._make_te(time_emb_dim, 160)
        self.b5 = nn.Sequential(
            MyBlock(160, 80),
            MyBlock(80, 40),
            MyBlock(40, 40)
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 2, 1)
        )
        
        self.te6 = self._make_te(time_emb_dim, 80)
        self.b6 = nn.Sequential(
            MyBlock(80, 40),
            MyBlock(40, 20),
            MyBlock(20, 20)
        )
        
        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te7 = self._make_te(time_emb_dim, 40)
        self.b7 = nn.Sequential(
            MyBlock(40, 20),
            MyBlock(20, 10),
            MyBlock(10, 10)
        )
        
        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            MyBlock(20, 10),
            MyBlock(10, 10),
            MyBlock(10, 10, normalize=False)
        )
        
        self.conv_out = nn.Conv2d(10, channels, 3, 1, 1)
    
    def forward(self, x, t):
        # x is (N, C, H, W) where C is channels, H and W are height and width
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))
        out4 = self.b4(self.down3(out3) + self.te4(t).reshape(n, -1, 1, 1))
        
        out_mid = self.b_mid(self.down4(out4) + self.te_mid(t).reshape(n, -1, 1, 1))
        
        out5 = torch.cat((out4, self.up0(out_mid)), dim=1)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))
        
        out6 = torch.cat((out3, self.up1(out5)), dim=1)
        out6 = self.b6(out6 + self.te6(t).reshape(n, -1, 1, 1))
        
        out7 = torch.cat((out2, self.up2(out6)), dim=1)
        out7 = self.b7(out7 + self.te7(t).reshape(n, -1, 1, 1))
        
        out = torch.cat((out1, self.up3(out7)), dim=1)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))
        
        out = self.conv_out(out)
        
        return out
    
    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )