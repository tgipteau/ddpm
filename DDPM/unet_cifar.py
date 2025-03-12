# 4 blocs down, 4 up . Group norm dynamique. Pas d'attention heads.
# adapté à CIFAR


import torch
from torch import nn
import yaml

# Charger la configuration
config = yaml.safe_load(open('config_ddpm.yaml', 'r'))


def sinusoidal_embedding(n, d):
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:, ::2] = torch.sin(t * wk[:, ::2])
    embedding[:, 1::2] = torch.cos(t * wk[:, ::2])
    
    return embedding


class MyBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super().__init__()
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
        for g in range(min(num_channels, 8), 0, -1):
            if num_channels % g == 0:
                return g
        return 1


class MyUNet(nn.Module):
    # adapté à CIFAR
    def __init__(self, im_size=32, channels=3, n_steps=config['n_steps'],
                 time_emb_dim=config['time_emb_dim']):
        super().__init__()
        
        self.im_size = im_size
        self.channels = channels
        
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)
        # Downsampling
        self.te1 = self._make_te(time_emb_dim, 3)
        self.b1 = MyBlock(channels, 32)
        self.down1 = nn.Conv2d(32, 32, 4, 2, 1)  # 32x32 -> 16x16
        
        self.te2 = self._make_te(time_emb_dim, 32)
        self.b2 = MyBlock(32, 64)
        self.down2 = nn.Conv2d(64, 64, 4, 2, 1)  # 16x16 -> 8x8
        
        self.te3 = self._make_te(time_emb_dim, 64)
        self.b3 = MyBlock(64, 128)
        self.down3 = nn.Conv2d(128, 128, 4, 2, 1)  # 8x8 -> 4x4
        
        self.te4 = self._make_te(time_emb_dim, 128)
        self.b4 = MyBlock(128, 256)
        
        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 256)
        self.b_mid = MyBlock(256, 128)
        
        # Upsampling
        self.up1 = nn.ConvTranspose2d(128, 128, 4, 2, 1)  # 4x4 -> 8x8
        self.te5 = self._make_te(time_emb_dim, 256)
        self.b5 = MyBlock(256, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 8x8 -> 16x16
        self.te6 = self._make_te(time_emb_dim, 128)
        self.b6 = MyBlock(128, 32)
        
        self.up3 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 16x16 -> 32x32
        self.te7 = self._make_te(time_emb_dim, 64)
        self.b7 = MyBlock(64, 32)
        
        self.conv_out = nn.Conv2d(32, channels, 3, 1, 1)
    
    def forward(self, x, t):
        t = self.time_embed(t)
        n = len(x)
        
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))
        out4 = self.b4(self.down3(out3) + self.te4(t).reshape(n, -1, 1, 1))
        
        out_mid = self.b_mid(out4 + self.te_mid(t).reshape(n, -1, 1, 1))
        print(out_mid.shape)
        print(out4.shape)
        out5 = torch.cat((out4, self.up1(out_mid)), dim=1)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))
        
        out6 = torch.cat((out3, self.up2(out5)), dim=1)
        out6 = self.b6(out6 + self.te6(t).reshape(n, -1, 1, 1))
        
        out7 = torch.cat((out2, self.up3(out6)), dim=1)
        out7 = self.b7(out7 + self.te7(t).reshape(n, -1, 1, 1))
        
        out = self.conv_out(out7)
        return out
    
    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )