from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
from modules.anti_aliasing_downsample import Down
from modules.res_block import ResBlock
from modules.average_pooling import FastGlobalAvgPool

class CNN(nn.Module):
    def __init__(self, nf, factors=[2, 2, 2]) -> None:
        super().__init__()
        block = [
            nn.Conv2d(1, nf, 5, 1, padding=2, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, True)            
        ]
        for _, f in enumerate(factors):
            block += [Down(nf, kernel_size=f+1, stride=f)]
            nf *= 2
            block += [ResBlock(dim=nf, dilation=1)]
            block += [ResBlock(dim=nf, dilation=3)]
        self.block = nn.Sequential(*block)        
    
    def forward(self, x):
        x = self.block(x)        
        return x

class TFAggregation(nn.Module):
    def __init__(self, emb_dim, ff_dim, n_heads, n_layers, p) -> None:
        super().__init__()
        self.emb_dim = emb_dim        
        
        from modules.transformer_encoder_my import TFEncoder
        self.tf = TFEncoder(num_layers=n_layers, num_heads=n_heads, d_model=emb_dim, ff_hidden_dim=ff_dim, p=p, norm=nn.LayerNorm(emb_dim))
        
        self.pos_emb = nn.Conv2d(emb_dim, emb_dim, kernel_size=7, stride=1, padding=3, padding_mode='zeros', groups=emb_dim, bias=True)
        
        self.avg_pool = FastGlobalAvgPool(flatten=True)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):                
        d_sz = x.shape[-1]
        x = x + self.pos_emb(x)
        x = x.view(x.shape[0], self.emb_dim, -1).transpose(2, 1).contiguous()        
        x = self.tf(x)
        x = x.transpose(2, 1).contiguous()
        x = x.view(x.shape[0], self.emb_dim, d_sz, d_sz) 
        return x

class Net(nn.Module):
    def __init__(self, emb_dim, nf, factors) -> None:
        super().__init__()
        self.nf = nf
        self.cnn = CNN(nf=16, factors=factors)
        self.tf = TFAggregation(emb_dim=emb_dim, ff_dim=emb_dim*4, n_heads=2, n_layers=4, p=0.1)                        
        self.up = []
        for i in range(3):
            self.up += [
                nn.ConvTranspose2d(emb_dim, emb_dim//2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(emb_dim//2),
                nn.LeakyReLU(0.2, True),
                ResBlock(dilation=1, dim=emb_dim//2),
                ResBlock(dilation=3, dim=emb_dim//2),
                ResBlock(dilation=5, dim=emb_dim//2),
            ]
            emb_dim //= 2
        self.up = nn.Sequential(*self.up)
        self.last = nn.Conv2d(emb_dim, 1, kernel_size=5, padding=2)

    def forward(self, x):
        y = self.cnn(x)
        y = self.tf(y)
        y = self.up(y)
        y = x * torch.sigmoid(self.last(y)-x)
        return y

if __name__ == "__main__":    
    pass