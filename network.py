# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.init as weight_init

class AE(nn.Module):
    def __init__(self, input_size, drop_rate=0.5):
        super(AE, self).__init__()
        self.drop_rate = drop_rate
        if self.drop_rate > 0:
            self.drop = nn.Dropout(drop_rate)
        
        self.encode_w1 = nn.Linear(input_size, 64)
        self.encode_w2 = nn.Linear(64, 32)
        self.decode_w1 = nn.Linear(32, 64)
        self.decode_w2 = nn.Linear(64, input_size)
    
    def encoder(self, x):
        x = self.encode_w1(x)
        x = torch.relu(x)
        x = self.encode_w2(x)
        x = torch.relu(x)
        if self.drop_rate:
            x = self.drop(x)
        return x

    def decoder(self, x):
        x = self.decode_w1(x)
        x = torch.relu(x)
        x = self.decode_w2(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class VAE(nn.Module):
    def __init__(self, input_size, drop_rate=0.5):
        super(VAE, self).__init__()
        if drop_rate > 0:
            self.drop = nn.Dropout(drop_rate)

        self.encode_w1 = nn.Linear(input_size, 64)
        self.encode_w2 = nn.Linear(64, 32)
        self.encode_w22 = nn.Linear(64, 32)
        self.decode_w1 = nn.Linear(32, 64)
        self.decode_w2 = nn.Linear(64, input_size)

    def encoder(self, x):
        x = self.encode_w1(x)
        x = torch.relu(x)
        x1 = self.encode_w2(x)
        x2 = self.encode_w22(x)
        return x1, x2

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decoder(self, x):
        x = self.decode_w1(x)
        x = torch.relu(x)
        x = self.decode_w2(x)
        return x

    def forward(self, x):
        mu, log_var = self.encoder(x)
        x = self.sampling(mu, log_var)
        x = self.decoder(x)
        return x, mu, log_var

