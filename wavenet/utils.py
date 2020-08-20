#%%
import numpy as np
import math
import torch
import torch.nn as nn
import torchaudio
# %%
def data_generation(data, framerate, seq_size=6000, mu=256):
    div = max(data.max(), abs(data.min()))
    data = data/div
    while True:
        if isinstance(data, torch.Tensor):
            start = torch.randint(0, data.shape[0] - seq_size)
            ys = data[start : start + seq_size]
            ys = encode_mu_law(ys, mu)
            yield torch.Tensor(ys[:seq_size])
        
        elif isinstance(data, np.ndarray):
            start = np.random.randint(0, data.shape[0] - seq_size)
            ys = data[start : start + seq_size]
            ys = encode_mu_law(ys, mu)
            yield np.array(ys[:seq_size])

class mu_law_encoder(nn.Module):
    def __init__(self, quantization_channels=256, rescale_factor=100):
        super().__init__()
        self.encoder = torchaudio.transforms.MuLawEncoding(quantization_channels=quantization_channels)
        self.rescale_factor = rescale_factor
    
    def forward(self, x):
        x = x / self.rescale_factor
        x = self.encoder(x)
        return x

class mu_law_decoder(nn.Module):
    def init(self, quantization_channels=256, rescale_factor=100):
        super().__init__()
        self.quantization_channels = quantization_channels
        self.rescale_factor = rescale_factor
        self.decoder = torchaudio.transforms.MuLawDecoding(quantization_channels=quantization_channels)
    
    def forward(self, x):
        x = self.decoder(x)
        x = x * self.rescale_factor
        return x


def OneHot_encoder(depth=256):
    def _onehot_encoder(x):
        if isinstance(x, torch.Tensor):
            onehot = torch.zeros(len(idx), idx.max()+1).scatter_(1, idx.unsqueeze(1), 1.)
        elif isinstance(x, np.ndarray):
            onehot = np.eye(depth)[x]
        return onehot
    return _onehot_encoder

def encode_mu_law(x, mu = 256):
    mu = mu - 1
    fx = np.sign(x)*np.log(1 + mu * abs(x)) / np.log(1+mu)
    encoded = np.floor((fx + 1) / 2*mu + 0.5).astype(np.long)
    return encoded

def decode_mu_law(y, mu=256):
    mu = mu - 1
    fx = (y-0.5) / mu*2-1
    x = np.sign(fx)/mu*((1+mu)**np.abs(fx)-1)
    return x


#%%
def slice_window(data, win_size, hop_len):
    #Slice data and concatenate them
    #return them as numpy
    windows = []
    data_len = data.shape[0]
    n_windows = int((data_len - win_size - hop_len) / hop_len)

    for i in range(n_windows):
        window = data[i * hop_len : (i * hop_len) + win_size]
        windows.append(window)
    
    return np.array(windows)

class Wavenet_Dataset(torch.utils.data.Dataset): 
  def __init__(self, x, y, receptive_field, x_transform=None, y_transform=None):
    self.x_data = x
    self.y_data = y
    
    print("x shape : {}  y shape : {}".format(self.x_data.shape, self.y_data.shape))
    
    self.x_transform = x_transform
    self.y_transform = y_transform
    self.receptive_field = receptive_field

  def __len__(self): 
    return len(self.x_data)

  def __getitem__(self, idx): 
    x = self.x_data[idx, :, :]
    y = self.y_data[idx, self.receptive_field:, :]

    if self.x_transform is not None:
        x = self.x_transform(x)
    
    if self.x_transform is not None:
        y = self.y_transform(y)
    
    return x, y