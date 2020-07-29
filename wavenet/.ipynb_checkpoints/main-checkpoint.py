#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchvision
from torchsummary import summary

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import IPython.display as ipd
import pickle

import models
import utils
import params
# %%
"""Settings"""
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = "cuda" if torch.cuda.is_available() else "cpu"

"""HYPER PARAMETERS"""
input_channels = 12
output_channels = 8
win_size = 512 # Set it after calculate receptive field is calcuted
hop_len = 128
#%%
"""Load Data Pickle"""
with open(params.ACC_DATA, 'rb') as f:
    acc_pickle = pickle.load(f)

with open(params.SOUND_DATA, 'rb') as f:
    sound_pickle = pickle.load(f)
#%%
"""Load Model"""
model = models.WaveNet(layer_size=10, stack_size=1).to(device)
summary(model, (19999, 12))
receptive_field = model.calc_receptive_fields(layer_size=10, stack_size=1)
win_size += receptive_field
# %%
"""
Load Dataset
acc_data = np.concatenate(list(map(lambda x : utils.slice_window(x, win_size, hop_len), acc_pickle)), axis=0)
sound_data = np.concatenate(list(map(lambda x : utils.slice_window(x, win_size, hop_len), sound_pickle)), axis=0)
"""
acc_data = np.load(params.ACC_NPY)
sound_data = np.load(params.SOUND_NPY)

# %%
"""PreProcess Data"""
# 1. Mu encode the data
mu_encoder = torchaudio.transforms.MuLawEncoding(quantization_channels=256)
mu_decoder = torchaudio.transforms.MuLawDecoding(quantization_channels=256)
transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    mu_encoder])
class Wavenet_Dataset(torch.utils.data.Dataset): 
  def __init__(self, x, y, receptive_field, transform=None):
    self.x_data = x
    self.y_data = y
    
    print("x shape : {}  y shape : {}".format(self.x_data.shape, self.y_data.shape))
    
    self.transform = transform
    self.receptive_field = receptive_field

  def __len__(self): 
    return len(self.x_data)

  def __getitem__(self, idx): 
    x = self.x_data[idx, :, :]
    y = self.y_data[idx, self.receptive_field:, :]

    if self.transform is not None:
        x = self.transform(x)
        y = self.transform(y)
    
    return x, y
# %%
dataset = Wavenet_Dataset(acc_data, sound_data, receptive_field, transform=transform)

# %%
