#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchvision
from torchsummary import summary
from tensorboardX import SummaryWriter

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import IPython.display as ipd
import pickle
from scipy.io import loadmat
from tqdm import tqdm
import time

import models
import utils
import params
from transition_S import Conv_S
# %%
"""Settings"""
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = "cuda" if torch.cuda.is_available() else "cpu"

"""Set Tensorboard Writer"""
writer_path = os.path.join(os.getcwd(), "wavenet_linear")
plot_save_path = os.path.join(os.getcwd(), "wavenet_linear_plots")

if not os.path.isdir(writer_path):
    os.mkdir(writer_path)

if not os.path.isdir(plot_save_path):
    os.mkdir(plot_save_path)

writer = SummaryWriter(writer_path)

"""HYPER PARAMETERS"""
input_channels = 12
output_channels = 8
win_size = 512 # Set it after calculate receptive field is calcuted
hop_len = 128

#%%
"""Load Model"""
model = models.WaveNet(layer_size=10, stack_size=1).to(device)
summary(model, (19999, 12))
receptive_field = model.calc_receptive_fields(layer_size=10, stack_size=1)
win_size += receptive_field
# %%
print("================> Loading DATA <===================")
#Load Data Pickle
#with open(params.ACC_DATA, 'rb') as f:
#    acc_pickle = pickle.load(f)

#with open(params.SOUND_DATA, 'rb') as f:
#    sound_pickle = pickle.load(f)

#Load Dataset
#acc_data = np.concatenate(list(map(lambda x : utils.slice_window(x, win_size, hop_len), acc_pickle)), axis=0)
#sound_data = np.concatenate(list(map(lambda x : utils.slice_window(x, win_size, hop_len), sound_pickle)), axis=0)
#np.save(os.path.join(params.PATH, "train_acc_win_512_hop_128.npy"), acc_data)
#np.save(os.path.join(params.PATH, "train_sound_win_512_hop_128.npy"), sound_data)
acc_data = np.load(params.ACC_NPY)
sound_data = np.load(params.SOUND_NPY)
#acc_data = np.load(os.path.join(os.getcwd(), "sample_acc.npy"))
#sound_data = np.load(os.path.join(os.getcwd(), "sample_sound.npy"))
print("================> DATA Loaded Completed <===================")
# %%
"""PreProcess Data"""
print("================> DATA Preprocessing <===================")
transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    utils.mu_law_encoder()])
dataset = utils.Wavenet_Dataset(acc_data, sound_data, receptive_field, transform=transform)
print("================> DATA Preprocessed <===================")

#%%
BATCH = 16
EPOCH = 30
dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=BATCH)
#%%
#Define Loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
loss_fn = nn.CrossEntropyLoss()

# %%
global_step = 0
min_loss = 9999999.
epoch_loss = []
print("================> Train START <===================")
for epoch in range(EPOCH):
    model.train()
    with tqdm(dataloader, ncols=100) as _tqdm:
        for idx, (x, y) in enumerate(_tqdm):
            optimizer.zero_grad()
            x = x.float().squeeze(1).to(device)
            y = y.long().squeeze(1).to(device)

            pred = model(x)

            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            #compute accuracy
            _pred = torch.argmax(pred, dim=1)
            corrects = (_pred == y).sum().item()
            accuracy = corrects / (BATCH * win_size * output_channels)

            _tqdm.set_postfix(accuracy = "{:.2f}{}".format(accuracy, corrects))
