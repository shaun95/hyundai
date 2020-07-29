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
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
#Load Data Pickle
"""
with open(params.ACC_DATA, 'rb') as f:
    acc_pickle = pickle.load(f)

with open(params.SOUND_DATA, 'rb') as f:
    sound_pickle = pickle.load(f)
"""
#%%
"""Load Model"""
model = models.WaveNet_Linear(layer_size=10, stack_size=1).to(device)
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
#acc_data = np.load(os.path.join(os.getcwd(), "sample_acc.npy"))
#sound_data = np.load(os.path.join(os.getcwd(), "sample_sound.npy"))
# %%
"""PreProcess Data"""
# 1. Mu encode the data
mu_encoder = torchaudio.transforms.MuLawEncoding(quantization_channels=256)
mu_decoder = torchaudio.transforms.MuLawDecoding(quantization_channels=256)

# %%
transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor()])
dataset = utils.Wavenet_Dataset(acc_data, sound_data, receptive_field, transform=transform)

#%%
BATCH = 8
EPOCH = 100
dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=BATCH)
#%%
#Define Loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
loss_fn = nn.MSELoss()

# %%
global_step = 0
min_loss = 9999999.
epoch_loss = []
for epoch in range(EPOCH):
    epoch_loss = []
    with tqdm(dataloader, ncols=50, desc="EPOCH : {}".format(epoch + 1)) as _tqdm:
        for idx, (x, y) in enumerate(_tqdm):
            # Forward
            #weights_before = model.state_dict()
            #print(weights_before["causal_conv.conv.weight"])

            model.train()
            x, y = next(iter(dataloader))
            x = x.squeeze(1).float().to(device)
            y = y.squeeze(1).float().to(device)

            optimizer.zero_grad()
            pred = model(x)

            s_filtered = Conv_S(pred, device=device)

            loss = loss_fn(s_filtered, y)
            loss.backward()
            optimizer.step()

            #weights_after = model.state_dict()
            #print(weights_after["causal_conv.conv.weight"])

            global_step += 1
            _loss = loss.item()
            writer.add_scalar("Loss/Train", _loss, global_step)
            epoch_loss.append(_loss)
            _tqdm.set_postfix(loss='{0:.4f}'.format(_loss))
            if idx % 10 == 0:
                for data_idx in range(8):
                    fig = plt.figure()
                    wav_plot = fig.add_subplot()
                    wav_plot.plot(s_filtered.detach().cpu().numpy()[0, :, 0], color="red")
                    wav_plot.plot(y.detach().cpu().numpy()[0, :, 0], color="blue")
                    
                    writer.add_figure("Wavefrom/Channel" + str(data_idx), fig, data_idx)
    
    e_loss = np.mean(np.array(epoch_loss))
    writer.add_scalar("Loss/Epoch_loss", e_loss, epoch + 1)
    if e_loss <= min_loss:
        torch.save(model, os.path.join(os.getcwd(), "Wavenet_Linear.h5"))
