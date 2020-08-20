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
from sklearn.model_selection import train_test_split
import time

import modified_wavenet
import utils
import params
#
# from transition_S import Conv_S
# %%
"""Settings"""
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = "cuda" if torch.cuda.is_available() else "cpu"

"""Set Tensorboard Writer"""
writer_path = os.path.join(os.getcwd(), "wavenet_softmax_modified_v2")

if not os.path.isdir(writer_path):
    os.mkdir(writer_path)

writer = SummaryWriter(writer_path)

"""HYPER PARAMETERS"""
input_channels = 12
output_channels = 8
win_size = 512 # Set it after calculate receptive field is calcuted
hop_len = 128

#%%
"""Load Model"""
model = modified_wavenet.WaveNet(layer_size=8, stack_size=1, device=device).to(device)
#model.load_state_dict(torch.load(os.path.join(os.getcwd(), "wavenet_modified_softmax_v2.h5")))
summary(model, (19999, 12))
receptive_field = model.calc_receptive_fields(layer_size=8, stack_size=1)
win_size += receptive_field
# %%
print("================> Loading DATA <===================")
PATH = "/data/datasets/hyundai"
#Load Data Pickle
acc_data = np.load(os.path.join(PATH, "stationary_acc_win_1500_hop_256.npy"))
snd_data = np.load(os.path.join(PATH, "stationary_snd_win_1500_hop_256.npy"))

train_acc, test_acc, train_snd, test_snd = train_test_split(acc_data, snd_data, shuffle=True, random_state=321, test_size=0.1)
#with open(params.ACC_DATA, 'rb') as f:
#    acc_pickle = pickle.load(f)

#with open(params.SOUND_DATA, 'rb') as f:
#    sound_pickle = pickle.load(f)

#Load Dataset
#acc_data = np.concatenate(list(map(lambda x : utils.slice_window(x, win_size, hop_len), acc_pickle)), axis=0)
#sound_data = np.concatenate(list(map(lambda x : utils.slice_window(x, win_size, hop_len), sound_pickle)), axis=0)
#np.save(os.path.join(params.PATH, "train_acc_win_512_hop_128.npy"), acc_data)
#np.save(os.path.join(params.PATH, "train_sound_win_512_hop_128.npy"), sound_data)
#acc_data = np.load(params.ACC_NPY)
#sound_data = np.load(params.SOUND_NPY)
#acc_data = np.load(os.path.join(os.getcwd(), "sample_acc.npy"))
#sound_data = np.load(os.path.join(os.getcwd(), "sample_sound.npy"))
print("================> DATA Loaded Completed <===================")
# %%
"""PreProcess Data"""
print("================> DATA Preprocessing <===================")
x_transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))])
y_transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    utils.mu_law_encoder()])
train_dataset = utils.Wavenet_Dataset(train_acc, train_snd, receptive_field, \
                    x_transform=x_transform, y_transform=y_transform)

test_dataset = utils.Wavenet_Dataset(test_acc, test_snd, receptive_field, \
                    x_transform=x_transform, y_transform=y_transform)
print("================> DATA Preprocessed <===================")

#%%
BATCH = 16
EPOCH = 1000
train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=BATCH)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH)
#%%
#Define Loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999))
loss_fn = nn.CrossEntropyLoss()

# %%
global_step = 0
min_loss = 9999999.
epoch_loss = []
print("================> Train START <===================")
for epoch in range(EPOCH):
    model.train()
    with tqdm(train_dataloader, ncols=100, desc="TRAIN") as _tqdm:
        for idx, (x, y) in enumerate(_tqdm):
            global_step += 1
            optimizer.zero_grad()
            x = x.float().squeeze(1).to(device)
            y = y.long().squeeze(1).to(device)

            logit = model(x)
            pred = torch.nn.functional.log_softmax(logit,dim=1)

            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            #compute accuracy
            with torch.no_grad():
                _pred = torch.argmax(pred, dim=1)
                corrects = (_pred == y).sum().item()
                accuracy = corrects / (BATCH * win_size * output_channels)

            _tqdm.set_postfix(accuracy = "{:.2f} Loss {:.4f}".format(accuracy, loss.item()))

            if idx % 1000 == 0:
                with torch.no_grad():
                    a = _pred.detach().cpu().numpy()[0,:,0]
                    g = y.detach().cpu().numpy()[0,:,0]
                    fig = plt.figure()
                    ax = fig.add_subplot()
                    ax.plot(a)
                    ax.plot(g)
                    writer.add_figure("Train/Plot", fig, global_step)
                    writer.add_scalar("Train/Acc", accuracy, global_step)
                    writer.add_scalar("Train/loss", loss.item(), global_step)
    #evaluate
    model.eval()
    best_acc = 0.
    test_acc = []
    test_loss = []
    with tqdm(test_dataloader, ncols=100, desc="TEST") as _tqdm:
        for idx, (x, y) in enumerate(_tqdm):
            global_step += 1
            optimizer.zero_grad()
            x = x.float().squeeze(1).to(device)
            y = y.long().squeeze(1).to(device)

            logit = model(x)
            pred = torch.nn.functional.log_softmax(logit,dim=1)

            loss = loss_fn(pred, y)

            #compute accuracy
            with torch.no_grad():
                _pred = torch.argmax(pred, dim=1)
                corrects = (_pred == y).sum().item()
                accuracy = corrects / (BATCH * win_size * output_channels)
                accuracy = corrects / (BATCH * win_size * output_channels)
                test_loss.append(loss.item())
                test_acc.append(accuracy)

            _tqdm.set_postfix(accuracy = "{:.2f} Loss {:.4f}".format(accuracy, loss.item()))

            if idx % 50 == 0:
                with torch.no_grad():
                    a = _pred.detach().cpu().numpy()[0,:,0]
                    g = y.detach().cpu().numpy()[0,:,0]
                    fig = plt.figure()
                    ax = fig.add_subplot()
                    ax.plot(a)
                    ax.plot(g)
                    writer.add_figure("TEST/Plot", fig, global_step)
                    writer.add_scalar("TEST/Acc", accuracy, global_step)
                    writer.add_scalar("TEST/loss", loss.item(), global_step)
    test_total_acc = np.mean(np.array(test_acc))
    if test_total_acc > best_acc:
        best_acc = test_total_acc
        torch.save(model.state_dict(), os.path.join(os.getcwd(), "wavenet_modified_softmax_v2.h5"))

# %%
