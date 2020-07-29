#%%
import numpy as np
import librosa
import IPython.display as ipd

from glob import glob
from scipy.io import loadmat

import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
import math

import torch
import torch.nn
import torch.nn.functional as F
# %%
base_path = "/data/datasets/hyundai"

main_data_path = os.path.join(base_path, "data")
main_mats = glob(main_data_path + "/*.mat")

sample_data_path = os.path.join(base_path, "20200619_DNN_ANC_sample_secondary_path")
sample_mats = glob(sample_data_path + "/*.mat")

S_mat_path = "/data/datasets/hyundai/20200619_DNN_ANC_sample_secondary_path/S_128_8_8.mat"
S_mat = loadmat(S_mat_path, mat_dtype=True)

S_data = S_mat['S'] # (Ls, K, M)
S_data.shape
# %%
sine_array = np.arange(0, 10, step=0.01)
sine_array = list(map(lambda x: math.sin(x), sine_array))
sine_array = np.array([sine_array for _ in range(8)]) #(N, M)
sine_array = np.transpose(sine_array, (1,0))

print(S_data.shape, sine_array.shape)

#%%
y = sine_array
y_p = np.zeros(shape=(1000, 8))
y_buffer = np.zeros((128, 8))

print("y shape : {} \ny_p shape : {}\ny_buffer : {}".format(y.shape, y_p.shape, y_buffer.shape))

#%%
def shift_buffer(y_b, k, x):
    y_b[1:, k] = y_b[:-1, k]
    y_b[0, k] = x
    return y_b
#%%
for n in range(1000):
    for k in range(8):
        for m in range(8):
            y_p[n, m] += np.dot(y_buffer[:,k], S_data[:, k, m])
            y_buffer = shift_buffer(y_buffer, k, y[n, k])
#plt.plot(y_p[:, 7])
#plt.show()
# %%
# y = N x K
# y' = N x M
s = torch.Tensor(S_data) #(Ls, K, M)
#s = torch.transpose(s, 0, 2) #(M, K, Ls)
s = torch.transpose(s, 0, 2).transpose(0,1) # (K, M, Ls)
ty_p = torch.Tensor(sine_array.transpose(1,0)[np.newaxis,:,:]) #(M, N) # .transpose((1,0))[np.newaxis, :, :]
print("input : {}, s(weight) : {}".format(ty_p.shape, s.shape))
# %%
ty = F.conv1d(ty_p, s, stride=1, padding=127)
print(ty.shape)
plt.plot(y_p[:,0])
plt.plot(ty.numpy()[0, 0, :1000])
plt.show()

# %%


# %%
