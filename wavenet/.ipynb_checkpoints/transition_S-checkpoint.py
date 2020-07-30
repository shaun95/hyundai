import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
from scipy.io import loadmat
K = 8
M = 8
Ls = 128

S_mat_path = "/data/datasets/hyundai/20200619_DNN_ANC_sample_secondary_path/S_128_8_8.mat"
S_mat = loadmat(S_mat_path, mat_dtype=True)

S_data = S_mat['S'] # (Ls, K, M)

def Shift(y_buffer, k_idx, value):
    y_buffer[1:, k_idx] = y_buffer[:-1, k_idx]
    y_buffer[0, k_idx] = value
    return y_buffer

def Conv_S(signal, device='cpu'):
    #Process S filter to waveform data
    #the shape of signal should be (batch, time, 8)
    batch_size = signal.size(0)
    time_len = signal.size(1)
    y_pred = torch.zeros((batch_size, time_len, M), device=device)
    S_filter = torch.Tensor(S_data).to(device) #(Ls, K, M)
    Y_buffer = torch.zeros((Ls, K), device=device)
    for batch in range(batch_size):
        for n in range(time_len):
            for k in range(K):
                for m in range(M):
                    y_pred[batch, n, m] += torch.dot(Y_buffer[:, k], S_filter[:, k, m])
                    Y_buffer = Shift(Y_buffer, k, signal[batch, n, k])
        
    return y_pred