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

def conv_s(y, s_filter):
    # defined as a function
    ## New Conv S => Validated
    Ls, K, M = s_filter.shape
    Tn = y.shape[0]
    y_buffer = torch.zeros((Ls, K))
    y_p = torch.zeros(y.size())
    #e = torch.zeros(y.size())

    for n in range(Tn):
        for k in range(K):
            for m in range(M):
                y_p[n,m] += torch.dot(y_buffer[:, k], s_filter[:, k, m])

        #e[n, :] = d[n, :] - y_p[n, :]
        y_buffer[1:, :] = y_buffer[:-1, :].clone()
        y_buffer[0, :] = y[n , :]
    return y_p