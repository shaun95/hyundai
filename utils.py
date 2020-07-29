#%%
import torch
import torch.nn
import torch.nn.functional as F

# %%
"""
From y_p to y, Psuedo code:
s: Dimension of S: Ls x K x M

Ls = 128 -> number of FIR filter taps
K = 8 -> number of speakers for ANC
M = 8 -> number of error microphones inside cabin

Dimension of y(n) -> n x K
Dimension of y’(n) -> n x M

y’(n,M) = y(n,K) * S(Ls,K,M) 

for n = 1:N
    for k = 1:K
        for m = 1:M
            y_p(n,m) = y_p(n,m) + y(Ls,k)' * S(Ls,k,m)
        end
    end
end
"""

def reshape_s(s):
    # original S : (Ls, k, m)
    # S must be : (m, k, Ls)
    s = torch.transpose(s, 0, 2)
    return s

def s_conv(y_p, s):
    # 'y_p' should be (channels, time)
    # 's' should be (time, k, Ls)
    y = F.conv1d(y_p, s)
    return y

# %%
