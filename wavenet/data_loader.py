#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pickle
from glob import glob
from tqdm import tqdm

import params
# %%
base_path = params.PATH
mat_files = glob(base_path + "/*.mat")
accel_chan = params.accel_chans
# %%
def save_mat_data_in_pickle():
    sound_data = []
    acceleration_data = []
    #with tqdm(mat_files, ncols=100, desc="loading mat data") as _tqdm:
    for i, mat in tqdm(enumerate(mat_files)):
        #print(mat)
        mat_data = sio.loadmat(mat, mat_dtype=True)
        
        if mat_data['Signal_0'][0,0]['y_values']['values'][0,0].shape[-1] > 8:
            sig_sound = mat_data['Signal_1']
            sig_accel = mat_data['Signal_0']
        else:
            sig_sound = mat_data['Signal_0']
            sig_accel = mat_data['Signal_1']
        
        value_sound = sig_sound[0,0]['y_values']['values'][0,0][:,:]
        value_sound = np.array(value_sound)
        #print("value sound: ", value_sound.shape)
        sound_data.append(value_sound)
        value_accel = sig_accel[0,0]['y_values']['values'][0,0][:, :]
        value_accel = np.array(value_accel)
        acceleration_data.append(value_accel)
        #print("acceleration data: ", value_accel.shape)

    with open(os.path.join(base_path, "accel_data.pickle"), "wb") as f:
        pickle.dump(acceleration_data, f)

    with open(os.path.join(base_path, "sound_data.pickle"), "wb") as f:
        pickle.dump(sound_data, f)

# %%
def save_train_data():
    mat_files = glob(base_path + "/NAMYANG*.mat")
    mat_files += glob(base_path + "/PG*.mat")

    sound_data = []
    acceleration_data = []
    #with tqdm(mat_files, ncols=100, desc="loading mat data") as _tqdm:
    for i, mat in tqdm(enumerate(mat_files)):
        #print(mat)
        mat_data = sio.loadmat(mat, mat_dtype=True)
        
        if mat_data['Signal_0'][0,0]['y_values']['values'][0,0].shape[-1] > 8:
            sig_sound = mat_data['Signal_1']
            sig_accel = mat_data['Signal_0']
        else:
            sig_sound = mat_data['Signal_0']
            sig_accel = mat_data['Signal_1']
        
        value_sound = sig_sound[0,0]['y_values']['values'][0,0][:,:]
        value_sound = np.array(value_sound)
        #print("value sound: ", value_sound.shape)
        sound_data.append(value_sound)
        value_accel = sig_accel[0,0]['y_values']['values'][0,0][:, :]
        value_accel = np.array(value_accel)
        acceleration_data.append(value_accel)
        #print("acceleration data: ", value_accel.shape)

    with open(os.path.join(base_path, "train_accel_data.pickle"), "wb") as f:
        pickle.dump(acceleration_data, f)

    with open(os.path.join(base_path, "train_sound_data.pickle"), "wb") as f:
        pickle.dump(sound_data, f)

# %%
