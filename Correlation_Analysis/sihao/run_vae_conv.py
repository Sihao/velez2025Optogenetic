"""
Author: Giacomo Glotzer
"""
# %% 
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch.utils.data as data_utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import torch.nn.functional as F
import vae_encoder
import vae_decoder
import vae_variational_autoencoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm 
import random
from IPython.display import display, HTML
import os
import re

# %% Check GPU Availability 
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_name(1))

# %% Set precision of Torch 
torch.set_float32_matmul_precision('high')

# %% Find 10 Fish names 
file_names = os.listdir("../data/ten_fish/np")
pattern = re.compile(r"[A-Z]_\d+")
fish_names = np.unique([pattern.search(f).group() for f in file_names])

# %% PROCESS STIMULATIONS 

# Import Data 
stim = pd.read_csv("../data/Nico_Stimulations.csv", names=["seconds","neuromast"])

# Define Constants 
TR = 2.1802 # volumes per second 
THRESHOLD = 3 # determines number of seconds between stimulations 

# When do stimulations begin? 
jumps = np.diff(stim.seconds) > THRESHOLD 
jump_indxs = np.where(np.diff(stim.seconds) > THRESHOLD)[0]
jump_indxs = [i+1 for i in jump_indxs]
jump_indxs = np.concatenate([[0],jump_indxs,[len(stim)]])
print(f"Total number of stimulations: {np.sum(jumps)}")

# Add new columns 
stim["TR"] = (stim.seconds * TR).astype(int)
stim["stim_start"] = [1 if i in jump_indxs else 0 for i in range(len(stim))]
stim["stim_num"] = np.searchsorted(jump_indxs, np.arange(len(stim)), side='right')

unique_stims = []
stim_length = []

stim["unique_stim_num"] = 0
for i in range(len(jump_indxs)-1):
    s = stim.iloc[jump_indxs[i]:jump_indxs[i+1],:].neuromast.values
    times = stim.iloc[jump_indxs[i]:jump_indxs[i+1],:].seconds.values
    stim_length.append(np.round(times[-1] - times[0], 5))
    s_as_int = int(''.join(str(e) for e in s))
    if s_as_int not in unique_stims:
        unique_stims.append(s_as_int)
    stim.iloc[jump_indxs[i]:jump_indxs[i+1],-1] = np.where(np.array(unique_stims) == s_as_int)[0][0] + 1

print(f"Total number of unique stimuli: {len(unique_stims)}")
print(f"Stimulus repeated on avg {np.sum(jumps)//len(unique_stims)}")
print(f"Stimuli last on average {np.round(np.mean(stim_length),5)} seconds")

# Edit vector of unique_stims 
unique_stims = [str(i)[0:6] for i in unique_stims]

# Cut stim at max TRs
max_trs = 3264
stim = stim[stim.TR < max_trs]

display(stim)


# %% LOAD ROIs 
"""
ID: Fish ID 
Num: number of fish (1-10)
x,y,z: location of neuromast (unregistered)
AP2-5: Correlation with stimulating this neuromast 
max_1-26: Maximum response for this stimulus 
"""
all_rois = pd.read_csv("../data/all_rois.csv")
display(all_rois)

# %% Plot heatmap of neuron responses to each stimulus 
# test max responses 
max_response = all_rois.iloc[:,-26:]
# calculate the correlation matrix on the numeric columns
corr = max_response.select_dtypes('number').corr()
# plot the heatmap
sns.heatmap(corr)
plt.title("Heatmap: Correlations between neuron responsiveness")


# %% Load averaged activity 
"""
Imaging at 2.1802 volumes/second 
Stimulating for ~0.05 seconds, then 4 seconds recovery. 
929341 neurons 
26 stimulations 
8 TRs per stimulation 
"""

all_traces = np.load("../data/averaged_time_traces_nov_29.npy")
print(f"The averaged data is of shape: {all_traces.shape}")
all_traces = all_traces * 10

# Reshape 
all_traces_stacked = all_traces.reshape(all_traces.shape[0], all_traces.shape[1]*all_traces.shape[2])
print(f"Reshaped: {all_traces_stacked.shape}")

# %% Convert to np.float32
all_traces = all_traces.astype(np.float32)
print(f"Float32 traces have shape: {all_traces.shape}")

# %% Plot all_traces
n=26
neuron_id = 749205
f, axs = plt.subplots(13, 2, figsize=(8, 12))
axs = axs.ravel()
for i in range(n):
    # find starting TR for stimulation  
    activity = all_traces[neuron_id, i, :]
    axs[i].plot(range(len(activity)), activity)
    axs[i].axis(False)
    axs[i].set_title(f"{unique_stims[i]}")
    #axs[i].set_ylim([-0.15, 0.15])

# %% Plot all_traces_stacked

n=20
random_neurons = random.sample(range(all_traces.shape[0]), n)
f, axs = plt.subplots(n, 1, figsize=(8, n))
for i in range(n):
    activity = all_traces_stacked[random_neurons[i],:]
    axs[i].plot(range(len(activity)), activity)
    axs[i].axis(False)
    axs[i].set_title(f"{random_neurons[i]}")
    #axs[i].set_ylim([-1, 1])
    stimulation_trs = np.arange(0,208,8)
    axs[i].scatter(stimulation_trs, [0 for _ in stimulation_trs], c="red", marker="*")



# %% Split into train and test datasets 
batch_size = 52 

# Split on fish_number 
train_indices = np.where(all_rois.Num > 1)[0]
validation_indices = np.where(all_rois.Num == 1)[0]
test_indices = np.where(all_rois.Num > 0)[0]

# Split traces using indices 
train_traces = all_traces[train_indices, :]
test_traces = all_traces[test_indices, :]
validation_traces = all_traces[validation_indices, :]

# Convert to tensor 
train_traces = torch.from_numpy(train_traces) 
test_traces = torch.from_numpy(test_traces)
validation_traces = torch.from_numpy(validation_traces)

# Pad to get 4096 time points (divisible by 64)
"""
time_points = 4096 
front_pad = int(np.floor((time_points - max_timepoints)/2))
back_pad = int(np.ceil((time_points - max_timepoints)/2))
train_traces = F.pad(train_traces,(front_pad, back_pad),'constant',0) 
test_traces = F.pad(test_traces,(front_pad, back_pad),'constant',0) 
"""

# Prepare Dimensions: Neurons x 1 x Timepoints 
"""
#train_traces = train_traces.view(train_traces.size(0),1,train_traces.size(-1)) 
#test_traces = test_traces.view(test_traces.size(0),1,test_traces.size(-1)) 
"""

# Create labels for Dataset using x position 
train_labels = torch.as_tensor(all_rois.iloc[train_indices, -26:].values) 
test_labels = torch.as_tensor(all_rois.iloc[test_indices, -26:].values) 
validation_labels = torch.as_tensor(all_rois.iloc[validation_indices, -26:].values) 

# Verify sizes 
print(f"Train dataset size: {train_traces.size()}")
print(f"Train labels size: {train_labels.size()}")
print(f"Test dataset size: {test_traces.size()}")
print(f"Test labels size: {test_labels.size()}")
print(f"Validation dataset size: {validation_traces.size()}")
print(f"Validation labels size: {validation_labels.size()}")

# Create TensorDatasets 
train_dataset = data_utils.TensorDataset(train_traces, train_labels) 
test_dataset = data_utils.TensorDataset(test_traces, test_labels)
validation_dataset = data_utils.TensorDataset(validation_traces, validation_labels)

# Create DataLoaders (shuffling is off for test_loader)
train_loader = data_utils.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = data_utils.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
validation_loader = data_utils.DataLoader(dataset = validation_dataset, batch_size = batch_size, shuffle = False)

# Verify size of data 
x, y = next(iter(train_loader))
print(f"Batch size: {x.size()}")

# %% Retrieve any changes to function 
importlib.reload(vae_encoder)
importlib.reload(vae_decoder)
importlib.reload(vae_variational_autoencoder)
from vae_encoder import Encoder
from vae_decoder import Decoder
from vae_variational_autoencoder import variational_autoencoder
from lightning.pytorch.loggers import TensorBoardLogger

# %% Create Encoder and Decoder 
num_latents = 64 
max_pool_kernel = 1 # keep this small 
time_points = 8
channels=26
cfg = [52, 104, 208, 416]
cfg_rev = [416, 208, 104, 52, 26]
encoder = Encoder(cfg=cfg, in_channels=channels, in_features=time_points, latent_dim=num_latents, max_pool_kernel=max_pool_kernel)
decoder = Decoder(cfg=cfg_rev, out_features=time_points, latent_dim=num_latents, max_pool_kernel=max_pool_kernel)

# %% Create Autoencoder Object
autoencoder = variational_autoencoder(encoder, decoder, lr=1e-4, kld_weight=1e-3, unique_stims=unique_stims)

# %% Fit Data
logger = TensorBoardLogger("tb_logs", name="december")
early_stop_callback = EarlyStopping(monitor="val_reconstruction_loss", min_delta=0.0000, patience=10, mode="min")
trainer = Trainer(limit_train_batches=200,
                  limit_val_batches=100,
                  check_val_every_n_epoch=1,
                  max_epochs=100, 
                  logger=logger, 
                  callbacks=[early_stop_callback],
                  accelerator="cuda", 
                  devices=1)

# %% Fit Data 
trainer.fit(autoencoder, train_loader, validation_loader)
# tensorboard --logdir C:\Users\gglotzer\vaes\VAE\vae_conv\tb_logs\december\ --port 2480 
# Neuron 52 is good one, as is 20, 

# optuna-dashboard sqlite:///db.sqlite3 
# %% Test Data 
trainer.test(autoencoder, test_loader)

# %% Load VAE weights from checkpoint 
checkpoint_path = r"C:\Users\gglotzer\vaes\VAE\vae_conv\tb_logs\december\version_37\checkpoints\epoch=99-step=20000.ckpt"
checkpoint = torch.load(checkpoint_path)
autoencoder.load_state_dict(checkpoint["state_dict"])


# %% 








