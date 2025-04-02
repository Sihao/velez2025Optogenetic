import os
import numpy as np
from experiment_io import load_traces, load_stim_info, get_neuromast_identities
from timeseries import preprocess_traces, split_stims, create_timeseries, get_neuromast_stim_times
import re
import matplotlib.pyplot as plt
import pandas as pd
from significance import compute_stimulus_coherence

file = '/home/slu/cnmfe_traces.npy'

wd = '/mnt/bronknas/Sihao/SCAPE/20240702/'
# Load neural data
traces = np.load(file)

# Set sampling frequency
fs = 2.21


# Truncate first 30 frames
traces = traces[:, 30:]


# Preprocess traces
traces_filtered = preprocess_traces(traces, fs=fs, kernel_decay=20, bleaching_tau=100, zscore=True, filter=True)

# Plot heatmap
from plotting import plot_heatmap

fig, ax = plot_heatmap(traces_filtered, fs=fs, cmap='gray')
fig.show()

# Rastermap
from utils import get_rmap_idx

rmap_idx = get_rmap_idx(traces_filtered, n_clusters=8)
fig, ax = plot_heatmap(traces_filtered[rmap_idx], fs=fs, cmap='gray', interpolation='gaussian')
fig.show()