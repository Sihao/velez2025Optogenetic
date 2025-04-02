import os
from plotting import plot_heatmap, plot_traces, plot_trial_mosaic
import matplotlib.pyplot as plt
import numpy as np
import re
from experiment_io import load_traces, load_positions, load_stim_info
from timeseries import preprocess_traces, avg_trials, split_trials
from receptive_fields import compute_sta
from utils import get_rmap_idx
from dimensionality import compute_components

wd = '/mnt/bronknas/freq_controls/output_rsync/nvelez/Fish_C_freq/merged'

# Load neural data
traces = load_traces(os.path.join(wd, 'merged_raw.pkl'))
# Load stimulus information
stim_info_path = '/mnt/bronknas/freq_controls/output_rsync/nvelez/Fish_C_freq/NM_Stimul_setup_single_controls_2024-01-24_14-45-27.mat'
stim_info = load_stim_info(stim_info_path)

# Set sampling frequency
fs = 2.1802

# Load neural data
traces = load_traces(os.path.join(wd, 'merged_raw.pkl'))

# Preprocess traces
traces_filtered = preprocess_traces(traces, fs=fs, kernel_decay=5, bleaching_tau=200, zscore=True, filter=True)

# Split trials
traces_filtered_split = split_trials(traces_filtered, stim_info)

# decompose traces
dpca_obj = compute_components(traces_filtered_split)
