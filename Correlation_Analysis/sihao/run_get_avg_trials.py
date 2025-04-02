import os
from plotting import *
import numpy as np
import re
from experiment_io import load_traces, load_positions, load_stim_info
from timeseries import preprocess_traces, avg_trials
from receptive_fields import compute_sta
from utils import get_rmap_idx

wd = '/mnt/bronknas/Singles/Data_Singles_To_Analyze/Data_singles_2/240823/Fish_H/merged/'

# Load neural data
traces = load_traces(os.path.join(wd, 'merged_raw.pkl'))
# Load stimulus information
stim_info_path = '/mnt/bronknas/Singles/Data_Singles_To_Analyze/Data_singles_2/240823/Fish_H/NM_Stimul_setup_Combos_5_2023-08-24_18-51-07.mat'
stim_info = load_stim_info(stim_info_path)

# Load neural data
traces = load_traces(os.path.join(wd, 'merged_raw.pkl'))
fs = 2.18

# Preprocess traces
traces_filtered = preprocess_traces(traces, fs=2.18, kernel_decay=5, bleaching_tau=200, zscore=False, filter=True)

# Average over trials
traces_filtered_avg = avg_trials(traces_filtered, stim_info)
traces_avg = avg_trials(traces, stim_info)

# Get rastermap indices
rmap_idx_filtered = get_rmap_idx(traces_filtered_avg)
rmap_idx_raw = get_rmap_idx(traces_avg)

# Plot heatmap for averaged traces
fig_heatmap, ax_heatmap = plot_heatmap(traces_filtered_avg[rmap_idx_filtered, :], fs=fs,
                                       extent=[0, traces_avg.shape[1]/2.18, traces.shape[0], 0])
# Label axes
ax_heatmap.set_xlabel('Time (s)')
ax_heatmap.set_ylabel('Neuron')
fig_heatmap.show()

fig_raw, ax_raw = plot_heatmap(traces_avg[rmap_idx_raw, :], fs=fs)
fig_raw.show()



