import os
from plotting import *
import numpy as np
import re
from experiment_io import load_traces, load_positions, load_stim_info
from timeseries import preprocess_traces, construct_stimuli_timeseries
from receptive_fields import compute_sta

wd = '/mnt/bronknas/Singles/Data_Singles_To_Analyze/Data_singles_2/240823/Fish_H/merged/'

# Load neural data
traces = load_traces(os.path.join(wd, 'merged_raw.pkl'))
# Load stimulus information
stim_info_path = '/mnt/bronknas/Singles/Data_Singles_To_Analyze/Data_singles_2/240823/Fish_H/NM_Stimul_setup_Combos_5_2023-08-24_18-51-07.mat'
stim_info = load_stim_info(stim_info_path)

# Load neural data
traces = load_traces(os.path.join(wd, 'merged_raw.pkl'))

# Preprocess traces
traces_filtered = preprocess_traces(traces, fs=2.18, kernel_decay=5, bleaching_tau=200, zscore=False, filter=True)

# Load granger causality values
# Get directory above wd
# Clean up wd
wd = os.path.normpath(wd)
parent_wd = os.path.dirname(wd)

# List granger files in directory
granger_files = [f for f in os.listdir(parent_wd) if ('nm' in f) and ('granger' in f)]

# Key by nm\d{1} substring
granger_dict = {}
for f in granger_files:
    key = re.search('nm\d{1}', f).group()

    # Load granger causality values
    granger_causalities = np.load(os.path.join(parent_wd, f))

    # Determine significant neurons
    alpha = 0.05
    significant_neurons = np.where(granger_causalities < alpha)[0]

    # Add to dictionary
    granger_dict[key] = significant_neurons

# Construct stimulus timeseries
fs = 2.18
stimuli_timeseries = construct_stimuli_timeseries(stim_info, fs=fs, duration=traces.shape[1] / fs)

# Compute spike-triggered average for each significant neuron
sta = compute_sta(traces_filtered[granger_dict['nm2']], stimuli_timeseries)

# Compute gini coefficient for each neuron
gini = compute_gini(sta)

# Sort STAs by gini (descending)
i = np.argsort(gini)[::-1]
sta = sta[i, :]

# Plot a couple of STAs
fig, ax = plt.subplots(6, 6)
ax = ax.flatten()
n_neuromasts = sta.shape[1]
for i, a in enumerate(ax):
    x_vals = np.linspace(1, n_neuromasts, n_neuromasts)
    a.plot(x_vals, sta[i, :])
    a.set_title(f'Neuron {i}')
    # Shift x-axis to start at 1
fig.show()


# Order STAs by rastermap
from utils import get_rmap_idx
rmap_idx = get_rmap_idx(sta, n_clusters=3)
sta = sta[rmap_idx, :]

# Plot heatmap of STAs
# Normalise each neuron
sta_norm = sta / np.std(sta, axis=1)[:, None]
fig, ax = plt.subplots()
ax.imshow(sta_norm, aspect='auto', interpolation='none')
fig.show()




