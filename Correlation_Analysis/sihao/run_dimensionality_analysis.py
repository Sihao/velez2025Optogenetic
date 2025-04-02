import os

import matplotlib.pyplot as plt

from experiment_io import load_traces, load_stim_info, load_positions
from timeseries import preprocess_traces, create_timeseries
from plotting import *
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import sklearn as sk
from umap import UMAP

# Define working directory
wd = '/mnt/bronknas/Singles/Data_Singles_To_Analyze/Data_singles_2/240823/Fish_H/avgs/merged/'

# Load neural data
traces = load_traces(os.path.join(wd, 'merged_raw.pkl'))

# Load stimulus information
stim_info_path = '/mnt/bronknas/Singles/Data_Singles_To_Analyze/Data_singles_2/240823/Fish_H/avgs/NM_Stimul_setup_Combos_5_2023-08-24_18-51-07.mat'
stim_info = load_stim_info(stim_info_path)

# Get time series for all stimulus pattern and display as heatmap
# Get number of neuromasts
n_neuromasts = int(np.max(stim_info[:, 1]))

# Empty array to store all stimulus timeseries
fs = 2.18
stim_timeseries = np.zeros((n_neuromasts, traces.shape[1]))
for i in range(n_neuromasts):
    # Get stimulus times
    stim_times = get_neuromast_stim_times(stim_info, i+1)
    stim_timeseries[i, :] = create_timeseries(stim_times, fs=fs, duration=traces.shape[1] / 2.18)

# Heatmap of stimulus timeseries (dont interpolate heatmap)
# Invert values for colormap
stim_timeseries = -stim_timeseries
fig, ax = plt.subplots()
ax.plot(stim_timeseries[0,:])
fig.show()

# Load positions
centroids_path = '/mnt/bronknas/Singles/Data_Singles_To_Analyze/Data_singles_2/240823/Fish_H/avgs/merged/merged_centroids_x.pkl'
positions = load_positions(centroids_path)

# Preprocess traces
traces = preprocess_traces(traces, fs=2.18, kernel_decay=3, bleaching_tau=200, filter=True, zscore=True)

# Rastermap sort traces
rmap_idx = get_rmap_idx(traces)

# Plot heatmap of all traces (rastermap sorted)
fig_heatmap, ax_heatmap = plot_heatmap(-traces[rmap_idx, :], interpolation='sinc', cmap='gray')
fig_heatmap.show()

# Plot corresponing traces for cluster in rmap (starting from highest index)
fig_rmap, ax_rmap = plot_traces(traces, rmap_idx[1100:1110])

