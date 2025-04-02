import os
from experiment_io import load_traces, load_stim_info, load_positions
from timeseries import filter_traces, create_timeseries, zscore_traces
from significance import compute_null_coherence, compute_stimulus_coherence, get_significant_neurons, compute_xcorr_lag
from plotting import *
import numpy as np
import plotly.graph_objects as go
import seaborn as sns


# Define working directory
wd = '/mnt/bronknas/Singles/Data_Singles_To_Analyze/Data_singles_2/240823/Fish_H/avgs/merged/'

# Load neural data
traces = load_traces(os.path.join(wd, 'merged_raw.pkl'))
# Load stimulus information
stim_info_path = '/mnt/bronknas/Singles/Data_Singles_To_Analyze/Data_singles_2/240823/Fish_H/avgs/NM_Stimul_setup_Combos_5_2023-08-24_18-51-07.mat'
stim_info = load_stim_info(stim_info_path)

# Filter traces
traces_filtered = filter_traces(traces, fs=2.18, kernel_decay=3)

# zscore traces
traces_filtered = zscore_traces(traces_filtered)
traces = zscore_traces(traces)

# # Plot example traces
# fig_ex, ax_ex = plot_traces(traces, range(29925, 29930))
# fig_ex.show()
# #

# # Plot heatmap of all traces (rastermap sorted)
# rmap_idx = get_rmap_idx(traces)
# fig_heatmap, ax_heatmap = plot_heatmap(traces[rmap_idx, :])
# fig_heatmap.show()

# Find neurons which respond to a given stimulus pattern
stim_id = 1

# Compute null distribution of coherences
coherences_null = compute_null_coherence(traces, stim_info, stim_id, fs=2.18, use_saved=True, saved_name='coherence_null_240823_FishH_nm1.npy')


# Compute coherences for all neurons with respect to a given stimulus pattern
coherences = compute_stimulus_coherence(traces_filtered, stim_info, stim_id, fs=2.18, kernel_decay=3)

# Get time series for stimulus pattern
# Get stimulus times
stim_times = get_neuromast_stim_times(stim_info, stim_id)

# Create timeseries of stimulus events
fs = 2.18
stim_timeseries = create_timeseries(stim_times, fs, traces.shape[1] / fs)


# Get highest coherence values
highest_coherence = np.argsort(coherences)[-50:]

# Get lowest coherence values
lowest_coherence = np.argsort(coherences)[:50]

# Plot highest traces
fig, axs = plot_traces(traces, highest_coherence)
fig.show()

# Plot lowest traces
fig, axs = plot_traces(traces, lowest_coherence)
fig.show()

# Get indices of neurons which respond to the stimulus pattern
significant_neurons, significant_neurons_mask = get_significant_neurons(coherences, coherences_null, alpha=0.05, coherence_threshold=0.05)

# Remove neurons which precede stimulus (indicating coherence with other stimulus)
lags= []
for i in range(len(significant_neurons)):
    trace_lag = compute_xcorr_lag(traces_filtered[significant_neurons[i], :].squeeze(), stim_timeseries)
    lags.append(trace_lag)
    if trace_lag < 0 or trace_lag > int(4 * fs):
        significant_neurons_mask[significant_neurons[i]] = False
# New number of significant neurons
n_significant_neurons = np.sum(significant_neurons_mask)


# Get indices for significant neurons
significant_neurons = np.argwhere(significant_neurons_mask).squeeze()
# Intersect significant neurons with highest coherence neurons
significant_neurons = np.intersect1d(significant_neurons, highest_coherence)

# Plot significant neurons with associated coherence values
fig, axs = plot_traces(traces_filtered, significant_neurons[:7])
# Overlay stimulus pattern
for i in range(7):
    axs[i].plot(stim_timeseries * 5, color='red')
    # Text annotation for coherence
    # axs[i].text(0.1, 0.1, f'Coherence: {coherences[significant_neurons[i]]:.2f}', transform=axs[i].transAxes)
fig.show()

# Rastermap sort traces of significant neurons
significant_traces = traces_filtered[significant_neurons, :].squeeze()
rmap_idx = get_rmap_idx(significant_traces)

# Plot heatmap of traces (rastermap sorted)
fig_heatmap, ax_heatmap = plot_heatmap(-significant_traces[rmap_idx, :], interpolation='sinc', cmap='viridis')
fig_heatmap.show()

# Get positions of significant neurons
centroids_path = '/mnt/bronknas/Singles/Data_Singles_To_Analyze/Data_singles_2/240823/Fish_H/avgs/merged/merged_centroids_x.pkl'
positions = load_positions(centroids_path)
significant_positions = positions[significant_neurons_mask, :]
significant_positions = significant_positions[rmap_idx, :]
# Plot all positions in 3d using plotly
fig = plot_significant_positions_plotly(positions, significant_positions)
fig.show(renderer='browser')


