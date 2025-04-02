import os
from plotting import *
import numpy as np
from scipy.stats import norm
from experiment_io import load_traces, load_stim_info, load_positions
from timeseries import preprocess_traces, create_timeseries
from significance import compute_granger_causality
from utils import get_neuromast_stim_times, get_rmap_idx

# Define working directory
wd = '/mnt/bronknas/Singles/Data_Singles_To_Analyze/Data_singles_2/240823/Fish_H/avgs/merged/'

# Load neural data
traces = load_traces(os.path.join(wd, 'merged_raw.pkl'))
# Load stimulus information
stim_info_path = '/mnt/bronknas/Singles/Data_Singles_To_Analyze/Data_singles_2/240823/Fish_H/avgs/NM_Stimul_setup_Combos_5_2023-08-24_18-51-07.mat'
stim_info = load_stim_info(stim_info_path)

fitness_raw, _, _, _ = compute_event_exceptionality(traces)
comp_SNR = norm.ppf(np.exp(fitness_raw / traces.shape[0]))

# Filter traces
traces_filtered = preprocess_traces(traces, fs=2.18, kernel_decay=3)

# Find neurons which respond to a given stimulus pattern
stim_id = 4

# Get time series for stimulus pattern
# Get stimulus times
stim_times = get_neuromast_stim_times(stim_info, stim_id)

# Create timeseries of stimulus events
fs = 2.18
stim_timeseries = create_timeseries(stim_times, fs, traces.shape[1] / fs)

# Compute Granger causality for all neurons with respect to a given stimulus pattern
n_lags = 1
save_name = 'granger_240823_FishH_nm' + str(stim_id) + '_lag' + str(n_lags) + '.npy'
granger_causalities = compute_granger_causality(traces_filtered, stim_info, stim_id, fs=2.18, n_lags=n_lags,
                                                use_saved=False, save_name=save_name)

# Save granger causality values
np.save(os.path.join(wd, save_name), granger_causalities)
# Get significant neurons
alpha = 0.001
significant_neurons = np.where(granger_causalities < alpha)[0]
# Print number of significant neurons
print(f'Number of significant neurons: {len(significant_neurons)}')

# Make mask
significant_neurons_mask = np.zeros_like(granger_causalities, dtype=bool)
significant_neurons_mask[significant_neurons] = True

# Get most significant granger causality values
significant_g = np.argsort(granger_causalities)[:10]

# Plot highest traces
fig, axs = plot_traces(traces, significant_g)
fig.show()

# Plot significant neurons with associated coherence values
fig, axs = plot_traces(traces, significant_g[:10])
# Overlay stimulus pattern
for i in range(10):
    axs[i].plot(stim_timeseries * np.max(traces[significant_g[:10]]) / 2, color='red')
    # Text annotation for coherence
    # axs[i].text(0.1, 0.1, f'Coherence: {coherences[significant_neurons[i]]:.2f}', transform=axs[i].transAxes)
fig.show()

# Rastermap sort traces of significant neurons
significant_traces = traces_filtered[significant_neurons, 300:].squeeze()
rmap_idx = get_rmap_idx(significant_traces)

# Plot heatmap of traces (rastermap sorted)
fig_heatmap, ax_heatmap = plot_heatmap(-significant_traces[rmap_idx, :], interpolation='sinc', cmap='gray')
fig_heatmap.show()

# Get positions of significant neurons
centroids_path = '/mnt/bronknas/Singles/Data_Singles_To_Analyze/Data_singles_2/240823/Fish_H/avgs/merged/merged_centroids_x.pkl'
positions = load_positions(centroids_path)
significant_positions = positions[significant_neurons_mask, :]
significant_positions = significant_positions[rmap_idx, :]
# Plot all positions in 3d using plotly
fig = plot_significant_positions_plotly(positions, significant_positions)
fig.show(renderer='browser')

# Plot as 2d projection over z
fig, ax = plt.subplots()
ax.scatter(positions[:, 0], positions[:, 1], c='black', s=1)
ax.scatter(significant_positions[:, 0], significant_positions[:, 1], c='red', s=1)
plt.show()
