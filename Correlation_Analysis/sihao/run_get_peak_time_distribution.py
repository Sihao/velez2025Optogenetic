import os
from plotting import plot_heatmap, plot_traces, plot_trial_mosaic
import matplotlib.pyplot as plt
import numpy as np
import re
from experiment_io import load_traces, load_positions, load_stim_info
from timeseries import preprocess_traces, avg_trials
from receptive_fields import compute_sta
from utils import get_rmap_idx

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
traces_filtered = preprocess_traces(traces, fs=fs, kernel_decay=5, bleaching_tau=200, zscore=False, filter=True)

# Average over trials
traces_filtered_avg = avg_trials(traces_filtered, stim_info)
traces_avg = avg_trials(traces, stim_info)

# Get number of neuromasts stimulated
n_stim = len(np.unique(stim_info[:, 1]))

# Approximate inter-neuromast time (stim_time)
trial_time = traces_avg.shape[1]
stim_time = int(np.ceil(trial_time / n_stim))

# Pad out trial_time with zeros to be a multiple of stim_time
if trial_time % stim_time != 0:
    traces_avg = np.pad(traces_avg, ((0, 0), (0, stim_time - (trial_time % stim_time))), mode='empty')

# Reshape traces_avg to (n_neurons, stim_time, n_stimuli)
for i in range(n_stim):
    traces_stim_avg = np.stack(np.split(traces_avg, n_stim, axis=1), axis=2)



# Get peak times of indiciual stims
from utils import get_peak_time

fig, ax = plt.subplots(n_stim, 1, figsize=(10, 10))
for stim in range(n_stim):
    peak_times = get_peak_time(traces_stim_avg[:, :, stim])
    peak_times = peak_times / 2.1802

    # Plot histogram of peak times
    ax[stim].hist(peak_times, bins=15)
    ax[stim].set_xlabel('Time (s)')
    ax[stim].set_ylabel('Number of neurons')
    ax[stim].set_title(f'Stimulus {stim}')

fig.show()


# Set peak times to peak times of first stim
peak_times = get_peak_time(traces_stim_avg[:, :, 0])

# Get global peak times
peak_times_global = get_peak_time(traces_filtered_avg)

# Sort traces_avg by peak times
sort_idx = np.argsort(peak_times_global)
traces_avg_sorted = traces_avg[sort_idx, :]
traces_avg_sorted_filtered = traces_filtered_avg[sort_idx, :]

# Rastermap sort
rmap_idx = get_rmap_idx(traces_avg_sorted_filtered, n_clusters=3)
traces_avg_sorted_filtered = traces_avg_sorted_filtered[rmap_idx, :]
traces_avg_sorted = traces_avg_sorted[rmap_idx, :]


# Plot heatmap for averaged traces
fig_heatmap, ax_heatmap = plot_heatmap(traces_avg_sorted_filtered, fs=fs,
                                       interpolation='sinc')
# Label axes
ax_heatmap.set_xlabel('Time (s)')
ax_heatmap.set_ylabel('Neuron')

# Get stimulation times for individual stims
from timeseries import construct_stimuli_timeseries
stimuli_timeseries = construct_stimuli_timeseries(stim_info, fs=fs, duration=traces.shape[1] / fs)
# Average over trials
stim_avgs = avg_trials(stimuli_timeseries.T, stim_info, zscore=False)
# Binarise
stim_avgs[stim_avgs > 0] = 1
# Set first time frame to 1
stim_avgs[:, 0] = 1

# Plot times as vertical lines
prev_tick = 0
for i, tick in enumerate(np.max(stim_avgs, axis = 0)):
    if tick:
        if i == 0:
            ax_heatmap.axvline(i/fs + 0.1, color='r', linestyle='--')

        if i - prev_tick > 2:
            ax_heatmap.axvline(i/fs, color='r', linestyle='--')
            print(f"Drawing line at {i/fs} s")
        prev_tick = i


fig_heatmap.show()

# Plot 20 random individual neurons
fig, ax = plt.subplots(20, 4, figsize=(10, 10))
ax = ax.flatten()
import random
for i in range(20):
    # Pick random neuron
    rnd_id = random.randint(0, traces_avg.shape[0])

    ax[i].plot(traces_avg_sorted[rnd_id, :])
    ax[i].set_xlabel('Time (s)')
    ax[i].set_ylabel('Fluorescence')
    ax[i].set_title(f'Neuron {rnd_id}')

fig.show()

# Visualise single average response to stimulus
fig, ax = plt.subplots()
ax.plot(traces_stim_avg[0, :, 0])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Fluorescence')
ax.set_title('Single neuron response to first stimulus')
fig.show()

# Sort by largest response amplitude
response_amplitude = np.max(traces_stim_avg[:, :, 0], axis=1)
sort_idx = np.argsort(response_amplitude)
traces_stim_avg_sorted = traces_stim_avg[sort_idx, :, :]


# Plot average response to each stimulus
range_to_plot = range(35,55)
from plotting import plot_trial_mosaic

fig, ax = plot_trial_mosaic(traces_stim_avg_sorted, range_to_plot, fs=fs)
fig.show()


# Get mahalanobis distance
n_neurons = traces.shape[0]
similarities = np.zeros((n_neurons, n_stim, n_stim))
from scipy.spatial.distance import mahalanobis
# Compute covariance between stimuli for each neuron
cov = np.cov(traces_stim_avg[:, :, 0].T)
inv_cov = np.linalg.inv(cov)

for i in range(n_neurons):
    for j in range(n_stim):
        for k in range(n_stim):

            inv_cov = np.linalg.inv(cov)
            # Compute mahalanobis distance
            similarities[i, j, k] = mahalanobis(traces_stim_avg[i, :, j], traces_stim_avg[i, :, k], inv_cov)

# Average over stimuli
similarities_avg = np.mean(similarities, axis=-1)
# heatmap
fig, ax = plt.subplots()
ax.plot(similarities_avg[:, -1])
fig.show()

# Get indices of lowest 10 similarity absolute value
idx_to_plot = []
for i in range(n_stim):
    threshold = np.mean(similarities_avg[:, i]) + 2 * np.std(similarities_avg[:, i])

    idx_to_plot = np.stack(idx_to_plot, np.argwhere(similarities_avg[:, i] > 8).flatten())
    # idx_to_plot.append(np.argwhere(similarities_avg[:, i] > 8).flatten())

idx_to_plot = np.array(idx_to_plot).flatten()


# Mosaic those
fig, ax = plot_trial_mosaic(traces_stim_avg, idx_to_plot[:10], fs=fs, visual_control=False)
fig.show()