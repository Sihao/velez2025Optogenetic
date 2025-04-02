import os
import numpy as np
from experiment_io import load_traces, load_stim_info, get_neuromast_identities
from timeseries import preprocess_traces, split_stims, create_timeseries, get_neuromast_stim_times
import re
import matplotlib.pyplot as plt
# set backend to qt5
plt.switch_backend('Qt5Agg')

wd = '/mnt/bronknas/Singles/Data_random_to_analyze/01_randomized/Fish_E_rand/'

# Load neural data
traces = load_traces(os.path.join(wd, 'merged', 'merged_raw.pkl'))
# Load stimulus information
# Regex find NM_Stimul_setup_Combos_5_YYYY-MM-DD_HH-MM-SS.mat
pattern = 'NM_Stimul_setup_'
dir_content = os.listdir(wd)

file_found = False
for file in dir_content:
    if re.search(pattern, file):
        stim_info_path = os.path.join(wd, file)
        file_found = True
        break

if not file_found:
    print('Stimulus information file not found. Skipping...')
    # Skip this fish
    raise FileNotFoundError('Stimulus information file not found')

stim_info = load_stim_info(stim_info_path)

# Set sampling frequency
fs = 2.1802

# Preprocess traces
traces_filtered = preprocess_traces(traces, fs=fs, kernel_decay=5, bleaching_tau=400, zscore=False, filter=True)


# Split into trials
# Determine number of samples per trial
n_stim = len(np.unique(stim_info[:, 1]))
timeseries_all = []
for i in range(1, n_stim + 1):
    stim_times = get_neuromast_stim_times(stim_info, i)
    timeseries = create_timeseries(stim_times, fs, duration=traces_filtered.shape[1] / fs, single_ticks=True)
    timeseries_all.append(timeseries)
# Make into array
timeseries_all = np.array(timeseries_all)
# Bitwise OR over all trials
timeseries_all = np.any(timeseries_all, axis=0)

# Difference between stimulations for one stimulus
stim_indices = np.where(timeseries_all == 1)[0]

# Get differences
stim_diff = np.diff(stim_indices)

# Get max difference
n_samples = int((np.max(stim_diff) - 1))

# (# neurons, # samples, # trials, # stim)
traces_filtered_trials = split_stims(traces_filtered, stim_info, fs=fs, n_samples=n_samples)
traces_trials = split_stims(traces, stim_info, fs=fs, n_samples=n_samples)

# Get number of neuromasts stimulated
n_stim = len(np.unique(stim_info[:, 1]))


# Get highest responding for each class
neuron_selector = []
for i in range(n_stim):
    # Get 95th percentile zscore
    threshold = np.percentile(np.max((np.mean(traces_filtered_trials[:,:,:,i], axis=2)), axis=1), 99.977)
    neuron_selector.append(np.argwhere(np.max((np.mean(traces_filtered_trials[:,:,:,i], axis=2)), axis=1) > threshold).squeeze())
neuron_selector = np.unique(np.concatenate(neuron_selector))

# Average over response interval
traces_filtered_trials_avg = np.max(traces_filtered_trials[neuron_selector, :, :, :], axis=1)
traces_trials_avg = np.max(traces_trials[neuron_selector, :, :, :], axis=1)

# Unwrap final axis into penultimate axis
traces_stim_avg = np.concatenate(np.split(traces_filtered_trials_avg, n_stim, axis=2), axis=1).squeeze()
traces_stim_raw_avg = np.concatenate(np.split(traces_trials_avg, n_stim, axis=2), axis=1).squeeze()


# zscore
traces_stim_avg = (traces_stim_avg - traces_stim_avg.mean(axis=0)) / traces_stim_avg.std(axis=0)
traces_stim_raw_avg = (traces_stim_raw_avg - traces_stim_raw_avg.mean(axis=0)) / traces_stim_raw_avg.std(axis=0)

# Show mosaic for top sparsity neurons
nm_identities = get_neuromast_identities(wd)

# Get neighbour correlation
from dimensionality import compute_neighbour_correlation
neighbour_correlation = compute_neighbour_correlation(traces_trials_avg, nm_identities)

# Remove all zeros
neighbour_correlation = {key: value for key, value in neighbour_correlation.items() if value != 0}
# dict has keys for each neuromast, and values are the correlation with neighbours
# Heatmap for mean correlation with neighbours
import seaborn as sns
keys = list(neighbour_correlation.keys())
fig, ax = plt.subplots(1, len(keys), figsize=(10, 5))
for i in range(len(keys)):
    key = keys[i]
    sns.violinplot(neighbour_correlation[key], ax=ax[i])
    # Same y-axis for all plots
    ax[i].set_ylim(0, 1)

    ax[i].set_title(key)

# Save neighbour correlation
np.save(os.path.join(wd, 'neighbour_correlation.npy'), neighbour_correlation)

