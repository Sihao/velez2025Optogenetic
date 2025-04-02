import os
import numpy as np
from experiment_io import load_traces, load_stim_info, get_neuromast_identities, load_combos_info
from timeseries import preprocess_traces, split_stims, create_timeseries, get_neuromast_stim_times
import re
import matplotlib.pyplot as plt
import pandas as pd
from significance import compute_stimulus_coherence
import plotting
from tqdm import tqdm

wd = '/mnt/bronknas/Trials_on_Multiplexed_Stimuli/Oscillatory_Stimuli/4_sec/Data_Ready_To_Analyze/101023/Fish_H/'

# Load neural data
traces = load_traces(os.path.join(wd, 'merged', 'merged_raw.pkl'))

# Set sampling frequency
fs = 2.1802

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

# Make sure only has two columns
if stim_info.shape[1] > 2:
    stim_info = stim_info[:, :2]
stim_info_raw = stim_info
# Load combination mask
stim_info_single, stim_info_double, stim_info_triple, stim_id_map = load_combos_info(os.path.join(wd, '4secseq_5nm_combo_stim_binary.csv'))

# # Remove first 60 seconds and last 120 seconds
# mask = (stim_info[:, 0] > 60 * fs) & (stim_info[:, 0] < (traces.shape[1] / fs - 600))
# stim_info = stim_info[mask, :]

# Combine singles and doubles
stim_info = np.concatenate((stim_info_single, stim_info_double), axis=0)

# Remove likely outliers
mean_val = np.mean(traces)
std_val = np.std(traces)

# Find traces where there is any timepoint that is 5 standard deviations away from the mean
mask = np.zeros(traces.shape[0], dtype=bool)
for i in range(traces.shape[0]):
    if np.any(traces[i,:] > mean_val + 6 * std_val):
        mask[i] = False
    else:
        mask[i] = True

# Recompute zscore without outliers
traces_filtered = preprocess_traces(traces[mask], fs=fs, kernel_decay=3, bleaching_tau=100, zscore=False, filter=True)

# Split into trials
n_samples = 8
# n_stim is unique integers in stim_info[:, 1]
n_stim = len(np.unique(stim_info[:, 1]))

# Split into trials
# (# neurons, # samples, # trials, # stim)
traces_filtered_trials = split_stims(traces_filtered, stim_info, fs=fs, n_samples=n_samples, zscore=True)
traces_trials = split_stims(traces[mask], stim_info, fs=fs, n_samples=n_samples, zscore=True)


# Get highest responding for each class
neuron_selector = []
for i in range(n_stim):
    # Get nth percentile zscore
    threshold = np.percentile(np.nanmax((np.nanmean(traces_filtered_trials[:,:,:,i], axis=2)), axis=1), 99.977)
    neuron_selector.append(np.argwhere(np.nanmax((np.nanmean(traces_filtered_trials[:,:,:,i], axis=2)), axis=1) > threshold).squeeze())
neuron_selector = np.unique(np.concatenate(neuron_selector))

# View raw traces of selected neurons
from plotting import plot_traces
fig, ax = plot_traces(traces_filtered, neuron_selector[:20])
fig.show()

# Plot mosaic for validation
from plotting import plot_trial_mosaic
fig, ax = plot_trial_mosaic(traces_filtered_trials, neuron_selector[-20:], single_traces=True, fs=fs)
fig.show()

# from dimensionality import binarise_response
# binarised_responses = binarise_response(traces_filtered_trials, threshold=2.5)
#
# # Heatmap of binarised responses
# fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# ax.imshow(np.flipud(np.mean(binarised_responses[neuron_selector[-20:], :, :], axis=1)), aspect='auto', cmap='viridis')
# fig.show()

# Take max value over time as response
traces_filtered_trials_avg = np.max(traces_filtered_trials[neuron_selector[-100:], :, :, :], axis=1)

# Average over trials (# neurons, # stim)
traces_stim_avg = np.mean(traces_filtered_trials_avg, axis=1)

# Take responses to id = 1 and id = 2
response_1 = traces_stim_avg[:, 0]
response_2 = traces_stim_avg[:, 1]

# Take response to combination of 1 and 2 (==6)
response_6 = traces_stim_avg[:, 5]

# Project response 6 onto responses 1 and 2
# Find the angle between response 6 and response 1

# Project response 6 onto the plane spanned by responses 1 and 2
# Find the projection of response 6 onto response 1
proj_1 = np.dot(response_6, response_1) / np.dot(response_1, response_1) * response_1

# Find the projection of response 6 onto response 2
proj_2 = np.dot(response_6, response_2) / np.dot(response_2, response_2) * response_2

# Find the projection of response 6 onto the plane spanned by responses 1 and 2
proj_3 = proj_1 + proj_2

# Find the angle between response 6 and the projection of response 6 onto the plane spanned by responses 1 and 2
angle_3 = np.arccos(np.dot(response_6, proj_3) / (np.linalg.norm(response_6) * np.linalg.norm(proj_3)))

# In degrees
angle_3_deg = np.degrees(angle_3)