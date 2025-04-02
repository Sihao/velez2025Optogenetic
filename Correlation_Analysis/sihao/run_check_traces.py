import os
import numpy as np
from experiment_io import load_traces, load_stim_info, get_neuromast_identities
from timeseries import preprocess_traces, split_stims, create_timeseries, get_neuromast_stim_times
import re
import matplotlib.pyplot as plt
import pandas as pd
from significance import compute_stimulus_coherence

wd = '/mnt/bronknas/rotation_students/Joey/neural_data/240823/Fish_A/'

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



# # Remove first 60 seconds and last 120 seconds
# mask = (stim_info[:, 0] > 60 * fs) & (stim_info[:, 0] < (traces.shape[1] / fs - 600))
# stim_info = stim_info[mask, :]

# Preprocess traces
traces_filtered = preprocess_traces(traces, fs=fs, kernel_decay=3, bleaching_tau=100, zscore=False, filter=True)

# Split into trials
# Determine number of samples per trial
from timeseries import compute_n_samples
n_samples = compute_n_samples(stim_info, duration=traces.shape[1] / fs, fs=fs)
n_stim = int(stim_info[:, 1].max())

# Split into trials
# (# neurons, # samples, # trials, # stim)
traces_filtered_trials = split_stims(traces_filtered, stim_info, fs=fs, n_samples=n_samples)
traces_trials = split_stims(traces, stim_info, fs=fs, n_samples=n_samples)

# Compute coherence for all neurons with respect to a given stimulus pattern
coherent_neurons = []
for i in np.arange(1, n_stim + 1):
    coherences = compute_stimulus_coherence(traces, stim_info, i, fs=fs, kernel_decay=3, method='bins')
    # Get nth percentile
    threshold = np.percentile(coherences, 99.97)
    # Get indices of neurons which respond to the stimulus pattern
    coherent_neurons.append(np.argwhere(coherences > threshold).squeeze())

# Get unique neurons
coherent_neurons = np.unique(np.concatenate(coherent_neurons))
coherent_sort = np.argsort(coherences[coherent_neurons])

# Plot example trace
from plotting import plot_traces
fig, ax = plot_traces(traces, coherent_neurons[coherent_sort[-2:]])
fig.show()

# Plot example trace with stimulus overlay
from plotting import plot_stimulus_overlay
fig, ax = plot_stimulus_overlay(traces, stim_info, fs, neuron_idx=coherent_neurons[coherent_sort[-1:]], stim_id=1)
fig.show()

# Plot mosaic of coherent neurons
from plotting import plot_trial_mosaic
fig, ax = plot_trial_mosaic(traces_trials, coherent_neurons[coherent_sort[-20:]], fs=fs, single_traces=True)
fig.show()

# Get highest responding for each class
neuron_selector = []
for i in range(n_stim):
    # Get 95th percentile zscore
    threshold = np.percentile(np.nanmax((np.nanmean(traces_filtered_trials[:,:,:,i], axis=2)), axis=1), 99.977)
    neuron_selector.append(np.argwhere(np.nanmax((np.nanmean(traces_filtered_trials[:,:,:,i], axis=2)), axis=1) > threshold).squeeze())
neuron_selector = np.unique(np.concatenate(neuron_selector))
# Take union of coherent neurons and highest responding neurons
# neuron_selector = np.unique(np.concatenate([neuron_selector, coherent_neurons]))
# Difference between two sets
selector_diff = np.setdiff1d(neuron_selector, np.intersect1d(neuron_selector, coherent_neurons))

# Average over response interval
traces_filtered_trials_avg = np.max(traces_filtered_trials[neuron_selector, :, :, :], axis=1)
traces_trials_avg = np.max(traces_trials[neuron_selector, :, :, :], axis=1)

# Unwrap final axis into penultimate axis
traces_stim_avg = np.concatenate(np.split(traces_filtered_trials_avg, n_stim, axis=2), axis=1).squeeze()
traces_stim_raw_avg = np.concatenate(np.split(traces_trials_avg, n_stim, axis=2), axis=1).squeeze()

# Apply non-linear transformation
traces_stim_avg = np.square(traces_stim_avg)

# zscore
traces_stim_avg = (traces_stim_avg - traces_stim_avg.mean(axis=0)) / traces_stim_avg.std(axis=0)
traces_stim_raw_avg = (traces_stim_raw_avg - traces_stim_raw_avg.mean(axis=0)) / traces_stim_raw_avg.std(axis=0)


# Get labels
n_labels = traces_stim_avg.shape[1]
n_unique_labels = len(np.unique(stim_info[:, 1]))
labels = np.repeat(np.arange(n_unique_labels), n_labels / n_unique_labels)

from dimensionality import classify_stims
from sklearn.metrics import accuracy_score


# Train classifier
clf, X_test, y_test, y_pred = classify_stims(traces_stim_avg, labels)


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
# Scaled accuracy
scaled_accuracy = (accuracy - 1 / n_unique_labels) / (1 - 1 / n_unique_labels)
print(f'Scaled accuracy: {scaled_accuracy}')

# Confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_pred, normalize='true')

# Get weights
weights = clf.coef_

# Compute hoyer sparsity
from utils import hoyer_sparsity

population_sparsity = hoyer_sparsity(weights)
lifetime_sparsity = hoyer_sparsity(weights.T)

from plotting import plot_svm_results
fig, ax = plot_svm_results(confusion, lifetime_sparsity, population_sparsity, weights, subplot_labels=['B', 'C', 'D', 'E'])
fig.show()

