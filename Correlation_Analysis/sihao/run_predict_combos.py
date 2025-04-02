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
stim_info_single, stim_info_double, stim_info_triple = load_combos_info(os.path.join(wd, '4secseq_5nm_combo_stim_binary.csv'))

# # Remove first 60 seconds and last 120 seconds
# mask = (stim_info[:, 0] > 60 * fs) & (stim_info[:, 0] < (traces.shape[1] / fs - 600))
# stim_info = stim_info[mask, :]
stim_info = stim_info_double

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

from dimensionality import binarise_response
binarised_responses = binarise_response(traces_filtered_trials, threshold=2.5)

# Heatmap of binarised responses
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.imshow(np.flipud(np.mean(binarised_responses[neuron_selector[-20:], :, :], axis=1)), aspect='auto', cmap='viridis')
fig.show()

# Use binarised responses instead of maximal values in the response interval
traces_filtered_trials_avg = binarised_responses[neuron_selector, :, :]

# Unwrap final axis into penultimate axis, do explicitly to ensure correct label
n_stim = traces_filtered_trials_avg.shape[-1]
n_neurons = traces_filtered_trials_avg.shape[0]
n_trials = traces_filtered_trials_avg.shape[1]
traces_stim_avg = np.zeros((n_neurons, n_trials * n_stim))
labels = np.zeros(n_trials * n_stim)
for stim in range(n_stim):
    traces_stim_avg[:, stim * n_trials:(stim + 1) * n_trials] = traces_filtered_trials_avg[:, :, stim]
    labels[stim * n_trials:(stim + 1) * n_trials] = stim + 1

traces_stim_avg = np.concatenate(np.split(traces_filtered_trials_avg, n_stim, axis=2), axis=1).squeeze()

#
# # zscore
traces_stim_avg = (traces_stim_avg - traces_stim_avg.mean(axis=0)) / traces_stim_avg.std(axis=0)

from dimensionality import classify_stims, classify_stims_mlp
from sklearn.metrics import accuracy_score


# Create dataframe to hold results
results = pd.DataFrame(columns=['repetition', 'depth', 'scaled_accuracy'])

# Generate 100 random random seeds
np.random.seed(42)
seeds = np.random.randint(0, 10000, 3)
for i, seed in tqdm(enumerate(seeds)):
    for depth in [1, 2, 3]:
        clf, X_test, y_test, y_pred = classify_stims_mlp(traces_stim_avg, labels, depth=depth, random_state=seed)
        accuracy = accuracy_score(y_test, y_pred)
        scaled_accuracy = (accuracy - 1 / n_stim) / (1 - 1 / n_stim)
        new_data = pd.DataFrame([{'repetition': i, 'depth': depth, 'scaled_accuracy':scaled_accuracy}])
        results = pd.concat([results, new_data], ignore_index=True)

# Show violinplot of accuracy at each depth across repetitions
import seaborn as sns
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.swarmplot(x='depth', y='scaled_accuracy', data=results, ax=ax)
sns.violinplot(x='depth', y='scaled_accuracy', data=results, ax=ax, inner=None, color='0.8')

fig.show()

# Paired t-test between depth 1 and 2
from scipy.stats import ttest_rel
t, p = ttest_rel(results[results['depth'] == 1]['scaled_accuracy'], results[results['depth'] == 2]['scaled_accuracy'])
print(f'Paired t-test between depth 1 and 2: t = {t}, p = {p}')


# Confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_pred, normalize='true')

# Get weights
weights = clf.coefs_[0]

# PCA reduce to 3 dimensions
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
weights_pca = pca.fit_transform(weights.T)

# Plot 3D PCA
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(weights_pca[:, 0], weights_pca[:, 1], weights_pca[:, 2], c=np.arange(weights_pca.shape[0]), cmap='viridis')
fig.show()

# Compute hoyer sparsity
from utils import hoyer_sparsity

population_sparsity = hoyer_sparsity(weights)
lifetime_sparsity = hoyer_sparsity(weights.T)

from plotting import plot_svm_results
fig, ax = plot_svm_results(confusion, lifetime_sparsity, population_sparsity, weights, subplot_labels=['B', 'C', 'D', 'E'])
fig.show()