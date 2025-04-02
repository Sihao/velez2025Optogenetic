import os
import numpy as np
from experiment_io import load_traces, load_stim_info
from timeseries import preprocess_traces, split_stims, create_timeseries, get_neuromast_stim_times
import plotting
wd = '/mnt/bronknas/Singles/Data_random_to_analyze/02_randomized/Fish_C_randomized_2/merged'

# Load neural data
traces = load_traces(os.path.join(wd, 'merged_raw.pkl'))
# Load stimulus information
stim_info_path = '/mnt/bronknas/Singles/Data_random_to_analyze/02_randomized/Fish_C_randomized_2/NM_Stimul_setup_Combos_5_2024-02-16_20-03-53.mat'
stim_info = load_stim_info(stim_info_path)

# Set sampling frequency
fs = 2.1802

# Load neural data
traces = load_traces(os.path.join(wd, 'merged_raw.pkl'))



# Preprocess traces
traces_filtered = preprocess_traces(traces, fs=fs, kernel_decay=3, bleaching_tau=100, zscore=False, filter=True)


# Split into trials
# Determine number of samples per trial
n_stim = int(stim_info[:, 1].max())
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

traces_filtered_trials = split_stims(traces_filtered, stim_info, fs=fs, n_samples=n_samples)

# Get number of neuromasts stimulated
n_stim = int(stim_info[:, 1].max())

# Sanity check plot
from plotting import plot_trial_mosaic
peak_sort = np.argsort(np.max((np.mean(traces_filtered_trials, axis=2)), axis=1), axis=0)

# Pick two neurons for each stim
idx_to_plot = []
for i in range(n_stim):
    idx_to_plot.append(peak_sort[-2:, i])
# Make 1-D
idx_to_plot = np.concatenate(idx_to_plot)
# Take unique idx
idx_to_plot = np.unique(idx_to_plot)
fig, ax = plot_trial_mosaic(np.mean(traces_filtered_trials, axis=2), idx_to_plot, fs=fs)
fig.show()

# Get highest responding for each class
neuron_selector = []
for i in range(n_stim):
    # Get 95th percentile zscore
    threshold = np.percentile(np.max((np.mean(traces_filtered_trials[:,:,:,i], axis=2)), axis=1), 99.977)
    neuron_selector.append(np.argwhere(np.max((np.mean(traces_filtered_trials[:,:,:,i], axis=2)), axis=1) > threshold).squeeze())
neuron_selector = np.unique(np.concatenate(neuron_selector))

# Average over response interval
traces_filtered_trials_avg = np.max(traces_filtered_trials[neuron_selector, :, :, :], axis=1)

# Unwrap final axis into penultimate axis
traces_stim_avg = np.concatenate(np.split(traces_filtered_trials_avg, n_stim, axis=2), axis=1).squeeze()

# zscore
traces_stim_avg = (traces_stim_avg - traces_stim_avg.mean(axis=0)) / traces_stim_avg.std(axis=0)

# Get labels
n_labels = traces_stim_avg.shape[1]
n_unique_labels = len(np.unique(stim_info[:, 1]))
labels = np.repeat(np.arange(n_unique_labels), n_labels / n_unique_labels)

# Compute linear discriminant between two classes (stim IDs)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(traces_stim_avg.T, labels, test_size=0.2, random_state=42)

# Split train into train-validation and k-fold cross validation
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import svm

cross_val_weights = []
n_class_members = np.min(np.unique(y_train, return_counts=True)[1])
skf = StratifiedShuffleSplit(n_splits=n_class_members, random_state=42)
for train_index, val_index in skf.split(X_train, y_train):
    X_train_train, X_train_val = X_train[train_index], X_train[val_index]
    y_train_train, y_train_val = y_train[train_index], y_train[val_index]

    # Train SVM
    clf = svm.LinearSVC(dual='auto', penalty='l1', C=0.8,
                        random_state=42, max_iter=100000, tol=1e-6)
    clf.fit(X_train_train, y_train_train)
    weights = clf.coef_
    cross_val_weights.append(weights)

# Average weights
cross_val_weights = np.mean(np.array(cross_val_weights), axis=0)

# Assign weights to clf
clf.coef_ = cross_val_weights

# Predict
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# Confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_pred, normalize='pred')

# Get weights
weights = clf.coef_

# Compute hoyer sparsity
from utils import hoyer_sparsity

population_sparsity = hoyer_sparsity(weights)
lifetime_sparsity = hoyer_sparsity(weights.T)

from plotting import plot_svm_results
fig, ax = plot_svm_results(confusion, lifetime_sparsity, population_sparsity, weights, subplot_labels=['B', 'C', 'D', 'E'])
fig.show()