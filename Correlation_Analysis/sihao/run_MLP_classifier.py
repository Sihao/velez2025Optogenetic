import os
from plotting import plot_heatmap, plot_traces, plot_trial_mosaic
import matplotlib.pyplot as plt
import numpy as np
import re
from experiment_io import load_traces, load_positions, load_stim_info
from timeseries import preprocess_traces, avg_trials, split_trials, split_stims
from receptive_fields import compute_sta
from utils import get_rmap_idx

wd = '/mnt/bronknas/Singles/Data_Singles_To_Analyze/Data_singles_2/240823/Fish_H/merged'

# Load neural data
traces = load_traces(os.path.join(wd, 'merged_raw.pkl'))
# Load stimulus information
stim_info_path = '/mnt/bronknas/Singles/Data_Singles_To_Analyze/Data_singles_2/240823/Fish_H/NM_Stimul_setup_Combos_5_2023-08-24_18-51-07.mat'
stim_info = load_stim_info(stim_info_path)

# Set sampling frequency
fs = 2.1802

# Load neural data
traces = load_traces(os.path.join(wd, 'merged_raw.pkl'))

# Preprocess traces
traces_filtered = preprocess_traces(traces, fs=fs, kernel_decay=5, bleaching_tau=200, zscore=False, filter=True)

# Only keep neurons that are responsive to the stimulus
# Load granger causality values
# List granger files in directory above
wd = os.path.dirname(wd)
granger_files = [f for f in os.listdir(wd) if ('nm' in f) and ('granger' in f)]

# Key by nm\d{1} substring
granger_dict = {}
for f in granger_files:
    key = re.search('nm\d{1}', f).group()

    # Load granger causality values
    granger_causalities = np.load(os.path.join(wd, f))

    # Determine significant neurons
    alpha = 0.01
    significant_neurons = np.where(granger_causalities < alpha)[0]

    # Add to dictionary
    granger_dict[key] = significant_neurons

# Take union over all significant neurons
significant_neurons = np.unique(np.concatenate(list(granger_dict.values())))
significant_traces = traces_filtered[significant_neurons, :]

# Split into trials
traces_filtered_trials = split_stims(significant_traces, stim_info)

# Get number of neuromasts stimulated
n_stim = int(stim_info[:, 1].max())

# Average over response interval
traces_filtered_trials_avg = np.mean(traces_filtered_trials, axis=1)

# Unwrap final axis into penultimate axis
traces_stim_avg = np.concatenate(np.split(traces_filtered_trials_avg, n_stim, axis=2), axis=1).squeeze()

# zscore
traces_stim_avg = (traces_stim_avg - traces_stim_avg.mean(axis=0)) / traces_stim_avg.std(axis=0)

# Train MLP to predict neuromast being stimulated
# Split into train-test
from sklearn.model_selection import train_test_split

# Get labels
n_labels = traces_stim_avg.shape[1]
n_unique_labels = len(np.unique(stim_info[:, 1]))
labels = np.repeat(np.arange(n_unique_labels), n_labels / n_unique_labels)

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(traces_stim_avg.T, labels, test_size=0.2, random_state=42)

# Train MLP
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Train MLP
mlp = MLPClassifier(hidden_layer_sizes=(512,256), max_iter=10000, random_state=42, learning_rate_init=1e-3, shuffle=True)
mlp.fit(X_train, y_train)

# Predict
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion matrix')
