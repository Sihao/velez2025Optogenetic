from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from timeseries import preprocess_traces, avg_trials, split_stims
from utils import hoyer_sparsity


def run_SVM_decoder(traces, stim_info, fs):
    """Run SVM decoder on traces
    Picks only neurons that are responsive to the stimulus by taking the top 0.3% of the zscored average response for a
    given neuron to a given stimulus. Then, the traces are zscored and split into train-test sets. The SVM decoder is
    trained on the train set and tested on the test set.

    Args:
        traces (np.ndarray): 2D array of traces, shape (n_neurons, n_timepoints)

    Returns:
        accuracy: float
            Accuracy of the SVM decoder
        confusion: np.ndarray
            Confusion matrix
        weights: np.ndarray
            Weights of the SVM decoder
        sparsity: float
            Hoyer sparsity of the weights
    """

    # Split into trials for each stimulus
    traces_trials = split_stims(traces, stim_info)

    # Get number of neuromasts stimulated
    n_stim = len(np.unique(stim_info[:, 1]))

    # Sanity check plot
    from plotting import plot_trial_mosaic
    peak_sort = np.argsort(np.max((np.mean(traces_trials, axis=2)), axis=1), axis=0)

    fig, ax = plot_trial_mosaic(np.mean(traces_trials, axis=2), peak_sort[-10:, -3], fs=fs)
    fig.show()

    # Get highest responding for each class
    neuron_selector = []
    for i in range(n_stim):
        # Get 99.7th percentile zscore (3 sigma)
        threshold = np.percentile(np.max((np.mean(traces_trials[:, :, :, i], axis=2)), axis=1), 99.7)
        neuron_selector.append(
            np.argwhere(np.max((np.mean(traces_trials[:, :, :, i], axis=2)), axis=1) > threshold).squeeze())
    neuron_selector = np.unique(np.concatenate(neuron_selector))
    # Average over response interval
    traces_filtered_trials_avg = np.max(traces_trials[neuron_selector, :], axis=1)

    # Unwrap final axis into penultimate axis
    traces_stim_avg = np.concatenate(np.split(traces_filtered_trials_avg, n_stim, axis=2), axis=1).squeeze()

    # zscore
    traces_stim_avg = (traces_stim_avg - traces_stim_avg.mean(axis=0)) / traces_stim_avg.std(axis=0)

    # Get labels
    n_labels = traces_stim_avg.shape[1]
    n_unique_labels = len(np.unique(stim_info[:, 1]))
    labels = np.repeat(np.arange(n_unique_labels), n_labels / n_unique_labels)

    # Split into train-test
    X_train, X_test, y_train, y_test = train_test_split(traces_stim_avg.T, labels, test_size=0.2, random_state=42)

    # Train SVM
    clf = svm.LinearSVC(dual='auto', penalty='l1', random_state=42, max_iter=10000, tol=1e-6, C=1)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Confusion matrix
    confusion = confusion_matrix(y_test, y_pred)

    # Sparsity
    weights = clf.coef_
    sparsity = hoyer_sparsity(weights)

    return accuracy, confusion, weights, sparsity
