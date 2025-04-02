import argparse
import logging
import os
import numpy as np
import datetime
import re
import pickle
from experiment_io import load_traces, load_stim_info
from timeseries import preprocess_traces, split_stims, compute_n_samples
from sklearn.metrics import accuracy_score, confusion_matrix
from utils import hoyer_sparsity
from plotting import plot_svm_results, plot_trial_mosaic
from dimensionality import classify_stims_mlp
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def main(input_path, fs=2.1802, overwrite=True):
    """Train an SVM classifier for a given fish to predict the stimulus ID from neural data.
    Outputs the confusion matrix and the accuracy of the classifier alongside the weights of the neurons,
    the sparsity of the weights (lifetime and population)

    Parameters
    ----------
    input_path : str
        Path to input file containing neural data. Assumes directory structure to be:
        input_path
        ├── merged
        │   ├── merged_raw.pkl
        ├── NM_Stimul_setup_Combos_5_YYYY-MM-DD_HH-MM-SS.mat (stimulation information file)
    fs : float
        Sampling frequency of neural data.
    """
    # Start logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    output_path = os.path.join(input_path, 'MLP_results')

    # Create log file
    os.makedirs(output_path, exist_ok=True)
    handler = logging.FileHandler(os.path.join(output_path, 'log.txt'))
    logger.addHandler(handler)
    # Add date of current run
    logger.info(f'Run on {datetime.datetime.now()}')

    # Check if output_path/confusion.npy exists
    if os.path.exists(os.path.join(output_path, 'confusion.npy')) and not overwrite:
        print('SVM results already exist. Skipping...')
        logger.info('SVM results already exist. Skipping...')
        return

    # Load stimulus information
    # Regex find NM_Stimul_setup_Combos_5_YYYY-MM-DD_HH-MM-SS.mat
    pattern = 'NM_Stimul_setup_'
    dir_content = os.listdir(input_path)

    file_found = False
    for file in dir_content:
        if re.search(pattern, file):
            stim_info_path = os.path.join(input_path, file)
            file_found = True
            break

    if not file_found:
        print('Stimulus information file not found. Skipping...')
        logger.info('Stimulus information file not found. Skipping...')
        # Skip this fish
        return

    stim_info = load_stim_info(stim_info_path)

    # Load neural data
    traces = load_traces(os.path.join(input_path, 'merged', 'merged_raw.pkl'))
    logger.info(f'Loaded neural data from {os.path.join(input_path, "merged", "merged_raw.pkl")}')

    # Filter traces
    traces_filtered = preprocess_traces(traces, fs=fs, kernel_decay=5, bleaching_tau=400, zscore=False, filter=True)

    # Split into trials (# neurons, # samples, # trials, # stim)
    n_samples = compute_n_samples(stim_info, traces.shape[1] / fs, fs)
    traces_filtered_trials = split_stims(traces_filtered, stim_info, fs=fs, n_samples=n_samples)

    # Find neurons which respond to a given stimulus pattern
    # Get number of stimuli in experiment
    n_stim = len(np.unique(stim_info[:, 1]))

    # Get highest responding for each class
    neuron_selector = []
    for i in range(n_stim):
        # Get nth percentile zscore
        threshold = np.percentile(np.max((np.mean(traces_filtered_trials[:, :, :, i], axis=2)), axis=1), 99.977)
        neuron_selector.append(
            np.argwhere(np.max((np.mean(traces_filtered_trials[:, :, :, i], axis=2)), axis=1) > threshold).squeeze())
    neuron_selector = np.unique(np.concatenate(neuron_selector))

    # Save selected neuron indices
    np.save(os.path.join(output_path, 'mlp_neurons.npy'), neuron_selector)

    # Take maximum value as response (# neurons, # trials, # stim)
    traces_filtered_trials_avg = np.max(traces_filtered_trials[neuron_selector, :, :, :], axis=1)

    # Unwrap final axis into penultimate axis (# neurons, # trials * # stim)
    traces_stim_avg = np.concatenate(np.split(traces_filtered_trials_avg, n_stim, axis=2), axis=1).squeeze()

    # Take the second power of the traces
    traces_stim_avg_sqr = np.power(traces_stim_avg, 2)

    # zscore
    traces_stim_avg = (traces_stim_avg - traces_stim_avg.mean(axis=0)) / traces_stim_avg.std(axis=0)
    traces_stim_avg_sqr = (traces_stim_avg_sqr - traces_stim_avg_sqr.mean(axis=0)) / traces_stim_avg_sqr.std(axis=0)

    # Get labels
    n_labels = traces_stim_avg.shape[1]
    n_unique_labels = len(np.unique(stim_info[:, 1]))
    labels = np.repeat(np.arange(n_unique_labels), n_labels / n_unique_labels)

    # Run SVM decoder
    clf, X_test, y_test, y_pred = classify_stims_mlp(traces_stim_avg_sqr, labels)

    # Save MLP and test set for future use
    with open(os.path.join(output_path, 'mlp.pkl'), 'wb') as f:
        pickle.dump(clf, f)
    np.save(os.path.join(output_path, 'X_test.npy'), X_test)
    np.save(os.path.join(output_path, 'y_test.npy'), y_test)

    # Predict
    accuracy = accuracy_score(y_test, y_pred)

    # Scale accuracy with number of classes
    scaled_accuracy = (accuracy - 1 / n_unique_labels) / (1 - 1 / n_unique_labels)
    logger.info(f'Accuracy: {accuracy}')
    logger.info(f'Scaled accuracy: {scaled_accuracy}')
    print(f'Accuracy: {accuracy}')
    print(f'Scaled accuracy: {scaled_accuracy}')

    # Confusion matrix
    confusion = confusion_matrix(y_test, y_pred)

    # Get weights
    weights = clf.coefs_

    # Save results
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'weights.pkl'), 'wb') as f:
        pickle.dump(weights, f)
    np.save(os.path.join(output_path, 'confusion.npy'), confusion)
    np.save(os.path.join(output_path, 'accuracy.npy'), accuracy)
    np.save(os.path.join(output_path, 'scaled_accuracy.npy'), scaled_accuracy)
    logger.info(f'Saved results to {output_path}')

    # Save trace mosaic
    peak_sort = np.argsort(np.max((np.mean(traces_filtered_trials, axis=2)), axis=1), axis=0)
    # Pick two neurons for each stim
    idx_to_plot = []
    for i in range(n_stim):
        idx_to_plot.append(peak_sort[-2:, i])
    # Make 1-D
    idx_to_plot = np.concatenate(idx_to_plot)
    # Take unique idx
    idx_to_plot = np.unique(idx_to_plot)

    fig, ax = plot_trial_mosaic(traces_filtered_trials, idx_to_plot, fs=fs, single_traces=True)
    fig.savefig(os.path.join(output_path, 'trace_mosaic.svg'))

    # Close handler
    handler.close()


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='Path to input file containing neural data. Assumes directory '
                                                     'structure to be: input_path ├── avgs │ ├── merged │ │ ├── '
                                                     'merged_raw.pkl │ ├── NM_Stimul_setup_Combos_5_YYYY-MM-DD_HH-MM-SS'
                                                     '.mat (stimulation information file)')
    parser.add_argument('--fs', type=float, default=2.1802, help='Sampling frequency of neural data.')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    # Run main function
    main(os.path.join(args.input_path), fs=args.fs, overwrite=args.overwrite)