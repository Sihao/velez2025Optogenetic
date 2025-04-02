import argparse
import logging
import os
import numpy as np
import datetime
import re
import pickle
import pandas as pd
from experiment_io import load_traces, load_stim_info, get_neuron_region
from timeseries import preprocess_traces, split_stims, compute_n_samples
from utils import hoyer_sparsity
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def main(input_path, fs=2.1802, overwrite_df=False):
    """Loads the region information from the atlas registration into a pandas DataFrame and saves it along with the
    sparsity of both the raw response and of the weights of the SVM classifier. Assumes SVM has been trained previously.

    Parameters
    ----------
    input_path : str
        Path to input file containing neural data. Assumes directory structure to be:
        input_path
        ├── merged
        │   ├── merged_raw.pkl
        ├── NM_Stimul_setup_Combos_5_YYYY-MM-DD_HH-MM-SS.mat (stimulation information file)
        ├── Regions
        ├── SVM_results
            ├── svm.pkl
    fs : float
        Sampling frequency of neural data.
    overwrite_df : bool
        Whether to overwrite the DataFrame containing the region information.

    """
    # Start logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    output_path = os.path.join(input_path)

    # Create log file
    os.makedirs(output_path, exist_ok=True)
    handler = logging.FileHandler(os.path.join(output_path, 'region_df_log.txt'))
    logger.addHandler(handler)
    # Add date of current run
    logger.info(f'Run on {datetime.datetime.now()}')

    # Check if dataframe already exists
    if os.path.exists(os.path.join(input_path, 'region_df.pkl')) and not overwrite_df:
        print('Region DataFrame already exists. Skipping...')
        logger.info('Region DataFrame already exists. Skipping...')
        return

    # Check if SVM results already exist
    if not os.path.exists(os.path.join(input_path, 'SVM_results', 'svm.pkl')):
        print('SVM results not found. Skipping...')
        logger.info('SVM results not found. Skipping...')
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

    # Take mean across trials (# neurons, # samples, # trials, # stim) - > (# neurons, # samples, # stim)
    traces_filtered_trials_avg = np.mean(traces_filtered_trials, axis=2)

    # Compute raw sparsity for each neuron
    raw_sparsity = hoyer_sparsity(np.max(traces_filtered_trials_avg, axis=1))

    # Load SVM
    svm_path = os.path.join(input_path, 'SVM_results', 'svm.pkl')
    with open(svm_path, 'rb') as f:
        clf = pickle.load(f)
    logger.info(f'Loaded SVM from {svm_path}')

    # Load indices used for SVM
    indices_path = os.path.join(input_path, 'SVM_results', 'svm_neurons.npy')
    svm_neurons_idx = np.load(indices_path)

    # Get weights
    weights = clf.coef_

    # Compute hoyer sparsity of weights
    weight_sparsity = hoyer_sparsity(weights.T)

    # Populate DataFrame
    # Create empty DataFrame
    region_df = pd.DataFrame(columns=['neuron_idx',
                                      'raw_sparsity', 'weight_sparsity', 'svm_neuron', 'fish_id'])

    region_df['neuron_idx'] = np.arange(traces.shape[0])
    region_df['fish_id'] = '_'.join(input_path.split('/')[-2:])
    region_df['raw_sparsity'] = raw_sparsity
    region_df['svm_neuron'] = False

    # Fill in weight sparsity based on SVM neurons
    region_df.loc[svm_neurons_idx, 'weight_sparsity'] = weight_sparsity
    region_df.loc[svm_neurons_idx, 'svm_neuron'] = True

    # Load registration region information
    registration_df = get_neuron_region(input_path)

    # Merge dataframe based on 'idx' column, merge overlapping columns
    # Find all rows where neuron_idx matches registration_df.neuron_idx
    registration_df = registration_df.sort_values(by='neuron_idx')
    region_df = pd.merge(region_df, registration_df, on='neuron_idx', how='left')

    # Save DataFrame
    region_df.to_pickle(os.path.join(input_path, 'region_df.pkl'))
    logger.info(f'Saved region DataFrame to {os.path.join(input_path, "region_df.pkl")}')

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
    main(os.path.join(args.input_path), fs=args.fs, overwrite_df=args.overwrite)