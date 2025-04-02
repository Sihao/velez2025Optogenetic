import os
import re
import argparse
import logging
from experiment_io import load_traces, load_stim_info
from significance import compute_granger_causality
from timeseries import preprocess_traces
import numpy as np

def main(input_path, n_lags=1, fs=2.18):
    """Compute the Granger for all stimulation IDs (singles) for a given fish.

    Saves the Granger causality values for each neuron in a .npy file for each stimulation ID in the directory defined
    by input_path. Assumes Stimulation information file is present.

    Skips if granger causality file already exists and if stimulation information file is not found.

    Parameters
    ----------
    input_path : str
        Path to input file containing neural data. Assumes directory structure to be:
        input_path
        ├── merged
        │   ├── merged_raw.pkl
        ├── NM_Stimul_setup_Combos_5_YYYY-MM-DD_HH-MM-SS.mat (stimulation information file)
    n_lags : int
        Number of lags to use for Granger causality computation.
    fs : float
        Sampling frequency of neural data.
    """
    # Start logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

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
    traces_filtered = preprocess_traces(traces, fs=fs, kernel_decay=3)

    # Find neurons which respond to a given stimulus pattern
    # Get number of stimuli in experiment
    n_stim = len(np.unique(stim_info[:, 1]))

    # Compute Granger causality for all neurons with respect to a given stimulus pattern
    for stim_id in range(1, n_stim + 1):  # Stimulus IDs start at 1
        exp_date = input_path.split('/')[-2]
        exp_individual = input_path.split('/')[-1]
        save_name = os.path.join(input_path,  'granger_' + exp_date + '_' +
                                 exp_individual + '_nm' + str(stim_id) +
                                 '_lag' + str(n_lags) + '.npy')
        # Skip if file already exists
        if os.path.exists(save_name):
            print(f'File {save_name} already exists. Skipping...')
            logger.info(f'File {save_name} already exists. Skipping...')
            continue

        _ = compute_granger_causality(traces_filtered, stim_info, stim_id,
                                                              fs=fs, n_lags=n_lags,
                                                              use_saved=False, save_name=save_name)

        logger.info(f'Computed Granger causality for stimulation ID {stim_id} and saved in {save_name}')


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='Path to input file containing neural data. Assumes directory '
                                                     'structure to be: input_path ├── avgs │ ├── merged │ │ ├── '
                                                     'merged_raw.pkl │ ├── NM_Stimul_setup_Combos_5_YYYY-MM-DD_HH-MM-SS'
                                                     '.mat (stimulation information file)')
    parser.add_argument('--n_lags', type=int, default=1, help='Number of lags to use for Granger causality computation.')
    parser.add_argument('--fs', type=float, default=2.18, help='Sampling frequency of neural data.')
    args = parser.parse_args()

    # Run main function
    main(os.path.join(args.input_path), n_lags=args.n_lags, fs=args.fs)