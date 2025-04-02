import os
import re
import argparse
import logging
import numpy as np
from experiment_io import load_traces, load_stim_info
from significance import compute_granger_causality
from timeseries import preprocess_traces


def main(input_dir, trim=30, n_lags=1, fs=2.18):
    """Compute the Granger for all stimulation IDs (singles) for a given fish.
    !!! Trims the first 30 seconds from the recorded traces and the stimulus timeseries. !!!

    Saves the Granger causality values for each neuron in a .npy file for each stimulation ID in the directory defined
    by input_path. Assumes Stimulation information file is present.

    Skips if granger causality file already exists and if stimulation information file is not found.

    Parameters
    ----------
    input_dir : str
        Path to input file containing neural data. Assumes directory structure to be:
        input_path
        ├── merged
        │   ├── merged_raw.pkl
        ├── NM_Stimul_setup_Combos_5_YYYY-MM-DD_HH-MM-SS.mat (stimulation information file)
    trim : float
        Number of seconds to trim from the beginning of the recording.
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
    pattern = 'NM_Stimul_setup_Combos_5'
    dir_content = os.listdir(input_dir)

    file_found = False
    for file in dir_content:
        if re.search(pattern, file):
            stim_info_path = os.path.join(input_dir, file)
            file_found = True
            break

    if not file_found:
        print('Stimulus information file not found. Skipping...')
        logger.info('Stimulus information file not found. Skipping...')
        # Skip this fish
        return

    stim_info = load_stim_info(stim_info_path)

    # Load neural data
    traces = load_traces(os.path.join(input_dir, 'merged', 'merged_raw.pkl'))
    logger.info(f'Loaded neural data from {os.path.join(input_dir, "merged", "merged_raw.pkl")}')

    # Note original duration for stimulus timeseries
    orig_duration = traces.shape[1] / fs

    # Trim traces
    n_samples_to_trim = int(trim * fs)
    traces = traces[:, n_samples_to_trim:]

    # Remove times < trim from stim_info (shouldn't be necessary, but just in case)
    stim_info = stim_info[stim_info[:, 0] > trim, :]

    # Filter traces
    traces_filtered = preprocess_traces(traces, fs=fs, kernel_decay=3)

    # Find neurons which respond to a given stimulus pattern
    # Get number of stimuli in experiment
    n_stim = len(np.unique(stim_info[:, 1]))

    # Compute Granger causality for all neurons with respect to a given stimulus pattern
    for stim_id in range(1, n_stim + 1):  # Stimulus IDs start at 1
        exp_date = input_dir.split('/')[-2]
        exp_individual = input_dir.split('/')[-1]
        save_name = os.path.join(input_dir,  'trimmed_granger_' + exp_date + '_' +
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
    parser.add_argument('--input_dir', type=str, help='Path to input file containing neural data. Assumes directory '
                                                     'structure to be: input_path ├── avgs │ ├── merged │ │ ├── '
                                                     'merged_raw.pkl │ ├── NM_Stimul_setup_Combos_5_YYYY-MM-DD_HH-MM-SS'
                                                     '.mat (stimulation information file)')
    parser.add_argument('--trim', type=float, default=30, help='Number of seconds to trim from the beginning of the ')
    parser.add_argument('--n_lags', type=int, default=1, help='Number of lags to use for Granger causality computation.')
    parser.add_argument('--fs', type=float, default=2.18, help='Sampling frequency of neural data.')
    args = parser.parse_args()

    # Run main function
    main(os.path.join(args.input_dir), trim=args.trim, n_lags=args.n_lags, fs=args.fs)