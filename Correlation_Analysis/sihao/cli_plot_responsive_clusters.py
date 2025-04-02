from plotting import *
import numpy as np
import re
import os
from experiment_io import load_traces, load_positions
from timeseries import preprocess_traces


def main(input_dir, alpha=0.001):
    """Plot significant neurons in 3d space.

    Parameters
    ----------
    input_dir : str
        Path to input directory. Assumes directory structure to be:
        input_path
        ├── merged
        │   ├── merged_raw.pkl
        │   ├── merged_centroids.pkl
        ├── NM_Stimul_setup_Combos_5_YYYY-MM-DD_HH-MM-SS.mat (stimulation information file)
        ├── nm1_granger_causality.npy
        ├── nm2_granger_causality.npy
        ├── ...

    Returns
    -------
    None
        Saves plot as html file in input_path.
        Aborts if no granger causality files are found.
    """

    # Load neural data
    traces = load_traces(os.path.join(input_dir, 'merged', 'merged_raw.pkl'))

    # Preprocess traces
    traces_filtered = preprocess_traces(traces, fs=2.18, kernel_decay=5, bleaching_tau=200, zscore=False, filter=True)

    # Load centroids (merged_centroids.pkl)
    centroids_path = os.path.join(input_dir, 'merged', 'merged_centroids.pkl')
    positions = load_positions(centroids_path)

    # Load granger causality values
    # List granger files in directory
    granger_files = [f for f in os.listdir(input_dir) if ('nm' in f)]

    # Abort if no granger files found
    if len(granger_files) == 0:
        print('No granger causality files found. Skipping...')
        return

    # Key by nm\d{1} substring
    granger_dict = {}
    for f in granger_files:
        key = re.search('nm\d{1}', f).group()

        # Load granger causality values
        granger_causalities = np.load(os.path.join(input_dir, f))

        # Determine significant neurons
        alpha = alpha
        significant_neurons = np.where(granger_causalities < alpha)[0]

        # Add to dictionary
        granger_dict[key] = significant_neurons

    # Plot significant neurons
    # Plot all positions in 3d using plotly
    fig = plot_clusters_plotly(positions, traces_filtered, granger_dict)

    # Save figure as html (indicate experiment date and fish ID)
    exp_date = input_dir.split('/')[-3]
    fish_id = input_dir.split('/')[-2]

    fig.write_html(os.path.join(input_dir, f'granger_causality_{exp_date}_{fish_id}_alpha={alpha}.html'))
    print('Plot saved in ' + os.path.join(input_dir, f'granger_causality_{exp_date}_{fish_id}_alpha={alpha}.html'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualise granger causality results')
    parser.add_argument('--input_dir', type=str, help='Path to input directory')
    parser.add_argument('--alpha', type=float, default=0.001, help='Significance threshold for granger causality.')
    args = parser.parse_args()

    main(args.input_dir, args.alpha)