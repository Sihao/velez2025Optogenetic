import logging
import os
import re
import datetime
import numpy as np
import argparse


def main(input_path, output_path, overwrite=False):
    """Summarise confusion matrices from SVM results for all fish in specified directory

    Parameters
    ----------
    input_path : str
        Path to directory containing all experiment directories. Expected directory structure:
        input_path
        ├── YYYYMMDD
        │   ├── fish_A
        │   │   ├── SVM_results
        │   │   │   ├── lifetime_sparsity.npy
    output_path : str
        Path to directory to save the summarised confusion matrix.
    overwrite : bool, optional
        Overwrite existing summarised confusion matrix, by default False
    """
    # Start logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Create log file
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'logs'), exist_ok=True)
    handler = logging.FileHandler(os.path.join(output_path, 'log.txt'))
    logger.addHandler(handler)
    # Add date of current run
    logger.info(f'Run on {datetime.datetime.now()}')

    # Check if output_path/confusion_summary.npy exists
    if os.path.exists(os.path.join(output_path, 'lifetime_sparsity_summary.npy')) and not overwrite:
        print('Summarised lifetime sparsity already exists. Skipping...')
        logger.info('Summarised lifetime sparsity already exists. Skipping...')
        return

    # Regex pattern for 'Fish_*/'
    pattern = 'Fish_[^\/]*'

    # Traverse two depths
    def walk_limited_depth(top, max_depth):
        top = top.rstrip(os.path.sep)
        assert os.path.isdir(top)
        num_sep = top.count(os.path.sep)
        for root, dirs, files in os.walk(top):
            yield root, dirs, files
            current_depth = root.count(os.path.sep) - num_sep
            if current_depth >= max_depth:
                del dirs[:]

    fish_dirs = []
    for path in walk_limited_depth(input_path, 2):
        if re.search(pattern, path[0]):
            fish_dirs.append(path[0])

    # Load confusion matrices for all fish
    sparsities = []
    for fish_dir in fish_dirs:
        # Check if fish_dir/SVM_results/lifetime_sparsity.npy exists
        if os.path.exists(os.path.join(input_path, fish_dir, 'SVM_results', 'lifetime_sparsity.npy')):
            sparsities.append(np.load(os.path.join(input_path, fish_dir, 'SVM_results', 'lifetime_sparsity.npy')))
        else:
            print(f'Lifetime sparsity file not found for {fish_dir}. Skipping...')
            logger.info(f'Lifetime sparsity file not found for {fish_dir}. Skipping...')
            continue

    # Flatten list of arrays
    sparsities = np.array([item for sublist in sparsities for item in sublist])

    # Histogram
    bins = np.linspace(0, 1, 100)
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.histplot(sparsities, bins=bins, kde=True, ax=ax)

    # Save summarised confusion matrix
    np.save(os.path.join(output_path, 'lifetime_sparsity_summary.npy'), sparsities)
    print(f'Summarised confusion matrix saved to {os.path.join(output_path, "lifetime_sparsity_summary.npy")}')
    logger.info(f'Summarised confusion matrix saved to {os.path.join(output_path, "lifetime_sparsity_summary.npy")}')

    # Close log file
    handler.close()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--input_path', type=str, required=True)
    args.add_argument('--output_path', type=str, required=True)
    args.add_argument('--overwrite', type=bool, default=False)
    args = args.parse_args()

    main(args.input_path, args.output_path, args.overwrite)