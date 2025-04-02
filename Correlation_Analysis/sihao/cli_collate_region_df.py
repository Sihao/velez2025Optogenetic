import logging
import os
import re
import datetime
import numpy as np
import argparse
import pandas as pd


def main(input_path, output_path, overwrite=False):
    """Collage dataframes from all fish into a single dataframe.

    Parameters
    ----------
    input_path : str
        Path to directory containing all experiment directories. Expected directory structure:
        input_path
        ├── YYYYMMDD
        │   ├── fish_A
        │   │   ├── region_df.pkl
    output_path : str
        Path to directory to save the collated dataframe.
    overwrite : bool, optional
        Overwrite existing collated dataframe, by default False
    """
    # Start logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Create log file
    os.makedirs(output_path, exist_ok=True)
    handler = logging.FileHandler(os.path.join(output_path, 'collate_log.txt'))
    logger.addHandler(handler)
    # Add date of current run
    logger.info(f'Run on {datetime.datetime.now()}')

    # Check if output_path/collated_region_df.pkl exists
    if os.path.exists(os.path.join(output_path, 'collated_region_df.pkl')) and not overwrite:
        print('Collated region DataFrame already exists. Skipping...')
        logger.info('Collated region DataFrame already exists. Skipping...')
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

    # Load dataframe from each fish
    for fish_dir in fish_dirs:
        if os.path.exists(os.path.join(fish_dir, 'region_df.pkl')):
            print(f'Loading region dataframe from {fish_dir}...')
            logger.info(f'Loading region dataframe from {fish_dir}...')
            if 'region_df' not in locals():
                # Load dataframe
                region_df = np.load(os.path.join(fish_dir, 'region_df.pkl'), allow_pickle=True)
            else:
                # Concatenate dataframes
                region_df = pd.concat([region_df, np.load(os.path.join(fish_dir, 'region_df.pkl'), allow_pickle=True)])
        else:
            print(f'Region dataframe not found in {fish_dir}. Skipping...')
            logger.info(f'Region dataframe not found in {fish_dir}. Skipping...')
            continue

    # Save collated dataframe
    region_df.to_pickle(os.path.join(output_path, 'collated_region_df.pkl'))
    print(f'Collated region dataframe saved to {os.path.join(output_path, "collated_region_df.pkl")}')
    logger.info(f'Collated region dataframe saved to {os.path.join(output_path, "collated_region_df.pkl")}')

    # Close log file
    handler.close()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--input_path', type=str, required=True)
    args.add_argument('--output_path', type=str, required=True)
    args.add_argument('--overwrite', action='store_true', default=False)
    args = args.parse_args()

    main(args.input_path, args.output_path, args.overwrite)