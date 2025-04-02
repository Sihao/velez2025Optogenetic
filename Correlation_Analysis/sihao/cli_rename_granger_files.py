import os
import argparse
import re

def main(input_path):
    """Rename all files with *Data_singles_2* in input_dir.

    Parameters
    ----------
    input_path : str
        Path to input directory.

    Returns
    -------
    None
        Renames all files in input_dir.
    """
    # List files in input_dir
    files = os.listdir(input_path)

    # Rename files
    for f in files:
        # Get file extension
        file_extension = f.split('.')[-1]

        # Get file name
        file_name = f.split('.')[0]

        # Find substring containing date and date
        re_substring = r'\d{6}_\d{6}'

        try:
            substring = re.search(re_substring, file_name).group()
        except AttributeError:
            print(f'No match found for {f}. Skipping...')
            continue

        if substring in file_name:
            # Remove substring and replace with date (parent-parent) + parent directory name
            date = substring.split('_')[0]
            parent_dir = os.path.basename(os.path.normpath(input_path))  # Cleans up trailing '/'
            replacement = date + '_' + parent_dir
            file_name = file_name.replace(substring, replacement)

            # Rename file
            os.rename(os.path.join(input_path, f), os.path.join(input_path, file_name + '.' + file_extension))


if __name__ == '__main__':
    # Parse arguments
    args = argparse.ArgumentParser()
    args.add_argument('--input_path', type=str, help='Path to input directory.')
    args = args.parse_args()
    input_path = args.input_path

    main(input_path)