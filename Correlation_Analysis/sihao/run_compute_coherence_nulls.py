import os
from experiment_io import load_traces, load_stim_info
from significance import compute_null_coherence


def get_files(root_path):
    """Get all files containing neural traces for experiments given a root path. Assumes directory structure to be:
    root_path
    ├── date
    │   ├── fish
    │   │   ├── avgs
    │   │   │   ├── merged
    │   │   │   │   ├── merged_raw.pkl

    Parameters
    ----------
    root_path : str
        Path to root directory.

    Returns
    -------
    files : list
        List of all files corresponding to traces of experiments.

    """
    # Get all dates
    dates = os.listdir(root_path)
    # Get all fish
    files = []
    for date in dates:
        fish = os.listdir(os.path.join(root_path, date))
        for f in fish:
            # Get all files
            files.append(os.path.join(root_path, date, f, 'avgs', 'merged', 'merged_raw.pkl'))

    return files


def __main__():
    # Define root path
    root_path = '/mnt/bronknas/Singles/Data_Singles_To_Analyze/Data_singles_2'

    # Get all files
    files = get_files(root_path)

    # Get corresponding stimulus information
    stim_info_path = '/mnt/bronknas/Singles/Data_Singles_To_Analyze/Data_singles_2/240823/Fish_H/avgs/NM_Stimul_setup_Combos_5_2023-08-24_18-51-07.mat'
    stim_info = load_stim_info(stim_info_path)

    # Compute coherence for all files
    for file in files:
        # Load traces
        traces = load_traces(file)

        # Determine number of stimuli in experiment
        n_stim = stim_info[:, 0].max()

        # Compute coherence null for all stimuli
        for stim_id in range(n_stim):
            # Construct savename including date and fish
            savename = 'coherence_null_' + file.split('/')[-5] + '_' + file.split('/')[-4] + '_nm' + str(stim_id) + '.npy'

            # Save in same directory as traces
            savename = os.path.join(os.path.dirname(file), savename)

            coherences = compute_null_coherence(traces, stim_info, stim_id, fs=2.18, use_saved=False,
                                                      saved_name='coherence_null_240823_FishH_nm2.npy')