import numpy as np


def compute_sta(traces, stimuli):
    """Compute spike triggered average of the stimulus one frame before the recorded activity.

    Parameters
    ----------
    traces : np.ndarray
        Array of shape (n_neurons, n_frames) containing traces.
    stimuli : np.ndarray
        Array of shape (n_frames, n_stimuli) containing stimuli.

    Returns
    -------
    sta : np.ndarray
        Array of shape (n_neurons, n_stimuli) containing spike triggered averages.
    """
    # Shift stimuli by one frame in time
    stimuli = np.roll(stimuli, -1, axis=0)

    # Matrix multiply traces and stimuli
    sta = np.matmul(traces, stimuli)

    # Normalize by number of frames
    sta /= traces.shape[1]

    return sta