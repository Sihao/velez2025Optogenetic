import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm
from scipy.interpolate import splev, splrep
from scipy import signal
from scipy.optimize import curve_fit
from utils import get_neuromast_stim_times


def spline_fit_traces(traces, degree=3, smoothness=1 / np.pi):
    """Fit a spline to each trace

    Parameters
    ----------
    traces : array
        Array with shape (n_neurons, n_frames). The fluorescence traces of each neuron.
    degree : int, optional
        Degree of the spline.
    smoothness : float, optional
        Smoothness of the spline.

    Returns
    -------
    splined_traces : array
        Array with shape (n_neurons, n_frames). The spline fits of the fluorescence traces of each neuron.
    """
    # Create time array
    t = np.arange(traces.shape[1])

    # Fit spline to each trace
    splined_traces = np.empty_like(traces)
    for i in tqdm(range(traces.shape[0])):
        # Fit B-spline to trace
        trace = traces[i, :]
        s = splrep(t, trace, k=degree, s=smoothness)
        splined_traces[i, :] = splev(t, s)

    return splined_traces


def detrend_traces(traces, type='polynomial', parallel=False):
    """Detrend traces by subtracting an exponential fit

    Parameters
    ----------
    traces : array
        Array with shape (n_neurons, n_frames). The fluorescence traces of each neuron.
    type : str, optional
        Type of detrending. Can be 'polynomial' or 'exponential'.
    parallel : bool, optional
        If True, fit exponential to each trace in parallel. If False, fit exponential to each trace
        sequentially.
    Returns
    -------
    detrended : array
        Array with shape (n_neurons, n_frames). The detrended fluorescence traces of each neuron.
    """
    # Create time array
    t = np.arange(traces.shape[1])

    # Fit exponential to each trace
    if parallel:
        # Parallelize fitting
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor() as executor:
            detrended = np.array(list(executor.map(detrend_single_trace, traces, [t] * traces.shape[0], type=type)))

        traces = np.array(list(detrended))
    else:
        # Sequential fitting
        for i in tqdm(range(traces.shape[0])):
            traces[i, :] = detrend_single_trace(traces[i, :], t, type=type)

    return traces


def exponential(t, a, b, c):
    """Exponential function for curve fitting

    Parameters
    ----------
    t : array
        Array with shape (n_frames,). The time points at which to evaluate the function.
    a : float
        Amplitude of the exponential.
    b : float
        Decay constant of the exponential.
    c : float
        Offset of the exponential.

    Returns
    -------
    y : array
        Array with shape (n_frames,). The values of the exponential at each time point.
    """
    return a * np.exp(b * t) + c


def detrend_single_trace(trace, t, type='exponential'):
    """Detrend a single trace by subtracting a fit

    Parameters
    ----------
    trace : array
        Array with shape (n_frames,). The fluorescence trace of a neuron.
    t : array
        Array with shape (n_frames,). The time points at which to evaluate the function.
    type : str, optional
        Type of detrending. Can be 'polynomial' or 'exponential'.

    Returns
    -------
    detrended : array
        Array with shape (n_frames,). The detrended fluorescence trace of the neuron.
    """
    try:
        if type == 'polynomial':
            # Fit polynomial to trace
            p = np.polyfit(t, trace, 3)
            detrended = trace - np.polyval(p, t)
            # # Plot detrended trace versus original trace
            # fig, ax = plt.subplots(1, 1)
            # ax.plot(t, trace)
            # ax.plot(t, np.polyval(p, t))
            # ax.plot(t, detrended)
            # fig.show()
            return detrended

        elif type == 'exponential':
            p_init = [trace[0], -1, 0]
            popt, _ = curve_fit(exponential, t, trace,
                                p0=p_init,
                                bounds=([-np.inf, -np.inf, -np.inf], [np.inf, 0, np.inf]),
                                method='dogbox')
            detrended = trace - exponential(t, *popt)
            # # Plot detrended trace versus original trace
            # fig, ax = plt.subplots(1, 1)
            # ax.plot(t, trace)
            # ax.plot(t, exponential(t, *popt))
            # ax.plot(t, detrended)
            # fig.show()
            return detrended

    except RuntimeError or ValueError:
        print('Warning: Could not fit to trace')
        print('Trace will not be detrended')
        return trace


def filter_traces(traces, fs, kernel_decay, bleaching_tau=200):
    """Filter traces with a band pass filter to approximate temporal resolution of calcium decay kernel and remove low
    frequency bleaching. Approximating photo

    Parameters
    ----------
    traces : array
        Array with shape (n_neurons, n_frames). The fluorescence traces of each neuron.
    fs : float
        Sampling frequency of the traces.
    kernel_decay : float
        Decay constant of the exponential decay kernel in seconds.
    bleaching_tau : float, optional
        Decay constant of the bleaching in seconds.

    Returns
    -------
    filtered : array
        Array with shape (n_neurons, n_frames). The filtered fluorescence traces of each neuron.
    """
    # Band pass filter traces to approximate temporal resolution decay kernel and decay tau, use Chebyshev type 1
    traces = signal.filtfilt(
        signal.firwin(101, [1 / (bleaching_tau * 2 * np.pi), 1 / (kernel_decay)], fs=fs),
        1, traces, axis=1,
        method='gust', irlen=200)

    # # Plot some example traces (filtered and unfiltered on same subplot)
    # fig, ax = plt.subplots(3, 2)
    # ax[0][0].plot(traces[0, :])
    # ax[0][1].plot(traces_filtered[0, :])
    # ax[1][0].plot(traces[1, :])
    # ax[1][1].plot(traces_filtered[1, :])
    # ax[2][0].plot(traces[2, :])
    # ax[2][1].plot(traces_filtered[2, :])
    # fig.show()

    return traces


def digitize_traces(traces, n_bins=4):
    """Digitize traces by binning them into n_bins bins.

    Parameters
    ----------
    traces : array
        Array with shape (n_neurons, n_frames). The fluorescence traces of each neuron.
    n_bins : int, optional
        Number of bins to bin traces into.

    Returns
    -------
    digitized : array
        Array with shape (n_neurons, n_frames). The digitized fluorescence traces of each neuron.
    """
    # Compute bins based on distribution on mean and standard deviation of data
    for i, trace in enumerate(traces):
        bins = np.linspace(np.mean(trace), np.mean(trace) + 3 * np.std(trace), n_bins)
        trace = np.digitize(trace, bins=bins)
        traces[i, :] = trace

    return traces


def lag_response(trace, n_lags=4):
    """Shift the trace backward in time and stack them to create a new array with shape (n_frames - n_lags, n_lags).
    Here each lag can be interpreted as a feature.

    Parameters
    ----------
    trace : array
        Array with shape (n_frames,). The fluorescence trace of a neuron.
    Returns
    -------
    shifted : array
        Array with shape (n_frames - n_lags, n_lags). The shifted fluorescence traces of the neuron.
    """
    # Create empty array for shifted traces
    shifted = np.empty((trace.shape[0] - n_lags, n_lags))
    # Shift trace backward in time
    for i in range(n_lags):
        shifted[:, i] = trace[i:-n_lags + i]

    return shifted


def create_timeseries(stim_times, fs, duration, single_ticks=False):
    """Create a timeseries of stimulus events

    Parameters
    ----------
    stim_times : array
        Array of times (in seconds) at which a given neuromast was stimulated.
    fs : int
        Sampling frequency of the timeseries.
    duration : float
        Duration of the timeseries (in seconds).
    single_ticks : bool, optional
        If True, only allow one stimulus tick per stimulus ID (sometimes there are two frames with a tick
        because the stimulus is delivered between two frames). Default is False.

    Returns
    -------
    timeseries : array
        Array of length duration*fs. Each element is 1 if a stimulus was delivered at that time, 0
        otherwise.
    """
    # Create empty array for timeseries
    timeseries = np.zeros(int(np.ceil(duration * fs)))

    # Alert user that there are stimulus times outside the duration of the timeseries
    if max(stim_times) > duration:
        print('Warning: Stimulus times outside the duration of the timeseries')

    # Only choose stimulus times that are within the duration of the timeseries
    stim_times = [stim_time for stim_time in stim_times if stim_time < duration]

    # Set stimulus times to 1
    for stim_time in stim_times:
        timeseries[int((stim_time) * fs)] = 1

    if single_ticks:
        # Get idx of stimulus ticks
        tick_idx = np.where(timeseries == 1)[0]

        # Get differences between ticks
        diffs = np.diff(tick_idx)

        # Find ticks that are immediately preceded by another tick
        duplicate_tick_idx = np.where(diffs == 1)[0]

        # Set ticks to 0 if they are immediately preceded by another tick
        timeseries[tick_idx[duplicate_tick_idx]] = 0

    return timeseries


def convolve_kernel(timeseries, kernel_decay, fs):
    """Convolve a timeseries with an exponential decay kernel

    Parameters
    ----------
    timeseries : array
        Array of length duration*fs. Each element is 1 if a stimulus was delivered at that time, 0
        otherwise.
    kernel_decay : float
        Decay constant of the exponential decay kernel in seconds.
    fs : float
        Sampling frequency of the timeseries.

    Returns
    -------
    convolved : array
        Array of length duration*fs. The convolved timeseries.
    """
    # Create exponential decay kernel of length 2 * kernel_decay
    kernel = np.exp(-np.arange(0, 2 * kernel_decay, 1 / fs) / kernel_decay)

    # Convolve timeseries with kernel
    convolved = np.convolve(timeseries, kernel, mode='same')

    return convolved


def zscore_traces(traces, method='standard'):
    """Z-score an array of traces

    Parameters
    ----------
    traces : array
        Array with shape (n_neurons, n_frames). The fluorescence traces of each neuron.
    method : str, optional
        Method to use for z-scoring. Can be 'standard' or 'spline'. Default is 'standard'.

    Returns
    -------
    zscored : array
        Array with shape (n_neurons, n_frames). The z-scored fluorescence traces of each neuron.
    """
    if method == 'standard':
        # Z-score traces using mean and standard deviation
        zscored = (traces - np.mean(traces, axis=1)[:, np.newaxis]) / np.std(traces, axis=1)[:, np.newaxis]

    elif method == 'spline':
        # Fit spline sing splev and splrep
        splined = spline_fit_traces(traces)

        # Subtract spline to determine noise floor
        noise_floor = traces - splined
        noise_mean = np.mean(noise_floor, axis=1)
        noise_std = np.std(noise_floor, axis=1)

        # Z-score traces using noise floor mean and std
        zscored = (traces - noise_mean[:, np.newaxis]) / noise_std[:, np.newaxis]

    return zscored


def preprocess_traces(traces, fs, kernel_decay=3, bleaching_tau=200, filter=True, zscore=True):
    """Super function to preprocess traces by filtering and z-scoring

    Parameters
    ----------
    traces : array
        Array with shape (n_neurons, n_frames). The fluorescence traces of each neuron.
    fs : float
        Sampling frequency of the traces.
    kernel_decay : float, optional
        Decay constant of the exponential decay kernel in seconds. Used for filtering.
    bleaching_tau : float, optional
        Decay constant of the bleaching in seconds. Used for filtering.
    filter : bool, optional
        Whether to filter the traces or not.
    zscore : bool, optional
        Whether to z-score the traces or not.

    Returns
    -------
    traces : array
        Array with shape (n_neurons, n_frames). The preprocessed fluorescence traces of each neuron.
    """
    if filter:
        # Filter traces
        traces = filter_traces(traces, fs, kernel_decay, bleaching_tau)

    if zscore:
        # Z-score traces
        traces = zscore_traces(traces)

    return traces


def avg_responsive_cells(traces, significant_neurons):
    """Compute the average response of neurons which are significantly responsive to the stimulus.
    Small hack to deal with outlier traces with obvious artifacts by removing traces with SNR > 10.

    Parameters
    ----------
    traces : array
        Array with shape (n_neurons, n_frames). The fluorescence traces of each neuron.
    significant_neurons : array
        Array with shape (n_significant_neurons,). The indices of the neurons that are significantly
        responsive to the stimulus.

    Returns
    -------
    avg_trace : array
        Array with shape (n_frames,). The average response of neurons which are significantly responsive
        to the stimulus.

    """
    # Get traces of significant neurons
    significant_traces = traces[significant_neurons, :]

    # Remove traces with SNR > 10
    snr = np.max(significant_traces, axis=1) / np.std(significant_traces, axis=1)
    significant_traces = significant_traces[np.where(snr < 10)[0], :]

    # Z-score traces
    zscored = zscore_traces(significant_traces, method='standard')

    # Average traces
    avg_trace = np.mean(zscored, axis=0)

    return avg_trace


def construct_stimuli_timeseries(stim_info, duration, fs, single_ticks=False):
    """Construct a timeseries of all stimulus events

    Parameters
    ----------
    stim_info : array
        Array with shape (n_stimuli, 2). The columns are: the times (in seconds) at which a neuromast
        was stimulated, column 2 is the neuromast number.
    fs : float
        Sampling frequency of the timeseries.
    duration : float
        Duration of the timeseries (in seconds).
    single_ticks : bool, optional
        If True, only allow one stimulus tick per stimulus ID (sometimes there are two frames with a tick
        because the stimulus is delivered between two frames). Default is False.

    Returns
    -------
    timeseries_all : array
        Array of shape (n_frames, n_stimuli). Each column is a timeseries of stimulus events for a given
        neuromast.
        Each element is 1 if a stimulus was delivered at that time, 0
        otherwise.
    """
    # Get stimulus times for each neuromast
    stim_times = []
    n_neuromasts = int(np.max(stim_info[:, 1]))

    for i in range(1, n_neuromasts + 1):
        stim_times.append(get_neuromast_stim_times(stim_info, i))

    # Construct timeseries for each neuromast
    for i, stim_time in enumerate(stim_times):
        # Create timeseries of stimulus events
        timeseries = create_timeseries(stim_time, fs, duration, single_ticks=single_ticks)

        # Ensure column vector
        if len(timeseries.shape) == 1:
            timeseries = timeseries[:, np.newaxis]

        if i == 0:
            timeseries_all = timeseries
        else:
            timeseries_all = np.hstack((timeseries_all, timeseries))

    return timeseries_all


def avg_trials(traces, stim_info, fs=2.18, n_samples=None, zscore=True):
    """Average trials for each stimulus.
    Average is taken over the time from stimulus until the start of the next stimulus of same ID.

    Parameters
    ----------
    traces : array
        Array with shape (n_neurons, n_samples). The traces of each neuron.
    stim_info : array
        Array with shape (n_stimuli, 2). The stimulus information. The first column is the time (in seconds) at which
        the stimulus was presented. The second column is the stimulus ID.
    fs : float, optional
        Sampling frequency of the traces. Default is 2.18.
    n_samples : int, optional
        Number of samples to take for each trial. Default is the shortest time between two stimulations of
        stimulus ID == 1.
    zscore : bool, optional
        Whether to z-score the traces or not. Default is True.

    Returns
    -------
    traces_avg : array
        Array with shape (n_neurons, n_samples). The trial-averaged traces for each neuron.
    """
    # Get number of neurons
    n_neurons = traces.shape[0]

    if n_samples is None:
        # Create timeseries of stimlus times
        stim_times = get_neuromast_stim_times(stim_info, 1)
        timeseries = create_timeseries(stim_times, fs=fs, duration=traces.shape[1] / fs)

        # Difference between stimulations for one stimulus
        stim_indices = np.where(timeseries == 1)[0]

        # Get differences
        stim_diff = np.diff(stim_indices)

        # Get max difference
        n_samples = int(np.max(stim_diff) - 1)

    else:
        n_samples = int(n_samples)

    # Average trials for each stimulus
    stim_times = get_neuromast_stim_times(stim_info, 1)
    # Convert to frames
    stim_frames = np.array(stim_times) * fs
    stim_frames = stim_frames.astype(int)

    # Remove duplicate entries
    stim_frames = np.unique(stim_frames)

    # Create empty array for averaged traces
    n_trials = len(stim_frames)
    traces_avg = np.zeros((n_neurons, n_samples, n_trials))

    # Get trials
    for j, stim_frame in enumerate(stim_frames):
        # Get start and end frames
        start_frame = stim_frame
        end_frame = int(stim_frame + n_samples)

        # Get trace
        trace = traces[:, start_frame:end_frame]

        # Add to array
        traces_avg[:, :trace.shape[1], j] = trace

    if zscore:
        # Zscore traces
        traces_avg = zscore_traces(traces_avg, method='standard')

    # Average trials
    traces_avg = np.mean(traces_avg, axis=-1).squeeze()

    return traces_avg


def split_trials(traces, stim_info, fs=2.18, n_samples=None, zscore=True):
    """Split trials into trials (each trial contains all stimuli once).
    Split is taken over the time from stimulus until the start of the next stimulus of same ID.

    Parameters
    ----------
    traces : array
        Array with shape (n_neurons, n_samples). The traces of each neuron.
    stim_info : array
        Array with shape (n_stimuli, 2). The stimulus information. The first column is the time (in seconds) at which
        the stimulus was presented. The second column is the stimulus ID.
    fs : float, optional
        Sampling frequency of the traces. Default is 2.18.
    n_samples : int, optional
        Number of samples to take for each trial. Default is the longest time between two stimulations of
        stimulus ID == 1.
    zscore : bool, optional
        Whether to z-score the traces or not. Default is True.

    Returns
    -------
    traces_split : array
        Array with shape (n_neurons, n_samples, n_trials). The trial-split traces for each neuron.
    """
    # Get number of neurons
    n_neurons = traces.shape[0]

    if n_samples is None:
        # Create timeseries of stimlus times
        stim_times = get_neuromast_stim_times(stim_info, 1)
        timeseries = create_timeseries(stim_times, fs=fs, duration=traces.shape[1] / fs)

        # Difference between stimulations for one stimulus
        stim_indices = np.where(timeseries == 1)[0]

        # Get differences
        stim_diff = np.diff(stim_indices)

        # Get max difference
        n_samples = int(np.max(stim_diff) - 1)

    else:
        n_samples = int(n_samples)

    # Split trials for each stimulus
    stim_times = get_neuromast_stim_times(stim_info, 1)
    # Convert to frames
    stim_frames = np.array(stim_times) * fs
    stim_frames = stim_frames.astype(int)

    # Remove duplicate entries
    stim_frames = np.unique(stim_frames)

    # Remove stim frame if it is too close to preceding stim frame
    stim_diff = np.diff(stim_frames)
    stim_frames = stim_frames[np.where(stim_diff > 1)[0]]

    # Create empty array for split traces
    n_trials = len(stim_frames)
    traces_split = np.zeros((n_neurons, n_samples, n_trials))

    # Get trials
    for j, stim_frame in enumerate(stim_frames):
        # Get start and end frames
        start_frame = stim_frame
        end_frame = int(stim_frame + n_samples)

        # Get trace
        trace = traces[:, start_frame:end_frame]

        # Add to array
        traces_split[:, :trace.shape[1], j] = trace

    if zscore:
        # Zscore traces
        traces_split = zscore_traces(traces_split, method='standard')

    return traces_split


def split_stims(traces, stim_info, fs=2.1802, n_samples=None, zscore=True):
    """Split trials for each stimulus.
    Split is taken over the time of each stimulus ID in the stim_info array.

    Parameters
    ----------
    traces : array
        Array with shape (n_neurons, n_samples). The traces of each neuron.
    stim_info : array
        Array with shape (n_stimuli, 2). The stimulus information. The first column is the time (in seconds) at which
        the stimulus was presented. The second column is the stimulus ID.
    fs : float, optional
        Sampling frequency of the traces. Default is 2.18.
    n_samples : int, optional
        Number of samples to take for each post-stimulus interval. Default is the longest time between two stimulations of
        stimulus ID == 1 / n_stim.
    zscore : bool, optional
        Whether to z-score the traces or not. Default is True.

    Returns
    -------
    traces_split : array
        Array with shape (n_neurons, n_samples, n_trials, n_stim). The neural activity post-stimulus for each neuron,
        for each stimulus, for each trial.
    """
    # Get number of neurons
    n_neurons = traces.shape[0]

    # Get number of stimuli
    n_stim = len(np.unique(stim_info[:, 1]))

    if n_samples is None:
        # Create timeseries of stimulus times
        stim_times = get_neuromast_stim_times(stim_info, stim_info[0, 1])
        timeseries = create_timeseries(stim_times, fs=fs, duration=traces.shape[1] / fs)

        # Difference between stimulations for one stimulus
        stim_indices = np.where(timeseries == 1)[0]

        # Get differences
        stim_diff = np.diff(stim_indices)

        # Get max difference
        n_samples = int((np.min(stim_diff) - 1)/n_stim)

    else:
        n_samples = int(n_samples)

    # Array to hold final output
    traces_split = np.zeros((n_neurons, n_samples, n_stim))
    traces_split_full = []

    # Split trials for each stimulus independently
    stim_idx = np.unique(stim_info[:, 1])
    for stim_id in stim_idx:
        stim_times = get_neuromast_stim_times(stim_info, stim_id)

        # Convert to frames
        stim_frames = np.array(stim_times) * fs
        stim_frames = stim_frames.astype(int)

        # Remove duplicate entries
        stim_frames = np.unique(stim_frames)

        # Remove stim frame if it is too close to preceding stim frame
        stim_diff = np.diff(stim_frames)
        stim_frames = stim_frames[np.where(stim_diff > 1)[0]]

        # Create empty array for split traces
        n_trials = len(stim_frames)
        traces_split = np.zeros((n_neurons, n_samples, n_trials))

        # Get trials
        for j, stim_frame in enumerate(stim_frames):
            # Get start and end frames
            start_frame = stim_frame
            end_frame = int(stim_frame + n_samples)

            # Get trace
            trace = traces[:, start_frame:end_frame]

            # Add to array
            traces_split[:, :trace.shape[1], j] = trace

        # Append to list of all split traces
        traces_split_full.append(traces_split)

    # Check number of trials for each stim, and shorten all traces to the minimum number of trials
    n_trials = [traces_stim.shape[2] for traces_stim in traces_split_full]

    # Get minimum number of trials
    min_trials = min(n_trials)

    # Shorten all traces to the minimum number of trials
    traces_split = np.zeros((n_neurons, n_samples, min_trials, n_stim))
    for i, traces_stim in enumerate(traces_split_full):
        traces_split[:, :, :, i] = traces_stim[:, :, :min_trials]

    if zscore:
        # Zscore traces
        traces_split = zscore_traces(traces_split, method='standard')

    return traces_split


def compute_n_samples(stim_info, duration, fs=2.1802):
    """Compute the number of samples to take for each post-stimulus interval.
    The number of samples is taken as the maximum time between two stimulations of stimulus ID == 1.

    Parameters
    ----------
    stim_info : array
        Array with shape (n_stimuli, 2). The stimulus information. The first column is the time (in seconds) at which
        the stimulus was presented. The second column is the stimulus ID.
    duration : float
        Duration of the traces (in seconds).
    fs : float, optional
        Sampling frequency of the traces. Default is 2.18.

    Returns
    -------
        n_samples : int
        The number of samples to take for each post-stimulus interval.
    """
    n_stim = len(np.unique(stim_info[:, 1]))
    timeseries_all = []
    # Get range of stimulus IDs
    stim_idx = np.unique(stim_info[:,1])
    for id in stim_idx:
        stim_times = get_neuromast_stim_times(stim_info, id)
        timeseries = create_timeseries(stim_times, fs, duration=duration, single_ticks=True)
        timeseries_all.append(timeseries)

    # Make into array
    timeseries_all = np.array(timeseries_all)

    # Bitwise OR over all trials
    timeseries_all = np.any(timeseries_all, axis=0)

    # Difference between stimulations for one stimulus
    stim_indices = np.where(timeseries_all == 1)[0]

    # Get differences
    stim_diff = np.diff(stim_indices)

    # Get max difference
    n_samples = int((np.min(stim_diff) - 1))

    return n_samples

