import os
import datetime
import numpy as np
import pyinform as pyinf
from scipy import signal
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import norm
from tqdm import tqdm
from timeseries import create_timeseries, digitize_traces, lag_response, convolve_kernel
from utils import get_neuromast_stim_times


def compute_stimulus_transfer_entropy(traces, stim_info, stimulation_id, fs, n_lags=2, use_saved=False,
                                      saved_name='transfer_entropy.npy',
                                      save_name='transfer_entropy.npy'):
    """Compute the transfer entropy between the fluorescence traces of each neuron and the stimulus.

    Parameters
    ----------
    traces : array
        Array with shape (n_neurons, n_frames). The fluorescence traces of each neuron.
    stim_info : array
        Array with shape (n_stimuli, 2). The columns are: the times (in seconds) at which a neuromast
        was stimulated, column 2 is the neuromast number.
    stimulation_id: int
        In single neuromast stimulation, the neuromast number.
    fs : float
        Sampling frequency of the traces.
    n_lags : int, optional
        Number of lags to use for the transfer entropy computation (history window)
    use_saved : bool, optional
        If True, load saved transfer entropy from file. If False, compute transfer entropy.
    saved_name : str, optional
        Name of file to load saved transfer entropy from.
    save_name : str, optional
        Name of file to save transfer entropy to.

    Returns
    -------
    transfer_entropies : array
        Array with shape (n_neurons,). The transfer entropy between the fluorescence traces of each neuron
        and the stimulus.
    """
    if use_saved:
        print('Loading saved transfer entropy at {}'.format(os.path.abspath(saved_name)))
        # Load saved transfer entropy
        transfer_entropies = np.load(saved_name)
    else:
        # Get stimulus times
        stim_times = get_neuromast_stim_times(stim_info, stimulation_id)

        # Create timeseries of stimulus events
        timeseries = create_timeseries(stim_times, fs, traces.shape[1] / fs)

        # Digitize traces
        traces = digitize_traces(traces, n_bins=4)

        # Compute transfer entropy between stimulus and traces
        transfer_entropies = np.empty(traces.shape[0])
        for i, trace in enumerate(tqdm(traces)):
            # Make sure trace is non-negative
            trace = trace - np.min(trace)

            # Compute transfer entropy
            transfer_entropies[i] = pyinf.transfer_entropy(source=timeseries, target=trace, k=n_lags)

        # Save transfer entropy and notify user of absolute path (use new filename if file already exists, based
        # on timestamp)
        if os.path.exists(save_name):
            save_name = 'transfer_entropy_{}.npy'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            # Warn user that file already exists
            print('Warning: File {} already exists. Saving transfer entropy to {}'.format('transfer_entropy.npy',
                                                                                          os.path.abspath(save_name)))
        else:
            print('Saving transfer entropy to {}'.format(os.path.abspath(save_name)))

        np.save(save_name, transfer_entropies)

    return transfer_entropies


def compute_te_null(traces, stim_info, stimulation_id, fs, n_lags, n_shuffles=50, use_saved=False,
                    saved_name='transfer_entropy_null.npy', save_name='transfer_entropy_null.npy'):
    """Compute the null distribution of transfer entropy between the fluorescence traces of each neuron and the
    stimulus. The null distribution is computed by shuffling the stimulus times.

    Parameters
    ----------
    traces : array
        Array with shape (n_neurons, n_frames). The fluorescence traces of each neuron.
    stim_info : array
        Array with shape (n_stimuli, 2). The columns are: the times (in seconds) at which a neuromast
        was stimulated, column 2 is the neuromast number.
    stimulation_id: int
        In single neuromast stimulation, the neuromast number.
    fs : float
        Sampling frequency of the traces.
    n_lags : int, optional
        Number of lags to use for the transfer entropy computation (history window).
    n_shuffles : int, optional
        Number of shuffles to perform.
    use_saved : bool, optional
        If True, load saved null distribution from file. If False, compute null distribution.
    saved_name : str, optional
        Name of file to load saved null distribution from.
    save_name : str, optional
        Name of file to save null distribution to.

    Returns
    -------
    distributions : array
        Array with shape (n_neurons, n_shuffles). The null distribution of transfer entropy between the
        fluorescence traces of each neuron and the stimulus.
    """
    if use_saved:
        print('Loading saved null distribution at {}'.format(os.path.abspath(saved_name)))
        # Load saved null distribution
        distributions = np.load(saved_name)
    else:
        # Get stimulus times
        stim_times = get_neuromast_stim_times(stim_info, stimulation_id)

        # Create timeseries of stimulus events
        timeseries = create_timeseries(stim_times, fs, traces.shape[1] / fs)

        # Expand to 2D array with shape (n_frames, n_shuffles)
        timeseries = np.tile(timeseries, (n_shuffles, 1)).T

        # Shuffle timeseries along time axis
        timeseries = np.apply_along_axis(np.random.permutation, axis=0, arr=timeseries)

        # Digitize traces
        traces = digitize_traces(traces, n_bins=4)

        # Compute transfer entropy between stimulus and traces for each shuffle
        distributions = np.empty((traces.shape[0], n_shuffles))

        for i in tqdm(range(n_shuffles)):
            for j, trace in enumerate(traces):
                # Make sure trace is non-negative
                trace = trace - np.min(trace)

                # Compute transfer entropy
                distributions[j, i] = pyinf.transfer_entropy(source=timeseries[:, i], target=trace, k=n_lags)

        # Save null distribution and notify user of absolute path (use new filename if file already exists, based
        # on timestamp)
        if os.path.exists(save_name):
            save_name = 'transfer_entropy_null_{}.npy'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            # Warn user that file already exists
            print('Warning: File {} already exists. Saving null distribution to {}'.format('transfer_entropy_null.npy',
                                                                                           os.path.abspath(save_name)))
        print('Saving null distribution to {}'.format(os.path.abspath(save_name)))

        np.save(save_name, distributions)

    return distributions


def compute_stimulus_mutual_info(traces, stim_info, stimulation_id, fs, n_lags=4, use_saved=False,
                                 saved_name='mutual_info.npy',
                                 save_name='mutual_info.npy'):
    """Compute the mutual information between the fluorescence traces of each neuron and the stimulus.

    Parameters
    ----------
    traces : array
        Array with shape (n_neurons, n_frames). The fluorescence traces of each neuron.
    stim_info : array
        Array with shape (n_stimuli, 2). The columns are: the times (in seconds) at which a neuromast
        was stimulated, column 2 is the neuromast number.
    stimulation_id: int
        In single neuromast stimulation, the neuromast number.
    fs : float
        Sampling frequency of the traces.
    n_lags : int, optional
        Number of lags to use for the mutual information computation.
    use_saved : bool, optional
        If True, load saved mutual information from file. If False, compute mutual information.
    saved_name : str, optional
        Name of file to load saved mutual information from.
    save_name : str, optional
        Name of file to save mutual information to.

    Returns
    -------
    mutual_infos : array
        Array with shape (n_neurons,). The mutual information between the fluorescence traces of each neuron
        and the stimulus.
    """
    if use_saved:
        print('Loading saved mutual information at {}'.format(os.path.abspath(saved_name)))
        # Load saved mutual information
        mutual_infos = np.load(saved_name)
    else:
        # Get stimulus times
        stim_times = get_neuromast_stim_times(stim_info, stimulation_id)

        # Create timeseries of stimulus events
        timeseries = create_timeseries(stim_times, fs, traces.shape[1] / fs)
        timeseries = timeseries.reshape(-1, 1).ravel()[n_lags:]

        # Compute mutual information between stimulus and traces
        mutual_infos = np.empty(traces.shape[0])
        for i, trace in enumerate(tqdm(traces)):
            # Create lags of stimulus timeseries as features
            trace_lagged = lag_response(trace, n_lags=n_lags)
            n_timepoints, _ = trace_lagged.shape

            # Randomly sample N timepoints with no stimulation
            N = len(stim_times) * 3
            idx = np.random.choice(np.arange(n_timepoints), N, replace=False)
            trace_lagged = trace_lagged[idx, :]
            timeseries_subset = timeseries[idx]

            mutual_infos[i] = np.max(mutual_info_regression(trace_lagged, timeseries_subset))

        # Save mutual information and notify user of absolute path (use new filename if file already exists, based
        # on timestamp)
        if os.path.exists(save_name):
            save_name = 'mutual_info_{}.npy'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            # Warn user that file already exists
            print('Warning: File {} already exists. Saving mutual information to {}'.format('mutual_info.npy',
                                                                                            os.path.abspath(save_name)))

    return mutual_infos


def compute_granger_causality(traces, stim_info, stimulation_id, fs, n_lags=1, use_saved=False,
                              saved_name='granger_causality.npy',
                              save_name='granger_causality.npy'):
    """Compute the Granger causality between the fluorescence traces of each neuron and the stimulus.

    Parameters
    ----------
    traces : array
        Array with shape (n_neurons, n_frames). The fluorescence traces of each neuron.
    stim_info : array
        Array with shape (n_stimuli, 2). The columns are: the times (in seconds) at which a neuromast
        was stimulated, column 2 is the neuromast number.
    stimulation_id: int
        In single neuromast stimulation, the neuromast number.
    fs : float
        Sampling frequency of the traces.
    n_lags : int, optional
        Number of lags to use for the granger causality computation (history window).
    use_saved : bool, optional
        If True, load saved mutual information from file. If False, compute granger causality.
    saved_name : str, optional
        Name of file to load saved granger causality from.
    save_name : str, optional
        Name of file to save granger causality to.

    Returns
    -------
    granger_causality : array
        Array with shape (n_neurons,). The Granger causality between the fluorescence traces of each neuron
        and the stimulus.
    """

    if use_saved:
        print('Loading saved Granger causality at {}'.format(os.path.abspath(saved_name)))
        # Load saved mutual information
        granger_causality = np.load(saved_name)
    else:
        # Get stimulus times
        stim_times = get_neuromast_stim_times(stim_info, stimulation_id)

        # Create timeseries of stimulus events
        timeseries = create_timeseries(stim_times, fs, traces.shape[1] / fs)

        # Compute Granger causality between stimulus and traces
        granger_causality = np.empty(traces.shape[0])
        for i, trace in enumerate(tqdm(traces)):
            # Combine trace and timeseries as two columns
            trace = trace.reshape(-1, 1)
            timeseries = timeseries.reshape(-1, 1)
            x = np.hstack((trace, timeseries))

            # Compute Granger causality
            granger_causality[i] = grangercausalitytests(x, maxlag=n_lags, verbose=False)[1][0]['ssr_ftest'][1]

        # Save mutual information and notify user of absolute path (use new filename if file already exists, based
        # on timestamp)
        if os.path.exists(save_name):
            save_name = 'granger_causality_{}.npy'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            # Warn user that file already exists
            print('Warning: File {} already exists. Saving Granger causality to {}'.format('granger_causality.npy',
                                                                                           os.path.abspath(save_name)))
        print('Saving Granger causality to {}'.format(os.path.abspath(save_name)))

        np.save(save_name, granger_causality)

    return granger_causality


def compute_null_mutual_info(traces, stim_info, stimulation_id, fs, n_shuffles=50, use_saved=False,
                             saved_name='mutual_info_null.npy', save_name='mutual_info_null.npy'):
    """Compute the null distribution of mutual information between the fluorescence traces of each neuron and the
    stimulus. The null distribution is computed by shuffling the stimulus times.

    Parameters
    ----------
    traces : array
        Array with shape (n_neurons, n_frames). The fluorescence traces of each neuron.
    stim_info : array
        Array with shape (n_stimuli, 2). The columns are: the times (in seconds) at which a neuromast
        was stimulated, column 2 is the neuromast number.
    stimulation_id: int
        In single neuromast stimulation, the neuromast number.
    fs : float
        Sampling frequency of the traces.
    n_shuffles : int, optional
        Number of shuffles to perform.
    use_saved : bool, optional
        If True, load saved null distribution from file. If False, compute null distribution.
    saved_name : str, optional
        Name of file to load saved null distribution from.
    save_name : str, optional
        Name of file to save null distribution to.

    Returns
    -------
    distributions : array
        Array with shape (n_neurons, n_shuffles). The null distribution of mutual information between the
        fluorescence traces of each neuron and the stimulus.
    """
    if use_saved:
        print('Loading saved null distribution at {}'.format(os.path.abspath(saved_name)))
        # Load saved null distribution
        distributions = np.load(saved_name)
    else:
        # Get stimulus times
        stim_times = get_neuromast_stim_times(stim_info, stimulation_id)

        # Create timeseries of stimulus events
        timeseries = create_timeseries(stim_times, fs, traces.shape[1] / fs)

        # Expand to 2D array with shape (n_frames, n_shuffles)
        timeseries = np.tile(timeseries, (n_shuffles, 1)).T

        # Shuffle timeseries along time axis
        timeseries = np.apply_along_axis(np.random.permutation, axis=0, arr=timeseries)

        # Compute mutual information between stimulus and traces for each shuffle
        distributions = np.empty((traces.shape[0], n_shuffles))
        for i in tqdm(range(n_shuffles)):
            for j, trace in enumerate(traces):
                distributions[j, i] = mutual_info_regression(trace.reshape(-1, 1),
                                                             timeseries[:, i].reshape(-1, 1).ravel())

        # Save null distribution and notify user of absolute path (use new filename if file already exists, based
        # on timestamp)
        if os.path.exists(save_name):
            save_name = 'mutual_info_null_{}.npy'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            # Warn user that file already exists
            print('Warning: File {} already exists. Saving null distribution to {}'.format('mutual_info_null.npy',
                                                                                           os.path.abspath(save_name)))
        print('Saving null distribution to {}'.format(os.path.abspath(save_name)))

        np.save(save_name, distributions)

    return distributions


def compute_stimulus_coherence(traces, stim_info, stimulation_id, fs, kernel_decay=3, method='all'):
    """Compute the coherence of the fluorescence traces of each neuron with the stimulus. Coherence is
    given as the mean of the coherence over all frequencies.

    Parameters
    ----------
    traces : array
        Array with shape (n_neurons, n_frames). The fluorescence traces of each neuron.
    stim_info : array
        Array with shape (n_stimuli, 2). The columns are: the times (in seconds) at which a neuromast
        was stimulated, column 2 is the neuromast number.
    stimulation_id: int
        In single neuromast stimulation, the neuromast number. In multi neuromast stimulation, the pattern ID.
    fs : float
        Sampling frequency of the traces.

    kernel_decay : float, optional
        Decay constant of the exponential decay kernel in seconds.
    method : str, optional
        Method to compute coherence. 'all' computes coherence over all frequencies. 'bins' computes weighted mean
        coherence over frequency bins estimated from the stimulus.

    Returns
    -------
    correlations : array
        Array with shape (n_neurons,). The correlation of the fluorescence traces of each neuron with
        the stimulus.
    """
    # Get stimulus times
    stim_times = get_neuromast_stim_times(stim_info, stimulation_id)

    # Create timeseries of stimulus events
    timeseries = create_timeseries(stim_times, fs, traces.shape[1] / fs)

    # Convolve timeseries with exponential decay kernel
    timeseries = convolve_kernel(timeseries, kernel_decay, fs)

    # Get coherence coefficients for each neuron
    _, coherences = signal.coherence(timeseries, traces, fs, nperseg=256)

    if method == 'bins':
        # Compute fft of stimulus
        f, Pxx = signal.welch(timeseries, fs, nperseg=256)

        # Compute weighted mean coherence over frequency bins
        coherences = np.average(coherences, axis=1, weights=Pxx)

    elif method == 'all':
        # Average coherence over frequencies
        coherences = np.mean(coherences, axis=1)
    else:
        raise ValueError('Method not recognized. Use "all" or "bins"')

    return coherences


def compute_null_coherence(traces, stim_info, stimulation_id, fs, n_shuffles=50, use_saved=False,
                           saved_name='coherence_null.npy', save_name='coherence_null.npy'):
    """Compute the null distribution of coherence of the fluorescence traces for each neuron with the stimulus.
    Coherence is given as the mean of the coherence over all frequencies. The null distribution is computed
    by shuffling the stimulus times.

    Parameters
    ----------
    traces : array
        Array with shape (n_neurons, n_frames). The fluorescence traces of each neuron.
    stim_info : array
        Array with shape (n_stimuli, 2). The columns are: the times (in seconds) at which a neuromast
        was stimulated, column 2 is the neuromast number.
    stimulation_id: int
        In single neuromast stimulation, the neuromast number. In multi neuromast stimulation, the pattern ID.
    fs : float
        Sampling frequency of the traces.
    n_shuffles : int, optional
        Number of shuffles to perform.
    use_saved : bool, optional
        If True, load saved null distribution from file. If False, compute null distribution.
    saved_name : str, optional
        Name of file to load saved null distribution from.
    save_name : str, optional
        Name of file to save null distribution to.

    Returns
    -------
    distributions : array
        Array with shape (n_neurons, n_shuffles). The null distribution of coherence of the fluorescence
        traces of each neuron with the stimulus.
    """
    if use_saved:
        print('Loading saved null distribution at {}'.format(os.path.abspath(saved_name)))
        # Load saved null distribution
        distributions = np.load(saved_name)
    else:
        # Get stimulus times
        stim_times = get_neuromast_stim_times(stim_info, stimulation_id)

        # Create timeseries of stimulus events
        timeseries = create_timeseries(stim_times, fs, traces.shape[1] / fs)

        # Expand to 2D array with shape (n_frames, n_shuffles)
        timeseries = np.tile(timeseries, (n_shuffles, 1)).T

        # Shuffle timeseries along time axis
        timeseries = np.apply_along_axis(np.random.permutation, axis=0, arr=timeseries)

        # Get coherence coefficients for each neuron for each shuffle
        distributions = np.empty((traces.shape[0], n_shuffles))
        for i in tqdm(range(n_shuffles)):
            _, coherences = signal.coherence(timeseries[:, i], traces, fs, nperseg=256)
            # Average coherence over frequencies
            coherences = np.mean(coherences, axis=1)
            distributions[:, i] = coherences

        # Save null distribution and notify user of absolute path (use new filename if file already exists, based
        # on timestamp)
        if os.path.exists(save_name):
            save_name = 'coherence_null_{}.npy'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            # Warn user that file already exists
            print('Warning: File {} already exists. Saving null distribution to {}'.format('coherence_null.npy',
                                                                                           os.path.abspath(save_name)))
        print('Saving null distribution to {}'.format(os.path.abspath(save_name)))

        np.save(save_name, distributions)
    return distributions


def compute_xcorr_lag(trace, stimulus):
    """Compute the lag at which the cross correlation between a trace and a stimulus is maximal

    Parameters
    ----------
    trace : array
        Array with shape (n_frames,). The fluorescence trace of a neuron.
    stimulus : array
        Array with shape (n_frames,). The stimulus.

    Returns
    -------
    lag : int
        The lag at which the cross correlation between the trace and the stimulus is maximal. Positive
        if the stimulus lags behind the trace, negative if the stimulus precedes the trace.
    """
    # Compute cross correlation
    xcorr = signal.correlate(trace, stimulus, mode='same')

    # Get lag at which cross correlation is maximal
    lag = np.argmax(xcorr) - len(trace) // 2

    return lag


def compute_stimulus_correlation(traces, stim_info, stimulation_id, fs, kernel_decay=3):
    """Compute the correlation of the fluorescence traces of each neuron with the stimulus. Finds the lag at
    which the cross correlation between the trace and the stimulus is maximal and computes the Pearson
    correlation coefficient at that lag.

    Parameters
    ----------
    traces : array
        Array with shape (n_neurons, n_frames). The fluorescence traces of each neuron.
    stim_info : array
        Array with shape (n_stimuli, 2). The columns are: the times (in seconds) at which a neuromast
        was stimulated, column 2 is the neuromast number.
    stimulation_id: int
        In single neuromast stimulation, the neuromast number.
    fs : float
        Sampling frequency of the traces.
    kernel_decay : float, optional
        Decay constant of the exponential decay kernel in seconds.

    Returns
    -------
    correlations : array
        Array with shape (n_neurons,). The correlation of the fluorescence traces of each neuron with
        the stimulus.
    """
    # Get stimulus times
    stim_times = get_neuromast_stim_times(stim_info, stimulation_id)

    # Create timeseries of stimulus events
    timeseries = create_timeseries(stim_times, fs, traces.shape[1] / fs, single_ticks=True)

    # Convolve timeseries with exponential decay kernel
    timeseries = convolve_kernel(timeseries, kernel_decay, fs)

    # Get correlation coefficients for each neuron
    correlations = np.empty(traces.shape[0])
    for i, trace in enumerate(traces):
        # Compute cross correlation and pick lag at which it is maximal
        lag = compute_xcorr_lag(trace, timeseries)

        # Shift timeseries by lag
        timeseries = np.roll(timeseries, lag)

        # Compute pearson correlation coefficient
        correlations[i] = np.corrcoef(trace, timeseries)[0, 1]

    return correlations


def get_significant_neurons(coherences, coherences_null, alpha=0.01, coherence_threshold=0.7):
    """Get the indices of neurons that are significantly correlated with the stimulus

    Parameters
    ----------
    coherences : array
        Array with shape (n_neurons,). The correlation of the fluorescence traces of each neuron with
        the stimulus.
    coherences_null : array
        Array with shape (n_neurons, n_shuffles). The null distribution of coherence of the fluorescence
        traces of each neuron with the stimulus.
    alpha : float, optional
        Significance level.
    coherence_threshold : float, optional
        Threshold for coherence. Neurons with coherence below this threshold are not considered.

    Returns
    -------
    significant_neurons : array
        Array with shape (n_significant_neurons,). The indices of the neurons that are significantly
        responsive to the stimulus.
    mask : array
        Boolean array with shape (n_neurons,). True if neuron has significant coherence with the
        stimulus, False otherwise.
    """
    # Prepare array for p-values
    p_values = np.empty(coherences.shape[0])

    # Fit normal distribution to null distribution for each neuron
    for neuron in range(coherences_null.shape[0]):
        mu, sigma = norm.fit(coherences_null[neuron, :])

        # Compute p-value
        p_value = norm.sf(coherences[neuron], loc=mu, scale=sigma)

        # Save p-value in array
        p_values[neuron] = p_value

    # Get significant neurons
    significant_neurons = np.where(p_values < alpha)[0]

    # Arrange as mask
    mask = np.zeros(coherences.shape[0], dtype=bool)
    mask[significant_neurons] = True

    # Check that coherence values for significant neurons are above 0.08 at least and set to False if not
    idx_to_remove = np.argwhere(coherences < coherence_threshold)
    mask[idx_to_remove] = False

    # Get indices of significant neurons
    significant_neurons = np.argwhere(mask == True).squeeze()

    # Print number of significant neurons
    print('Number of significant neurons: {}'.format(len(significant_neurons)))

    return significant_neurons, mask


def select_svm_neurons(traces_trials, method='percentile', threshold=None):
    """Select neurons for SVM analysis based on their response to the stimulus.

    Parameters
    ----------
    traces_trials : array
        Array with shape (n_neurons, n_samples, n_trials, n_stim). The fluorescence traces of each neuron.
    method : str, optional
        Method to select neurons. 'percentile' selects neurons that respond in the top percentile of the
        response distribution. Options are: 'percentile', 'threshold'.
        'percentile' selects neurons that respond in the top percentile of the response distribution.
        'threshold' selects neurons that respond above a certain threshold in response amplitude.
    threshold : float, optional
        Threshold for response. Neurons with response below this threshold are not considered. If method
        is 'percentile', the default is 99.977th percentile (3-sigma). If method is 'threshold', the default
        is 0.3.
    """
    # Determine default threshold if not provided
    if method == 'percentile' and threshold is None:
        threshold = 99.977
    elif method == 'threshold' and threshold is None:
        threshold = 0.3
    else:
        raise ValueError('Method not recognized. Options are: [percentile, threshold]')

    # Get highest responding for each class
    n_stim = traces_trials.shape[3]
    neuron_selector = []
    for i in range(n_stim):
        if method == 'percentile':
            # Get nth percentile zscore
            percentile = np.percentile(np.max((np.mean(traces_trials[:, :, :, i], axis=2)), axis=1), threshold)
            neuron_selector.append(np.argwhere(np.max((np.mean(traces_trials[:, :, :, i], axis=2)), axis=1) > percentile).squeeze())
        elif method == 'threshold':
            neuron_selector.append(np.argwhere(np.max((np.mean(traces_trials[:, :, :, i], axis=2)), axis=1) > threshold).squeeze())

    neuron_selector = np.unique(np.concatenate(neuron_selector))

    return neuron_selector
