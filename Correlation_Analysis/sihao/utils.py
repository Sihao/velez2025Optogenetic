import numpy as np
from scipy.special import log_ndtr
from rastermap import Rastermap
from scipy.spatial.distance import mahalanobis
import hdbscan

def get_neuromast_stim_times(stim_info, stimulation_id):
    """Get times (in seconds) at which a given neuromast or a combination of neuromasts was stimulated

    Parameters
    ----------
    stim_info : array
        Array with shape (n_stimuli, 2). The columns are: the times (in seconds) at
        which a neuromast was stimulated, column 2 is the neuromast number.
    stimulation_id : int
        Neuromast number in single neuromast stimulation, pattern ID in multi neuromast stimulation.

    Returns
    -------
    stim_times : array
        Array of times (in seconds) at which the given neuromast was stimulated.
    """
    return [nm[0] for nm in stim_info if int(nm[1]) == stimulation_id]


def get_rmap_idx(traces, n_clusters=9, n_PCs=512, locality=0.5, time_lag_window=5, grid_upsample=10):
    """Compute rastermap embedding and sort traces by embedding

    Parameters
    ----------
    traces : array
        Array with shape (n_neurons, n_frames). The fluorescence traces of each neuron.
    n_clusters : int, optional
        Number of clusters to compute. Set to correspond to number of stimuli. Default is 9.
    n_PCs : int, optional
        Number of PCs to use. Default is 512.
    locality : float, optional
        Locality in sorting is low here to get more global sorting (this is a value from 0-1). Default is 0.5.
    time_lag_window : int, optional
        Use future timepoints to compute correlation. Default is 5.

    Returns
    -------
    isort : array
        Array with shape (n_neurons,). The indices of the sorted neurons.
    """

    # Run rastermap
    model = Rastermap(n_clusters=n_clusters,  # number of clusters to compute
                      n_PCs=n_PCs,  # number of PCs to use
                      locality=locality,
                      # locality in sorting is low here to get more global sorting (this is a value from 0-1)
                      time_lag_window=5,  # use future timepoints to compute correlation
                      grid_upsample=grid_upsample,  # default value, 10 is good for large recordings
                      ).fit(traces)
    # Get sorting indices
    isort = model.isort
    # Get cluster labels
    cluster_labels = model.embedding_clust

    return isort, cluster_labels


def mode_robust_fast(inputData, axis=None):
    """ Robust estimator of the mode of a data set using the half-sample mode.
    Based on exceptionality.py from CaImAn

    .. versionadded: 1.0.3

    Parameters
    ----------
    inputData: ndarray
        input data
    axis: int, optional
        axis along which to compute the mode
    """

    if axis is not None:

        def fnc(x):
            return mode_robust_fast(x)

        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
        data = inputData.ravel()
        # The data need to be sorted for this to work
        data = np.sort(data)
        # Find the mode
        dataMode = _hsm(data)

    return dataMode


def mode_robust(inputData, axis=None, dtype=None):
    """ Robust estimator of the mode of a data set using the half-sample mode.
     Based on exceptionality.py from CaImAn

    .. versionadded: 1.0.3
    """
    if axis is not None:

        def fnc(x):
            return mode_robust(x, dtype=dtype)

        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
        def _hsm(data):
            if data.size == 1:
                return data[0]
            elif data.size == 2:
                return data.mean()
            elif data.size == 3:
                i1 = data[1] - data[0]
                i2 = data[2] - data[1]
                if i1 < i2:
                    return data[:2].mean()
                elif i2 > i1:
                    return data[1:].mean()
                else:
                    return data[1]
            else:

                wMin = np.inf
                N = data.size // 2 + data.size % 2

                for i in range(0, N):
                    w = data[i + N - 1] - data[i]
                    if w < wMin:
                        wMin = w
                        j = i

                return _hsm(data[j:j + N])

        data = inputData.ravel()
        if type(data).__name__ == "MaskedArray":
            data = data.compressed()
        if dtype is not None:
            data = data.astype(dtype)

        # The data need to be sorted for this to work
        data = np.sort(data)

        # Find the mode
        dataMode = _hsm(data)

    return dataMode


def compute_event_exceptionality(traces: np.ndarray,
                                 robust_std: bool = False,
                                 N: int = 5,
                                 use_mode_fast: bool = False,
                                 sigma_factor: float = 3.):
    """Define a metric and order components according to the probability of some "exceptional events" (like a spike).

    Such probability is defined as the likelihood of observing the actual trace value over N samples given an estimated noise distribution.
    The function first estimates the noise distribution by considering the dispersion around the mode.
    This is done only using values lower than the mode. The estimation of the noise std is made robust by using the approximation std=iqr/1.349.
    Then, the probability of having N consecutive events is estimated.
    This probability is used to order the components.

    Based on exceptionality.py from CaImAn

    Args:
       Y: ndarray
           movie x,y,t

       A: scipy sparse array
           spatial components

       traces: ndarray
           Fluorescence traces

       N: int
           N number of consecutive events (N must be greater than 0)

       sigma_factor: float
           multiplicative factor for noise estimate (added for backwards compatibility)

    Returns:
       fitness: ndarray
           value estimate of the quality of components (the lesser the better)

       erfc: ndarray
           probability at each time step of observing the N consequtive actual trace values given the distribution of noise

       noise_est: ndarray
           the components ordered according to the fitness

    Usage:
    fitness_raw, _, _, _ = compute_event_exceptionality(traces)
    comp_SNR = -norm.ppf(np.exp(fitness_raw / traces.shape[0]))
    """
    if N == 0:
        # Without this, numpy ranged syntax does not work correctly, and also N=0 is conceptually incoherent
        raise Exception("FATAL: N=0 is not a valid value for compute_event_exceptionality()")

    T = np.shape(traces)[-1]
    if use_mode_fast:
        md = mode_robust_fast(traces, axis=1)
    else:
        md = mode_robust(traces, axis=1)

    ff1 = traces - md[:, None]

    # only consider values under the mode to determine the noise standard deviation
    ff1 = -ff1 * (ff1 < 0)
    if robust_std:

        # compute 25 percentile
        ff1 = np.sort(ff1, axis=1)
        ff1[ff1 == 0] = np.nan
        Ns = np.round(np.sum(ff1 > 0, 1) * .5)
        iqr_h = np.zeros(traces.shape[0])

        for idx, _ in enumerate(ff1):
            iqr_h[idx] = ff1[idx, -Ns[idx]]

        # approximate standard deviation as iqr/1.349
        sd_r = 2 * iqr_h / 1.349

    else:
        Ns = np.sum(ff1 > 0, 1)
        sd_r = np.sqrt(np.sum(ff1 ** 2, 1) / Ns)

    # compute z value
    z = (traces - md[:, None]) / (sigma_factor * sd_r[:, None])

    # probability of observing values larger or equal to z given normal
    # distribution with mean md and std sd_r
    # erf = 1 - norm.cdf(z)

    # use logarithm so that multiplication becomes sum
    # erf = np.log(erf)
    # compute with this numerically stable function
    erf = log_ndtr(-z)

    # moving sum
    erfc = np.cumsum(erf, 1)
    erfc[:, N:] -= erfc[:, :-N]

    # select the maximum value of such probability for each trace
    fitness = np.min(erfc, 1)

    return fitness, erfc, sd_r, md

    # fitness_raw, _, _, _ = compute_event_exceptionality(neurons.T)
    # comp_SNR = -norm.ppf(np.exp(fitness_raw / neurons.shape[-1]))


def compute_gini(rf):
    """Compute Gini coefficient of a receptive field

    Parameters
    ----------
    rf : array
        Array with shape (n_neurons, n_stimuli). The receptive fields of each neuron.

    Returns
    -------
    gini : array
        Array with shape (n_neurons,). The Gini coefficient of each neuron.
    """
    # Get number of neurons
    n_neurons = rf.shape[0]

    # Create empty array to store Gini coefficients
    gini = np.zeros((n_neurons,))

    # Compute Gini coefficient for each neuron
    for i in range(n_neurons):
        # Get receptive field
        rf_i = rf[i, :]
        # Compute Gini coefficient
        gini[i] = np.sum(np.abs(np.subtract.outer(rf_i, rf_i))) / (2 * np.sum(rf_i))

    return gini


def compute_theil_index(traces_avg_trials):
    """Compute Theil index of the average response to each stimulus for each neuron

    Parameters
    ----------
    traces_avg_trials : array
        Array with shape (n_neurons, n_frames, n_stimuli).
        The average traces for each neuron for each stimulus and each trial.

    Returns
    -------
    theil_index : array
        Array with shape (n_neurons, ). The Theil index of the average traces of each neuron.
    """
    # Get number of neurons
    n_neurons = traces_avg_trials.shape[0]
    # Get number of stimuli
    n_stim = traces_avg_trials.shape[-1]

    # Use maximum over time as response
    response = np.max(traces_avg_trials, axis=1)

    # Create empty array to store Theil index
    theil_index = np.zeros((n_neurons,))
    # Compute Theil index for each neuron
    for i in range(n_neurons):
        # Compute Theil index
        theil_index[i] = _theil_index(response[i, :])
    return theil_index


def _theil_index(input_array):
    """
    Calculate the Theil index for a given distribution of values.

    Parameters
    ----------
    input_array : array
        Array with shape (n_values,). The distribution of values.

    Returns
    -------
    theil : float
        The Theil index of the distribution.
    """
    if len(input_array) == 0:
        return 0  # Avoid division by zero

    # Compute max entropy
    max_entropy = np.log(len(input_array))

    # Compute observed entropy
    ratios = input_array / np.sum(input_array)
    inverted_ratios = 1 / ratios

    observed_entropy = np.sum(ratios * np.log(inverted_ratios))

    # Compute Theil index
    theil = max_entropy - observed_entropy

    return theil


def get_peak_time(traces):
    """Get the time of the peak of each trace

    Parameters
    ----------
    traces : array
        Array with shape (n_neurons, n_frames). The fluorescence traces of each neuron.

    Returns
    -------
    peak_time : array
        Array with shape (n_neurons,). The time of the peak of each trace.
    """
    # Get number of neurons
    n_neurons = traces.shape[0]

    # Create empty array to store peak times
    peak_time = np.zeros((n_neurons,))

    # Get peak time for each neuron
    for i in range(n_neurons):
        peak_time[i] = np.argmax(traces[i, :])

    return peak_time

def compute_stim_similarity(traces_stim_avg):
    """Compute the similarity between stimuli for each neuron as
    the Mahalanobis distance between the responses to the stimuli.
    Where high similarity is a low Mahalanobis distance.

    Parameters
    ----------
    traces_stim_avg : array
        Array with shape (n_neurons, n_frames, n_stim). The average traces for each neuron for each stimulus.

    Returns
    -------
    similarities : array
        Array with shape (n_neurons, n_stim). The similarity of a response to one stimulus compared to
        the response to the other stimuli for each neuron.
    """
    # Get number of neurons
    n_neurons = traces_stim_avg.shape[0]
    # Get number of stimuli
    n_stim = traces_stim_avg.shape[1]

    # Compute covariance matrix of full data
    cov = np.cov(traces_stim_avg[:, :, 0].T)

    # Invert
    inv_cov = np.linalg.inv(cov)

    # Create empty array to store similarities
    similarities = np.zeros((n_neurons, n_stim, n_stim))

    # Compute similarity for each neuron
    for i in range(n_neurons):
        for j in range(n_stim):
            for k in range(n_stim):
                # Compute similarity
                similarities[i, j, k] = mahalanobis(traces_stim_avg[i, :, j], traces_stim_avg[i, :, k], inv_cov)

    # Average over stimuli
    similarities = np.mean(similarities, axis=-1)

    return similarities


def hoyer_sparsity(weights):
    """Compute the Hoyer sparsity of the given weights

    Parameters
    ----------
    weights : array
        Array with shape (n_neurons, n_features). The weights of the neurons.

    Returns
    -------
    sparsity : array
        Array with shape (n_neurons,). The Hoyer sparsity of each neuron. If all weights are zero, sparsity is 0.
    """
    # Scale rows to [0, 1]
    weights = (weights - np.min(weights, axis=1)[:, np.newaxis]) / (
            np.max(weights, axis=1)[:, np.newaxis] - np.min(weights, axis=1)[:, np.newaxis] + 1e-12)

    # Compute l1 norm
    l1_norm = np.sum(np.abs(weights), axis=1)
    # Compute square of l2 norm
    l2_norm = np.sqrt(np.sum(weights ** 2, axis=1))
    n = weights.shape[1]

    # Compute sparsity
    sparsity = (np.sqrt(n) - (l1_norm / l2_norm)) / (np.sqrt(n) - 1)
    if np.isnan(sparsity).any():
        sparsity[np.isnan(sparsity)] = 0

    return sparsity


def compute_fscore(confusion):
    """Compute the F1 score for each class

    Parameters
    ----------
    confusion : array
        Array with shape (n_classes, n_classes). The confusion matrix.
        Rows are true labels and columns are predicted labels.

    Returns
    -------
    f1_score : array
        Array with shape (n_classes,). The F1 score for each class.
    """
    # True Positives are the diagonal elements
    TP = np.diag(confusion)

    # False Positives are the sum of the columns minus the TP
    FP = np.sum(confusion, axis=0) - TP

    # False Negatives are the sum of the rows minus the TP
    FN = np.sum(confusion, axis=1) - TP

    # Precision
    precision = TP / (TP + FP)

    # Recall
    recall = TP / (TP + FN)

    # F1 score
    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score


def compute_margin_size(weights):
    """Compute the margin size of the given weights

    Parameters
    ----------
    weights : array
        Array with shape (n_neurons, n_features). The weights of the neurons.

    Returns
    -------
    margin_size : array
        Array with shape (n_neurons,). The margin size of each neuron.
    """
    # Compute l2 norm
    l2_norm = np.sqrt(np.sum(weights ** 2, axis=1))

    # Compute margin size
    margin_size = 2 / l2_norm

    return margin_size


def sort_responses(traces_trials, idx_to_plot):
    """Sort responses by clustering

    Parameters
    ----------
    traces_trials : array
        Array with shape (n_neurons, n_samples, n_trials, n_stimuli).
        The fluorescence traces for each neuron for each trial and each stimulus for a given number of samples
        post-stimulus onset.
    idx_to_plot : array
        Array with shape (n_neurons,). The indices of the neurons to plot.

    Returns
    -------
    idx_sorted : array
        Array with shape (n_neurons,). The indices of the neurons sorted by clustering. This will index into
        the original idx_to_plot array provided.
    """

    # Average over trials and concatenate stimuli
    traces_trials_avg = np.mean(traces_trials[idx_to_plot, :, :, :], axis=-2)

    traces_trials_avg_stim_concat = []
    for i in range(traces_trials_avg.shape[-1]):
        traces_trials_avg_stim_concat.append(traces_trials_avg[:, :, i])
    traces_trials_avg_stim_concat = np.concatenate(traces_trials_avg_stim_concat, axis=-1)

    # Cluster concatenated traces with rastermap
    model = Rastermap(n_clusters=3, n_PCs=10, locality=0.5, time_lag_window=3, grid_upsample=10).fit(
        traces_trials_avg_stim_concat)

    idx_sorted = model.isort

    return idx_sorted