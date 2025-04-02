import matplotlib.pyplot as plt
import scipy.io as io
import numpy as np
# Set matplotlib backend to Qt5Agg
plt.switch_backend('Qt5Agg')
import seaborn as sns
# Change to default matplotlib style only implied if this file is imported as a module
# Set font size
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
# Set standard figure size
plt.rcParams['figure.figsize'] = [18.3 / 2.54, 9 / 2.54]
# Set title padding
plt.rcParams['axes.titlepad'] = 16.8

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from timeseries import avg_responsive_cells, create_timeseries
from utils import get_neuromast_stim_times


def plot_traces(traces, neurons=None, axs=None):
    """Plot the fluorescence traces of a given neuromast

    Parameters
    ----------
    traces : array
        Array with shape (n_neurons, n_frames). The fluorescence traces of each neuron.
    neurons : array-like, optional
        List of neuron indices to plot. If None, first neuron is plotted.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot the traces. If None, a new figure and axes are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure on which the traces were plotted.
    ax : matplotlib.axes.Axes
        Axes on which the traces were plotted.

    """
    # Create figure and axes
    if axs is None:
        if neurons is None:
            fig, axs = plt.subplots(1, 1)
        else:
            n_rows = len(neurons)
            fig, axs = plt.subplots(n_rows, 1, sharex=True)
            axs = axs.ravel()
    else:
         # Check if number of neurons corresponds to number of subplots
        if len(axs) != len(neurons):
            raise ValueError('Number of neurons does not correspond to number of subplots')
        fig = axs[0].figure

    # Plot traces
    if neurons is None:
        # Create subplots
        axs.plot(traces[0, :])
    else:
        for neuron, ax in zip(neurons, axs):
            ax.plot(traces[neuron, :])

    return fig, axs


def plot_stimulus_overlay(traces, stim_info, fs, neuron_idx=None, stim_id=None):
    """Plot a trace with overlays indicating the times at which the individual stimuli were presented.

    Parameters
    ----------
    traces : array
        Array with shape (n_neurons, n_frames). The fluorescence traces of each neuron.
    stim_info : array
        Array with shape (n_stim, 2). The times at which the stimuli were presented.
    fs : float
        Sampling frequency of the traces.
    neuron_idx : int, optional
        Index of the neuron to plot. If None, first neuron is plotted.
    stim_id : int, optional


    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure on which the trace was plotted.
    ax : matplotlib.axes.Axes
        Axes on which the trace was plotted.
    """
    # Check if neuron_idx is valid
    if neuron_idx is not None:
        if len(neuron_idx) != 1:
            raise ValueError('neuron_idx must be a single integer. Only one neuron can be plotted at a time')

    # Create figure and axes
    fig, ax = plt.subplots(1, 1)

    # Plot trace
    if neuron_idx is None:
        ax.plot(traces[0, :])
    else:
        ax.plot(traces[neuron_idx, :])

    # Get number of stimuli
    n_stim = len(np.unique(stim_info[:, 1]))

    # Create timeseries for each stimulus
    timeseries_all = []
    for i in range(1, n_stim + 1):
        stim_times = get_neuromast_stim_times(stim_info, i)
        timeseries = create_timeseries(stim_times, fs, duration=traces.shape[1] / fs, single_ticks=True)
        timeseries_all.append(timeseries)

    # Make into array
    timeseries_all = np.array(timeseries_all)

    # Qualitative colormap for stimuli
    cmap = sns.color_palette()

    # Plot vertical line whenever stimulus is 1
    for i in range(n_stim):
        for j in range(len(timeseries_all[i])):
            if timeseries_all[i][j] == 1:
                ax.axvline(j, color=cmap[i], linestyle='-', alpha=0.5)

    return fig, ax


def plot_heatmap(traces, fs, stimulus_times=None, **kwargs):
    """Plot the heatmap of the fluorescence traces of a given neuromast

    Parameters
    ----------
    traces : array
        Array with shape (n_neurons, n_frames). The fluorescence traces of each neuron.
    fs : float
        Sampling frequency of the traces.
    stimulus_times : array, optional
        Array with shape (n_frames, ). Indicates the times at which the stimulus was presented.
        Used to plot vertical lines at stimulus times.
        If not provided, no vertical lines are plotted
    **kwargs : dict
        Keyword arguments passed to matplotlib.pyplot.imshow

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure on which the heatmap was plotted.
    ax : matplotlib.axes.Axes
        Axes on which the heatmap was plotted.

    """
    # Get extent of heatmap
    extent = [0, traces.shape[1] / fs, 0, traces.shape[0]]

    # Make sure traces and stimulus_times are same size
    if stimulus_times is not None:
        if traces.shape[1] != stimulus_times.shape[0]:
            raise ValueError('Number of frames in traces and stimulus_times must be the same')

    # Create figure and axes
    fig, ax = plt.subplots(1, 1)

    # # Set vmax and vmin of heatmap to 95th and 5th percentile of traces
    # vmax = np.percentile(traces, 95)
    # vmin = np.percentile(traces, 5)

    # Plot heatmap
    ax.imshow(traces, aspect='auto', extent=extent, **kwargs)

    # Plot vertical lines at stimulus times
    if stimulus_times is not None:
        prev_tick = 0
        for i, time_point in enumerate(stimulus_times):
            if time_point == 1:
                if i == 0: # Offset a little to visualize first line better (occluded by y-axis)
                    ax.axvline(i / fs + 0.1, color='r', linestyle='--')
                else:
                    if i - prev_tick > 2: # Only draw line if more than 2 frames have passed
                        ax.axvline(i / fs, color='r', linestyle='--')
            prev_tick = i

    return fig, ax


def plot_significant_positions(positions, significant_positions, volume=True):
    """Plot the positions of a given set of neurons

    Parameters
    ----------
    positions : array
        Array with shape (n_neurons, 3). The positions of each neuron.
    significant_positions : array
        Array with shape (n_significant_neurons, 3). The positions of each significant neuron.
    volume : bool, optional
        Whether to plot the positions in 3d or not. If False, only the x and y coordinates are plotted.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure on which the positions were plotted.
    ax : matplotlib.axes.Axes
        Axes on which the positions were plotted.

    """
    # Create figure and axes
    fig, ax = plt.subplots(1, 1)

    # Plot positions of all neurons, with low opacity
    if volume:
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], alpha=0.1)
    else:
        ax.scatter(positions[:, 0], positions[:, 1], alpha=0.1)

    # Plot positions of significant neurons
    if volume:
        ax.scatter(significant_positions[:, 0], significant_positions[:, 1], significant_positions[:, 2])
    else:
        ax.scatter(significant_positions[:, 0], significant_positions[:, 1])

    # Add labels
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')
    if volume:
        ax.set_zlabel('z (um)')

    # Add legend
    ax.legend(['All neurons', 'Significant neurons'])

    return fig, ax


def plot_significant_positions_plotly(positions, significant_idx):
    """Plot the positions of a given set of neurons in 3d using plotly

    Parameters
    ----------
    positions : array
        Array with shape (n_neurons, 3). The positions of each neuron.
    significant_idx : dict
        Dictionary with keys being the stimulus ids and
        each key contains an array with shape (n_significant_neurons, 3). The positions of each significant neuron.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figure on which the positions were plotted.

    """
    # Create figure
    fig = go.Figure()

    # Plot positions of subsampled neurons, with low opacity and small markers (subsampled to speed up plotting)
    positions_subs = positions[::5, :]
    fig.add_trace(go.Scatter3d(
        x=positions_subs[:, 0], y=positions_subs[:, 1], z=positions_subs[:, 2],
        mode='markers', marker=dict(size=2, opacity=0.1),
        name='All neurons'))

    # Colorvalues spanning discrete colormap
    colorvalues = np.linspace(0, 256, len(significant_idx.keys()))
    # Create colormap
    colormap = plt.get_cmap('Set1')

    for i, key in enumerate(significant_idx.keys()):
        # Plot positions of significant neurons and colour markers by index in significant neurons
        # Index centroids based on significant neurons mask
        significant_positions = positions[significant_idx[key], :]
        fig.add_trace(go.Scatter3d(
            x=significant_positions[:, 0], y=significant_positions[:, 1], z=significant_positions[:, 2],
            mode='markers', marker=dict(size=3, opacity=1, color=colorvalues[i], colorscale='Agsunset'),
            name=f'Stimulus {key}'))

    # Add labels
    fig.update_layout(scene=dict(xaxis_title='x (um)', yaxis_title='y (um)', zaxis_title='z (um)'))

    # Add colorbar
    fig.update_layout(coloraxis_colorbar=dict(title='Neuron index'))

    # Increase size of markers in legend
    fig.update_layout(legend=dict(itemsizing='constant'))

    return fig


def plot_interactive_rastermap(traces, positions, fs=2.18, **kwargs):
    """Plot an interactive rastermap of the fluorescence traces of a given neuromast

    Parameters
    ----------
    traces : array
        Array with shape (n_neurons, n_frames). The fluorescence traces of each neuron.
    positions : array
        Array with shape (n_neurons, 3). The positions of each neuron.
    fs : float, optional
        Sampling frequency of the traces.
    **kwargs : dict
        Keyword arguments passed to plotly.graph_objects.Figure.update_layout

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figure on which the rastermap was plotted.

    """
    # Create figure
    fig = go.Figure()

    # Create traces
    for i in range(traces.shape[0]):
        fig.add_trace(go.Scatter(x=np.arange(traces.shape[1]) / fs, y=traces[i, :], mode='lines'))

    # Add labels
    fig.update_layout(xaxis_title='Time (s)', yaxis_title='Fluorescence (a.u.)', **kwargs)

    return fig


def plot_clusters_plotly(positions, traces, significant_idx, fs=2.18):
    """Plot the positions of a given set of neurons in 3d and highlight the clusters responding
    to a given stimulus. Also plots the mean activity of each cluster.

    Parameters
    ----------
    positions : array
        Array with shape (n_neurons, 3). The positions of each neuron.
    traces : array
        Array with shape (n_neurons, n_frames). The fluorescence traces of each neuron.
    significant_idx : array
        Dictionary with keys being the stimulus ids and
        each key contains an array with shape (n_significant_neurons, 3). The positions of each significant neuron.
    fs : float, optional
        Sampling frequency of the traces.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figure on which the positions were plotted.

    """
    # Create figure with subplots
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}]])

    # Plot positions of subsampled neurons, with low opacity and small markers (subsampled to speed up plotting)
    positions_subs = positions[::5, :]
    fig.add_trace(go.Scatter3d(
        x=positions_subs[:, 0], y=positions_subs[:, 1], z=positions_subs[:, 2],
        mode='markers', marker=dict(size=2, opacity=0.1),
        name='All neurons'))

    # Colorvalues spanning discrete colormap
    colorvalues = np.linspace(0, 256, len(significant_idx.keys()))

    # Colours for colormap
    colors = px.colors.qualitative.Dark24

    for i, key in enumerate(significant_idx.keys()):
        # Plot positions of significant neurons and colour markers by index in significant neurons
        # Index centroids based on significant neurons mask
        # Plot in magma colorscale
        significant_positions = positions[significant_idx[key], :]
        fig.add_trace(go.Scatter3d(
            x=significant_positions[:, 0], y=significant_positions[:, 1], z=significant_positions[:, 2],
            mode='markers', marker=dict(size=3, color=colors[i], opacity=1),
            name=f'Stimulus {key}'),
            row=1, col=1)

    # Add labels
    fig.update_layout(scene=dict(xaxis_title='x (um)', yaxis_title='y (um)', zaxis_title='z (um)'))

    # Add colorbar
    fig.update_layout(coloraxis_colorbar=dict(title='Neuron index'))

    # Increase size of markers in legend
    fig.update_layout(legend=dict(itemsizing='constant'))

    # Plot mean activity of each cluster, link selection to scatterplot
    for i, key in enumerate(significant_idx.keys()):
        # Plot mean activity of each cluster
        mean_trace = avg_responsive_cells(traces, significant_idx[key])
        # Remove values below mean
        mean_trace[mean_trace < np.mean(mean_trace)] = np.mean(mean_trace)
        fig.add_trace(go.Scatter(x=np.arange(len(mean_trace)) / fs, y=mean_trace, mode='lines', name=f'Stimulus {key}',
                                 line=dict(color=colors[i])),
                      row=1, col=2)
    # Add axis labels
    fig.update_layout(xaxis_title='Time (s)', yaxis_title='Fluorescence (a.u.)')

    return fig


def plot_trial_mosaic(traces_stim, idx_to_plot, fs, visual_control=True, single_traces=False, nm_identities=None):
    """Plot the average response to each stimulus for a given set of neurons

    Parameters
    ----------
    traces_stim : array
        Array with shape (n_neurons, n_frames, n_stim). The average response to each stimulus for each neuron.
        If single_traces is True, the shape is (n_neurons, n_samples, n_trials, n_stim).
    idx_to_plot : array-like
        List of neuron indices to plot.
    fs : float
        Sampling frequency of the traces.
    visual_control : bool, optional
        Whether the last stimulus ID is a visual control or not.
        If True, the last stimulus is labelled as visual control.
    single_traces : bool, optional
        Whether to plot each individual trace or not. False by default, plots the average response.
    nm_identities : array-like, optional
        List of neuromast identities for each neuron. If provided, the neuromast identity is plotted on the column labels
        otherwise the columns are labeled as NM 1, NM 2, etc.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure on which the average response to each stimulus was plotted.
    ax : array
        Axes on which the average response to each stimulus was plotted.
    """
    # Check if traces_stim is not averaged if single_traces is True
    if single_traces and len(traces_stim.shape) < 4:
        raise ValueError('first argument must not be averaged over trials to plot single traces')
    elif not single_traces and len(traces_stim.shape) > 3:
        raise ValueError('first argument must be averaged over trials to plot average response')

    # Determine number of neurons to plot
    n_plots = len(idx_to_plot)

    # Determine stimulus time in frames
    n_stim_frames = traces_stim.shape[1]

    # Determine number of stimuli
    n_stim = traces_stim.shape[-1]

    # Create time vector for plotting
    time_vect = np.arange(0, n_stim_frames / fs, 1 / fs)

    # Create figure
    cm = 1 / 2.54  # centimeters in inches
    fig, ax = plt.subplots(n_plots, n_stim, figsize=(18.3 * cm, 22.3 * cm), layout='tight')

    # Expand dimensions of ax array if only one neuron or only one stimulus
    if n_plots == 1:
        ax = np.expand_dims(ax, axis=0)
    if n_stim == 1:
        ax = np.expand_dims(ax, axis=1)

    # Set title
    fig.suptitle(f'Average response to each stimulus\nfor {n_plots} neurons')

    # Increase horizontal spacing between subplots
    fig.subplots_adjust(hspace=0.2, wspace=8.2)

    # Loop through subplots
    for i, neuron_id in enumerate(idx_to_plot):
        for j in range(n_stim):
            # Draw line plot
            if single_traces:
                # Plot single traces
                for k in range(traces_stim.shape[2]):
                    ax[i, j].plot(time_vect[:n_stim_frames-int(n_stim_frames * 0.3)],
                                  traces_stim[neuron_id, :n_stim_frames-int(n_stim_frames * 0.3), k, j], alpha=0.05,
                                  color=sns.color_palette('Blues', n_stim)[-1], linewidth=1.5)
                # Plot mean over individual traces
                ax[i, j].plot(time_vect[:n_stim_frames-int(n_stim_frames * 0.3)],
                              np.nanmean(traces_stim[neuron_id, :n_stim_frames-int(n_stim_frames * 0.3), :, j], axis=1),
                              alpha=1, color=sns.light_palette('seagreen', n_stim)[-1], linewidth=2.5)
            else:
                ax[i, j].plot(time_vect[:n_stim_frames-int(n_stim_frames * 0.3)],
                              traces_stim[neuron_id, :n_stim_frames-int(n_stim_frames * 0.3), j],
                              color=sns.light_palette('seagreen', n_stim)[-1], linewidth=2.5)

            # Line styling
            ax[i, j].lines[0].set_solid_capstyle('round')

            # Border styling
            ax[i, j].spines['top'].set_visible(False)
            ax[i, j].spines['right'].set_visible(False)
            ax[i, j].spines['left'].set_visible(False)
            ax[i, j].spines['bottom'].set_visible(False)
            ax[i, j].xaxis.set_ticks_position('none')
            ax[i, j].yaxis.set_ticks_position('none')
            ax[i, j].tick_params(axis='both', which='both', length=0)

            # Titles for columns
            if nm_identities is not None:
                # Only label first row
                if i == 0:
                    try:
                        ax[i, j].set_title(f'{nm_identities[j]}')
                    except IndexError:
                        if visual_control:
                            # Label last one as visual control
                            if j == n_stim - 1:
                                ax[i, j].set_title('Visual control')
                        else:
                            raise IndexError
            else:
                # Only label first row
                if i == 0:
                    ax[i, j].set_title(f'NM {j + 1}')

            # Axes styling
            ax[i, j].set_yticks([])
            ax[i, j].set_ylabel('')
            ax[i, j].set_xticks([])
            ax[i, j].set_xlabel('')

            # Constant y axis
            ax[i, j].set_ylim([np.nanmin(traces_stim[idx_to_plot, :, :])-0.2,
                               np.nanmax(traces_stim[idx_to_plot, :, :]) * 1.2])

            if visual_control:
                # Label last one as visual control
                if (i == 0) and (j == n_stim - 1):
                    # Span two lines
                    ax[i, j].set_title('Visual\ncontrol')
            else:
                if (i == n_plots - 1) and (j == n_stim - 1):
                    ax[i, j].set_title(f'NM {j + 1}')

    return fig, ax


def plot_svm_results(confusion, lifetime_sparsity, population_sparsity, weights, subplot_labels=['A', 'B', 'C', 'D']):
    """Plot the results of the SVM decoder
    Plots (1) the confusion matrix, (2) the lifetime sparsity histogram, (3) the population sparsity histogram,
    and (4) the weights of the SVM decoder.

    Parameters
    ----------
    confusion : array
        Array with shape (n_stim, n_stim). The confusion matrix.
    lifetime_sparsity : array
        Array with shape (n_neurons, ). The lifetime sparsity of each neuron.
    population_sparsity : array
        Array with shape (n_neurons, ). The population sparsity of each neuron.
    weights : array
        Array with shape (n_neurons, n_stim). The weights of the SVM decoder.
    subplot_labels : array, optional
        List of subplot labels.
    """
    weights_sort = np.argsort(np.max(np.abs(weights), axis=0), axis=0)
    weights_sort = np.flip(weights_sort)
    bins = np.linspace(0, 1, 20)

    cm = 1 / 2.54  # centimeters in inches
    fig = plt.figure(constrained_layout=True, figsize=(18.3 * cm, 9 * cm))
    axd = fig.subplot_mosaic(
        """
        ABC
        DDD
        """
    )

    sns.heatmap(confusion, ax=axd['A'], cmap='Blues',
                cbar=True, cbar_kws={'pad': 0, "boundaries": np.linspace(0, 1, 100), 'ticks': [0, 1]})
    sns.histplot(lifetime_sparsity, ax=axd['B'], kde=False, bins=bins, color='seagreen')
    sns.histplot(population_sparsity, ax=axd['C'], kde=False, bins=bins, color='seagreen')
    sns.heatmap(np.abs(weights[:, weights_sort]), ax=axd['D'],
                cmap=sns.light_palette("seagreen", as_cmap=True), cbar=True, cbar_kws={'pad': 0.01},
                xticklabels=False, yticklabels=False)

    # Set titles for subplots
    for i, key in enumerate(axd.keys()):
        axd[key].set_title(subplot_labels[i], weight='bold', loc='left')

    # Axis labels
    axd['A'].set_xlabel('Predicted')
    axd['A'].set_ylabel('True')
    axd['B'].set_xlabel('Lifetime sparsity')
    axd['B'].set_ylabel('Frequency')
    axd['C'].set_xlabel('Population sparsity')
    axd['C'].set_ylabel(None)
    axd['D'].set_xlabel('Neuron index')
    axd['D'].set_ylabel('Neuromast \nstimulated')

    # Set ticks for histograms
    axd['B'].set_xticks([0, 0.5, 1])
    axd['C'].set_xticks([0, 0.5, 1])

    # Get current y-axis ticks fo ax D
    yticks = axd['C'].get_yticks()
    # Get range
    min_tick = np.min(yticks)
    max_tick = np.floor(np.max(yticks))

    # Set y-axis ticks for ax D
    axd['C'].set_yticks(np.arange(min_tick, max_tick + 1, 1))

    # Despine histograms with offset spines
    axd['B'].spines['left'].set_position(('outward', 10))
    axd['C'].spines['left'].set_position(('outward', 10))
    axd['B'].spines['top'].set_visible(False)
    axd['C'].spines['top'].set_visible(False)
    axd['B'].spines['right'].set_visible(False)
    axd['C'].spines['right'].set_visible(False)

    return fig, axd


def plot_region_sparsity_comparison(df, super_region=True, sparsity_measure='raw', subplot_labels='A'):
    """Plot the sparsity of each region in the brain

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns 'region', 'super_region', and 'sparsity'. Rows are individual neurons.
    super_region : bool, optional
        Whether to plot the super region or not. If True, the super region is plotted, otherwise the region is plotted.
    sparsity_measure : str, optional
        Measure of sparsity to plot. Can be 'raw' or 'weights'.
        Raw sparsity is the sparsity of the fluorescence responses of each neuron for the different stimuli.
        Weight sparsity is the sparsity of the weights of the SVM decoder.
    subplot_labels : str, optional
        Label for subplot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure on which the sparsity of each region was plotted.
    ax : matplotlib.axes.Axes
        Axes on which the sparsity of each region was plotted.

    """
    # Set colour palette
    if super_region:
        sns.set_palette('crest', n_colors=df.super_region.nunique())
    else:
        sns.set_palette('crest', n_colors=df.region.nunique())

    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(18.3 / 2.54 /2, 9 / 2.54), constrained_layout=True)

    # Plot sparsity of each region
    if super_region:
        if sparsity_measure == 'weights':
            # Handle NaNs
            df = df.loc[~df.weight_sparsity.isna()]
            sns.boxplot(x='super_region', y='weight_sparsity', data=df, ax=ax)
        elif sparsity_measure == 'raw':
            sns.boxplot(x='super_region', y='raw_sparsity', data=df, ax=ax)
        else:
            raise ValueError('sparsity_measure must be "raw" or "weights"')
    else:
        if sparsity_measure == 'weights':
            # Only select rows for which weight_sparsity is not NaN
            df = df.loc[~df.weight_sparsity.isna()]
            sns.boxplot(x='region', y='weight_sparsity', data=df, ax=ax)
        elif sparsity_measure == 'raw':
            sns.boxplot(x='region', y='raw_sparsity', data=df, ax=ax)
        else:
            raise ValueError('sparsity_measure must be "raw" or "weights"')

    # Set labels
    ax.set_ylabel('Sparsity')
    ax.set_xlabel('')

    # Set title
    ax.set_title(subplot_labels, weight='bold', loc='left')

    # Clean x-axis labels (replace '_' with ' ')
    ax.set_xticklabels([label.get_text().replace('_', '\n') for label in ax.get_xticklabels()])

    return fig, ax


# Deprecate this in favour of plot_spatial_neurons() with option to plot sparsity
def plot_spatial_sparsity(region_df, plane='horizontal', sparsity_measure='raw', opacity_scaler=8, bg=[],
                          subplot_labels='A', colorbar=True, flip_y=True):
    """Plot the spatial distribution of sparsity in the brain

    Parameters
    ----------
    region_df : pandas.DataFrame
        DataFrame with columns 'x', 'y', 'raw_sparsity'. Rows are individual neurons.
    plane : str, optional
        Plane to plot. Can be 'horizontal', 'coronal' or 'sagittal'. Default is 'horizontal'.
    sparsity_measure : str, optional
        Measure of sparsity to plot. Can be 'raw' or 'weights'.
        Raw sparsity is the sparsity of the fluorescence responses of each neuron for the different stimuli.
        Weight sparsity is the sparsity of the weights of the SVM decoder.
    opacity_scaler : float, optional
        Exponent to scale opacity of points by. Higher values make points with lower sparsity more transparent.
    bg : array, optional
        Background image to plot behind of the spatial distribution of sparsity.
    subplot_labels : str, optional
        Label for subplot.
    colorbar : bool, optional
        Whether to add a colorbar or not.
    flip_y : bool, optional
        Whether to flip the y-axis or not. If true, origin of the ordinate is in the top-left corner.
        If false, origin is in the bottom-left.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure on which the spatial distribution of sparsity was plotted.
    ax : matplotlib.axes.Axes
        Axes on which the spatial distribution of sparsity was plotted.
    """

    # Create figure and axes
    cm = 1 / 2.54  # centimeters in inches
    fig, ax = plt.subplots(1, 1, figsize=(18.3*cm / 2 , 18.3*cm / 1.68), constrained_layout=True)
    opacity = (region_df.raw_sparsity ** opacity_scaler - np.min(region_df.raw_sparsity ** opacity_scaler)) / \
                (np.max(region_df.raw_sparsity ** opacity_scaler) - np.min(region_df.raw_sparsity ** opacity_scaler))

    # Plot background image
    palette = sns.color_palette("RdPu", as_cmap=False)

    # Add white to the beginning of the palette
    modified_palette = [(1, 1, 1)] + palette  # (1, 1, 1) is RGB for white

    # Convert the modified palette to a colormap
    cmap = sns.blend_palette(modified_palette, as_cmap=True)
    if len(bg) > 0:
        ax.imshow(bg, cmap=cmap, alpha=0.36)
        if flip_y:
            ax.invert_yaxis()

    # Plot neurons in chosen plane and color by sparsity
    if plane == 'horizontal':
        abcsissa = region_df.x
        ordinate = region_df.y
    elif plane == 'coronal':
        abcsissa = region_df.x
        ordinate = region_df.z
    elif plane == 'sagittal':
        abcsissa = region_df.y
        ordinate = region_df.z
    else:
        raise ValueError('plane must be "horizontal", "coronal" or "sagittal"')

    if sparsity_measure == 'raw':
        sc = ax.scatter(abcsissa, ordinate, c=region_df.raw_sparsity, alpha=opacity, cmap='crest_r')
    elif sparsity_measure == 'weights':
        sc = ax.scatter(abcsissa, ordinate, c=region_df.weight_sparsity, alpha=opacity, cmap='crest_r')
    else:
        raise ValueError('sparsity_measure must be "raw" or "weights"')

    # Remove borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Remove edge color and opacity
    sc.set_edgecolor('none')

    # Control size of dots
    sc.set_sizes([12])

    # Set title
    ax.set_title(subplot_labels, weight='bold', loc='left')

    if colorbar:
        # Add colorbar
        fig.colorbar(sc, ax=ax, label='Sparsity', fraction=0.046, pad=0.1)
        # Force to 0,1
        sc.set_clim(0, 1)

    # Flip y
    if flip_y:
        ax.invert_yaxis()

    # Axis labels
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')

    return fig, ax

# Deprecate this in favour of plot_spatial_neurons() with option to plot weights
def plot_spatial_weights(df, plane='horizontal', opacity_scaler=8, bg=[], subplot_labels='A', colorbar=True,
                         flip_y=True):
    """Plot the spatial distribution of weights in the brain

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns 'x', 'y', 'weights'. Rows are individual neurons. 'weights' is assumed to be the
        sum of the absolute weights of the SVM decoder for each neuron.
    plane : str, optional
        Plane to plot. Can be 'horizontal', 'coronal' or 'sagittal'. Default is 'horizontal'.
    opacity_scaler : float, optional
        Exponent to scale opacity of points by. Higher values make points with lower weights more transparent.
    bg : array, optional
        Background image to plot behind of the spatial distribution of weights.
    subplot_labels : str, optional
        Label for subplot.
    colorbar : bool, optional
        Whether to add a colorbar or not.
    flip_y : bool, optional
        Whether to flip the y-axis or not. If true, origin of the ordinate is in the top-left corner.
        If false, origin is in the bottom-left.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure on which the spatial distribution of weights was plotted.
    ax : matplotlib.axes.Axes
        Axes on which the spatial distribution of weights was plotted.
    """

    # Create figure and axes
    cm = 1 / 2.54  # centimeters in inches
    fig, ax = plt.subplots(1, 1, figsize=(18.3*cm / 2 , 18.3*cm / 1.68), constrained_layout=True)
    opacity = (df.weights ** opacity_scaler - np.min(df.weights ** opacity_scaler)) / \
                (np.max(df.weights ** opacity_scaler) - np.min(df.weights ** opacity_scaler))

    # Plot background image
    palette = sns.color_palette("RdPu", as_cmap=False)

    # Add white to the beginning of the palette
    modified_palette = [(1, 1, 1)] + palette  # (1, 1, 1) is RGB for white

    # Convert the modified palette to a colormap
    cmap = sns.blend_palette(modified_palette, as_cmap=True)
    if len(bg) > 0:
        ax.imshow(bg, cmap=cmap, alpha=0.36)
        if flip_y:
            ax.invert_yaxis()

    # Plot neurons in chosen plane and color by weights
    if plane == 'horizontal':
        abcsissa = df.x
        ordinate = df.y
    elif plane == 'coronal':
        abcsissa = df.x
        ordinate = df.z
    elif plane == 'sagittal':
        abcsissa = df.y
        ordinate = df.z
    else:
        raise ValueError('plane must be "horizontal", "coronal" or "sagittal"')

    sc = ax.scatter(abcsissa, ordinate, c=df.weights, alpha=opacity, cmap='crest_r')

    # Remove borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Remove edge color and opacity
    sc.set_edgecolor('none')

    # Control size of dots
    sc.set_sizes([12])

    # Set title
    ax.set_title(subplot_labels, weight='bold', loc='left')

    if colorbar:
        # Add colorbar
        fig.colorbar(sc, ax=ax, label='Weights', fraction=0.046, pad=0.1)
        # Force to 0,1
        sc.set_clim(0, 1)

    # Flip y
    if flip_y:
        ax.invert_yaxis()

    # Axis labels
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')

    return fig, ax


def plot_spatial_neuron(df, quantity='weights', plane='horizontal', opacity_scaler=8, bg=[], subplot_labels='A', colorbar=True,
                        flip_y=True, sparsity=False, weights=False):
    """Plot the spatial distribution of neurons in the brain

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns 'x', 'y', 'weights', 'raw_sparsity'. Rows are individual neurons.
        Coordinates are assumed to be relative to the Zbrain atlas. Expected to have columns for intensity to
        associate with neurons, either 'weights', 'sparsity' or 'raw_sparsity'.
    quantity : str, optional
        Quantity to plot. Can be 'weights', 'sparsity' or 'raw_sparsity'. Default is 'weights'.
    plane : str, optional
        Plane to plot. Can be 'horizontal', 'coronal' or 'sagittal'. Default is 'horizontal'.
    opacity_scaler : float, optional
        Exponent to scale opacity of points by. Higher values make points with lower weights more transparent.
    bg : array, optional
        Background image to plot behind of the spatial distribution of weights.
    subplot_labels : str, optional
        Label for subplot.
    colorbar : bool, optional
        Whether to add a colorbar or not.
    flip_y : bool, optional
        Whether to flip the y-axis or not. If true, origin of the ordinate is in the top-left corner.
        If false, origin is in the bottom-left.
    sparsity : bool, optional
        Whether to plot the sparsity of each neuron or not.
    weights : bool, optional
        Whether to plot the weights of each neuron or not.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure on which the spatial distribution of neurons was plotted.
    ax : matplotlib.axes.Axes
        Axes on which the spatial distribution of neurons was plotted.
    """

    # Create figure and axes
    cm = 1 / 2.54  # centimeters in inches
    fig, ax = plt.subplots(1, 1, figsize=(18.3*cm / 2 , 18.3*cm / 1.68), constrained_layout=True)
    opacity = (df.weights ** opacity_scaler - np.min(df.weights ** opacity_scaler)) / \
                (np.max(df.weights ** opacity_scaler) - np.min(df.weights ** opacity_scaler))

    # Plot background image
    palette = sns.color_palette("RdPu", as_cmap=False)

    # Add white to the beginning of the palette
    modified_palette = [(1, 1, 1)] + palette  # (1, 1, 1) is RGB for white

    # Convert the modified palette to a colormap
    cmap = sns.blend_palette(modified_palette, as_cmap=True)
    if len(bg) > 0:
        ax.imshow(bg, cmap=cmap, alpha=0.36)
        if flip_y:
            ax.invert_yaxis()

    # Plot neurons in chosen plane and color by weights
    if plane == 'horizontal':
        abcsissa = df.x
        ordinate = df.y
    elif plane == 'coronal':
        abcsissa = df.x
        ordinate = df.z
    elif plane == 'sagittal':
        abcsissa = df.y
        ordinate = df.z
    else:
        raise ValueError('plane must be "horizontal", "coronal" or "sagittal')

    if quantity == 'weights':
        sc = ax.scatter(abcsissa, ordinate, c=df.weights, alpha=opacity, cmap='crest_r')
    elif quantity == 'sparsity':
        sc = ax.scatter(abcsissa, ordinate, c=df.raw_sparsity, alpha=opacity, cmap='crest_r')
    elif quantity == 'raw_sparsity':
        sc = ax.scatter(abcsissa, ordinate, c=df.raw_sparsity, alpha=opacity, cmap='crest_r')
    else:
        raise ValueError('quantity must be "weights", "sparsity" or "raw_sparsity"')

    # Remove borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Remove edge color and opacity
    sc.set_edgecolor('none')

    # Control size of dots
    sc.set_sizes([12])

    # Set title
    ax.set_title(subplot_labels, weight='bold', loc='left')

    if colorbar:
        # Add colorbar
        fig.colorbar(sc, ax=ax, label=quantity, fraction=0.046, pad=0.1)
        # Force to 0,1
        sc.set_clim(0, 1)

    # Flip y
    if flip_y:
        ax.invert_yaxis()

    # Axis labels
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')

    return fig, ax

