import os
from plotting import *
import numpy as np
import re
from experiment_io import load_traces, load_positions
from timeseries import preprocess_traces

# Define working directory
wd = '/mnt/bronknas/Singles/Data_Singles_To_Analyze/Data_singles_2/020823/Fish_K'

# Load neural data
traces = load_traces(os.path.join(wd, 'merged', 'merged_raw.pkl'))

# Preprocess traces
traces_filtered = preprocess_traces(traces, fs=2.18, kernel_decay=5, bleaching_tau=200, zscore=False, filter=True)

# Load centroids (merged_centroids.pkl)
centroids_path = os.path.join(wd, 'merged', 'merged_centroids.pkl')
positions = load_positions(centroids_path)

# Load granger causality values
# List granger files in directory
granger_files = [f for f in os.listdir(wd) if ('nm' in f) and ('granger' in f)]

# Key by nm\d{1} substring
granger_dict = {}
for f in granger_files:
    key = re.search('nm\d{1}', f).group()

    # Load granger causality values
    granger_causalities = np.load(os.path.join(wd, f))

    # Determine significant neurons
    alpha = 0.001
    significant_neurons = np.where(granger_causalities < alpha)[0]

    # Add to dictionary
    granger_dict[key] = significant_neurons

# Plot significant neurons
# Plot all positions in 3d using plotly
fig = plot_clusters_plotly(positions, traces_filtered, granger_dict)
fig.show(renderer='browser')

# # Save figure as html (indicate experiment date and fish ID)
# exp_date = wd.split('/')[-3]
# fish_id = wd.split('/')[-2]
#
# fig.write_html(os.path.join(wd, f'granger_causality_{exp_date}_{fish_id}.html'))

# Get mean trace for each stimulation ID


print('Done')
