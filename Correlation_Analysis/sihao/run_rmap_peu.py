import matplotlib.pyplot as plt
import numpy as np

file = '/home/slu/cnmfe_traces_decon_peu.npy'

wd = '/home/slu/'
# Load neural data
traces = np.load(file)

# Replace NaN with 0
traces = np.nan_to_num(traces)

# z-score
traces = (traces - traces.mean(axis=1)[:, None]) / traces.std(axis=1)[:, None]


# Set sampling frequency
fs = 5.44


# Truncate first 30 frames
# traces = traces[:, 30:]


# Preprocess traces
# traces_filtered = preprocess_traces(traces, fs=fs, kernel_decay=20, bleaching_tau=100, zscore=True, filter=True)

# Plot heatmap
from plotting import plot_heatmap

fig, ax = plot_heatmap(traces, fs=fs, cmap='gray')
fig.show()

# Rastermap
from utils import get_rmap_idx

rmap_idx, cluster_labels = get_rmap_idx(traces, n_clusters=12, n_PCs=16, locality=0.3, time_lag_window=30, grid_upsample=5)

# Only show rows for first cluster
sorted_cluster_labels = cluster_labels[rmap_idx]
sorted_traces = traces[rmap_idx]
# sorted_traces = sorted_traces[sorted_cluster_labels == 0]

fig, ax = plot_heatmap(sorted_traces, fs=fs, cmap='gray_r', interpolation='none')
# Move to right edge
fig.tight_layout()
fig.show()

# Relate clusters back to spatial footprint
footprints = np.load('/home/slu/cnmfe_spatial_components_peu.npy', allow_pickle=True)
# Convert from Compressed Sparse Column format to regular numpy array
footprints = footprints.item().toarray()

# Reshape back into 512x512 images
footprints = footprints.reshape((512, 512, -1))

# Cluster indices relative to spatial footprints
footprint_cluster_idx = np.argwhere(cluster_labels == 0)
# Clean up for indexing
footprint_cluster_idx = [idx[0] for idx in footprint_cluster_idx]

# Show footprint for first cluster
fig, ax = plt.subplots()
# Add footprints belonging to cluster
cluster_footprint = footprints[:, :, footprint_cluster_idx].sum(axis=-1)
for footprints_cluster_id in footprint_cluster_idx:
    ax.imshow(cluster_footprint.T, cmap='gray')
fig.show()
