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

rmap_idx, cluster_labels = get_rmap_idx(traces, n_clusters=12, n_PCs=2048, locality=0.1, time_lag_window=10, grid_upsample=2)

# Only show rows for first cluster
sorted_cluster_labels = cluster_labels[rmap_idx]
sorted_traces = traces[rmap_idx]
sorted_traces = sorted_traces[sorted_cluster_labels == 0]

fig, ax = plot_heatmap(sorted_traces, fs=fs, cmap='gray_r', interpolation='none')



# Move to right edge
fig.tight_layout()

fig.show()

# Plot as a joyplot
fig, ax = plt.subplots(figsize=(2.9*1.3, 2.5 * 1.3))
for i, trace in enumerate(traces[rmap_idx[1:51]]):
    ax.plot(trace[30:] + i*1.6, color='black', linewidth=0.5)

ax.set_yticks([])
ax.set_xticks([])
# Box off
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Draw small arrow orthogonal arrows in bottrom left corner with axis labels
# Use reverse arrow to indicate direction
# set fontsize
ax.annotate('2 s', xy=(0, -5), xytext=(fs*2,  -5), va='center',
            arrowprops=dict(arrowstyle='-', color='black', relpos=(0, 0.5)), fontsize=8, fontname='Nimbus Mono PS')

ax.annotate('10 z'.format(1), xy=(-5, 0), xytext=(-5, 10), ha='center',
            arrowprops=dict(arrowstyle='-', color='black', relpos=(0.5, 0)), fontsize=8, fontname='Nimbus Mono PS', rotation=90)


fig.savefig(wd + 'rastermap.svg')
