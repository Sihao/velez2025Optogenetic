import os
from plotting import *
import numpy as np
from Cascade.cascade2p import checks
checks.check_packages()
from Cascade.cascade2p import cascade
from timeseries import zscore_traces
from experiment_io import load_traces, load_stim_info, load_positions


# Define working directory
wd = '/mnt/bronknas/Singles/Data_Singles_To_Analyze/Data_singles_2/240823/Fish_H/avgs/merged/'

# Load neural data
traces = load_traces(os.path.join(wd, 'merged_raw.pkl'))

## Filter traces
# traces_filtered = filter_traces(traces, fs=2.18, kernel_decay=3)

# zscore traces
# traces_filtered = zscore_traces(traces_filtered)
traces = zscore_traces(traces)

model_name = "Zebrafish_1Hz_smoothing1000ms"

cascade.download_model( model_name,verbose = 1)

# pre-allocate array for results
total_array_size = traces.itemsize*traces.size*64/1e9

# If the expected array size is too large for the Colab Notebook, split up for processing
if total_array_size < 10:

    spike_prob = cascade.predict( model_name, traces, verbosity=1 )

# Will only be use for large input arrays (long recordings or many neurons)
else:

    print("Split analysis into chunks in order to fit into Colab memory.")

    # pre-allocate array for results
    spike_prob = np.zeros((traces.shape))
    # nb of neurons and nb of chuncks
    nb_neurons = traces.shape[0]
    nb_chunks = int(np.ceil(total_array_size/10))

    chunks = np.array_split(range(nb_neurons), nb_chunks)
    # infer spike rates independently for each chunk
    for part_array in range(nb_chunks):
        spike_prob[chunks[part_array],:] = cascade.predict( model_name, traces[chunks[part_array],:] )


test_idx = np.array([106864,  81122,  96101, 113643, 100283, 110331, 110518, 106867,
       103748, 113686])
# Plot some example traces
fig_ex, ax_ex = plot_traces(traces, test_idx)
fig_ex.show()

# Plot spike probabilities over traces
for i in range(len(test_idx)):
    ax_ex[i].plot(spike_prob[test_idx[i], :]*0.5, color='red')
