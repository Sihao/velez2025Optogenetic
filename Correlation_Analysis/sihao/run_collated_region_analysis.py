import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotting
from plotting import plot_spatial_sparsity, plot_region_sparsity_comparison
import seaborn as sns
import tifffile
from plotting import plot_spatial_sparsity

# Load data
df_path = '/mnt/bronknas/Singles/Data_Singles_To_Analyze/Data_singles_2/Summary/collated_region_df.pkl'
df = pd.read_pickle(df_path)

# Load background image for registered fish
background_path = '/mnt/bronknas/T_AVG_HuCH2BGCaMP2-tg_ch0.tiff'
# (Z, Y, X)
atlas_stack = tifffile.imread(background_path)

# Standard deviation projection in each plane
flips = [True, False, False]
for i, plane in enumerate(['horizontal', 'coronal', 'sagittal']):
    atlas_projection = np.std(atlas_stack, axis=i)
    # Normalise histogram
    atlas_projection = (atlas_projection - np.min(atlas_projection)) / (np.max(atlas_projection) - np.min(atlas_projection))

    # Flip if necessary
    flip_y = flips[i]

    # Include only SVM neurons, and neurons with x < 600, y < 800
    svm_mask = (df['svm_neuron'] == True)
    location_mask = (df['x'] < 500) & (df['y'] < 700)
    fig_raw_all, ax_raw_all = plot_spatial_sparsity(df[location_mask],
                                                    plane=plane, sparsity_measure='raw', opacity_scaler=16,
                                                    bg=atlas_projection, subplot_labels='All - raw', flip_y=flip_y)
    fig_raw_svms, ax_raw_svms = plot_spatial_sparsity(df[svm_mask & location_mask],
                                                      plane=plane, sparsity_measure='raw', opacity_scaler=4,
                                                      bg=atlas_projection, subplot_labels='SVMs - raw', flip_y=flip_y)
    fig_weights_svms, ax_weights_svms = plot_spatial_sparsity(df[svm_mask & location_mask],
                                                                plane=plane, sparsity_measure='weights', opacity_scaler=4,
                                                                bg=atlas_projection, subplot_labels='SVMs - weights', flip_y=flip_y)

    # Show
    fig_raw_all.show()
    fig_raw_svms.show()
    fig_weights_svms.show()


# Make gridspace to compute mean spatial sparsity
x = np.linspace(0, 800, 80)
y = np.linspace(0, 500, 50)
sparsity = np.zeros((len(y), len(x)))

# Sparsity mean
sparsity_mean = df.raw_sparsity.mean()

for i, x_ in enumerate(x):
    for j, y_ in enumerate(y):
        # Define radius around gridpoint to consider
        sample_radius_x = np.diff(x)[0]/2
        sample_radius_y = np.diff(y)[0]/2

        mask = ((df['x'] > x_ - sample_radius_x) & (df['x'] < x_ + sample_radius_x) &
                (df['y'] > y_ - sample_radius_y) & (df['y'] < y_ + sample_radius_y))

        # Intersect with svm mask
        local_sparsity = df[mask]['raw_sparsity'].mean()
        # Set to NaN if too low
        if local_sparsity < sparsity_mean:
            local_sparsity = np.nan
        sparsity[j, i] = local_sparsity
# Overlay sparsity on top of spatial plot
fig, ax = plot_spatial_sparsity(df[svm_mask], sparsity_measure='weights', opacity_scaler=12)
ax.imshow(sparsity, extent=(0, 500, 0, 800), origin='upper', cmap=sns.color_palette('crest_r', as_cmap=True), alpha=0.3, aspect='equal')
fig.show()


fig_raw_svms, ax_raw_svms = plot_region_sparsity_comparison(df[svm_mask], sparsity_measure='raw', subplot_labels='SVMs - raw')
fig_weights_svms, ax_weights_svms = plot_region_sparsity_comparison(df[svm_mask], sparsity_measure='weights', subplot_labels='SVMs - weights')
fig_raw_all, ax_raw_all = plot_region_sparsity_comparison(df, sparsity_measure='raw', subplot_labels='All - raw')

fig_raw_svms.show()
fig_weights_svms.show()
fig_raw_all.show()

# Some statistics on sparsity between super regions
# Get super regions
regions = df.loc[svm_mask, 'region'].dropna().unique()

# Do a t-test between super regions
from scipy.stats import ttest_ind, ranksums
ttest_results = pd.DataFrame(index=regions, columns=regions)

for i, region_1 in enumerate(regions):
    for j, region_2 in enumerate(regions):
        if i == j:
            continue
        mask_1 = df['region'] == region_1
        mask_2 = df['region'] == region_2
        ttest_results.loc[region_1, region_2] = ranksums(df[mask_1]['raw_sparsity'], df[mask_2]['raw_sparsity']).pvalue

ttest_results = ttest_results.astype(float)
# Multiple comparison correction
from statsmodels.stats.multitest import multipletests
ttest_results_corrected = ttest_results.copy()
ttest_results_corrected[:] = multipletests(ttest_results.values.flatten(), method='bonferroni')[1].reshape(ttest_results.shape)
ttest_results_corrected = ttest_results_corrected.astype(float)
ttest_results_corrected
# Plot heatmap
fig, ax = plt.subplots()
ttest_results_corrected[ttest_results_corrected > 0.05] = np.nan
sns.heatmap(ttest_results_corrected, ax=ax, cmap='viridis', annot=True)
ax.set_title('Ranksums p-values between super regions')
fig.show()
