#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imio import load
import os, glob
from skimage.segmentation import find_boundaries
from matplotlib_scalebar.scalebar import ScaleBar

wd = os.path.join('D:', 'Nico_data', 'scape_registration_pilots_combos')
name = 'Fish_C_072423_HR'

# load tiffs: zfbrain, regions
atlas = load.load_any(os.path.join(wd, 'T_AVG_HuCH2BGCaMP2-tg_ch0.tiff')) / 255
mon = load.load_any(os.path.join(wd, 'MON.tif')) / 255
torus = load.load_any(os.path.join(wd, 'Torus.tif')) / 255
sup_dor_med = load.load_any(os.path.join(wd, 'superior_dorsal_medulla_oblongata.tif')) / 255
int_dor_med = load.load_any(os.path.join(wd, 'intermediate_dorsal_medulla_oblongata.tif')) / 255
dor_med_obl = sup_dor_med + int_dor_med - mon
# fish = load.load_any(os.path.join(wd, name + '.nii'))[..., 0]
# roi_raw = pd.read_csv(os.path.join(wd, name + '.csv'))
roi_df = pd.read_csv(os.path.join(wd, name + '_Warped.csv'))   ## CHECK PATH
# roi_pool = pd.read_csv(os.path.join(wd, 'ROIs_pooled.csv'))

# # ROIs: load, pool, save
# roi_fn = sorted(list(glob.glob(os.path.join(wd, 'ROIs_MULTIPLES', '*_Warped.csv')))) ## CHECK PATH
# roi_df = pd.read_csv(roi_fn[0])
# for fn in roi_fn[1:]: roi_df = pd.concat([roi_df, pd.read_csv(fn)], axis = 0)
# ix0 = (roi_df['z'] >= 0) & (roi_df['z'] <= atlas.shape[0]) & \
#       (roi_df['y'] >= 0) & (roi_df['y'] <= atlas.shape[1]) & \
#       (roi_df['x'] >= 0) & (roi_df['x'] <= atlas.shape[2])      # in-frame only
# ix1 = (atlas > .2)[roi_df.loc[ix0, 'z'].values.astype(int), 
#                    roi_df.loc[ix0, 'y'].values.astype(int), 
#                    roi_df.loc[ix0, 'x'].values.astype(int)].astype(bool)  # brain only
# roi_df = roi_df.loc[ix0, :].loc[ix1, :].reset_index(drop = True)
# # roi_df.to_csv(os.path.join(wd, 'ROIs_pooled.csv'), index = False)

#%%
# XY plane
binsize = 9
fig, ax = plt.subplots(1, 1, figsize = (7, 7))
ax.imshow(atlas.max(0), cmap = 'binary_r')
xlim = ax.get_xlim(); ylim = ax.get_ylim()
H, _ = np.histogramdd(roi_df[['y', 'x']].values, range = [ylim[::-1], xlim], bins = [atlas.shape[1] // binsize, atlas.shape[2] // binsize])
ax.imshow(H, extent = [xlim[0], xlim[1], ylim[0], ylim[1]], cmap = 'afmhot', alpha = .8)

yy, xx = np.where(find_boundaries(mon.max(0)))
ax.scatter(xx, yy, c = 'chartreuse', s = .025, zorder = 100, label = 'Medial octavolateralis nucleus')

yy, xx = np.where(find_boundaries(dor_med_obl.max(0)))
ax.scatter(xx, yy, c = 'cyan', s = .025, alpha = .5, label = 'Superior/intermediate dorsal M.O.')

yy, xx = np.where(find_boundaries(torus.max(0)))
ax.scatter(xx, yy, c = 'darkviolet', s = .025, alpha = .5, label = 'Torus semicircularis')

ax.axis(False)
plt.legend(frameon = False, labelcolor = 'w', markerscale = 25, loc = 'upper center')
scalebar = ScaleBar(1., 'um', length_fraction = .2, location = 'lower right', color = 'w', frameon = False)
ax.add_artist(scalebar)
plt.show()
# plt.savefig(os.path.join(wd, 'figures', f'{name}_SINGLES_ROIs_heatmap_XYplane.svg'), dpi = 600)

# %%
# ZY plane
fig, ax = plt.subplots(1, 1, figsize = (7, 7))
ax.imshow(atlas.max(2).T, cmap = 'binary_r')
xlim = ax.get_xlim(); ylim = ax.get_ylim()
H, _ = np.histogramdd(roi_df[['y', 'z']].values, range = [ylim[::-1], xlim], bins = [atlas.shape[1] // binsize, atlas.shape[0] // binsize])
ax.imshow(H, extent = [xlim[0], xlim[1], ylim[0], ylim[1]], cmap = 'afmhot', alpha = .8)

yy, xx = np.where(find_boundaries(mon.max(2).T))
ax.scatter(xx, yy, c = 'chartreuse', s = .025, zorder = 100)

yy, xx = np.where(find_boundaries(dor_med_obl.max(2).T))
ax.scatter(xx, yy, c = 'cyan', s = .025, alpha = .5)

yy, xx = np.where(find_boundaries(torus.max(2).T))
ax.scatter(xx, yy, c = 'darkviolet', s = .025, alpha = .5)

ax.axis(False)
plt.show()
# plt.savefig(os.path.join(wd, 'figures', f'{name}_SINGLES_ROIs_heatmap_ZYplane.svg'), dpi = 600)

# %%
# XZ plane
fig, ax = plt.subplots(1, 1, figsize = (7, 7))
ax.imshow(atlas.max(1), cmap = 'binary_r', origin = 'lower')
xlim = ax.get_xlim(); ylim = ax.get_ylim()
H, _ = np.histogramdd(roi_df[['z', 'x']].values, range = [ylim, xlim], bins = [atlas.shape[0] // binsize, atlas.shape[2] // binsize])
ax.imshow(H, extent = [xlim[0], xlim[1], ylim[0], ylim[1]], cmap = 'afmhot', alpha = .8, origin = 'lower')

yy, xx = np.where(find_boundaries(mon.max(1)))
ax.scatter(xx, yy, c = 'chartreuse', s = .025, alpha = .75, zorder = 100)

yy, xx = np.where(find_boundaries(dor_med_obl.max(1)))
ax.scatter(xx, yy, c = 'cyan', s = .025)

yy, xx = np.where(find_boundaries(torus.max(1)))
ax.scatter(xx, yy, c = 'darkviolet', s = .025, alpha = .5)

ax.axis(False)
plt.show()
# plt.savefig(os.path.join(wd, 'figures', f'{name}_SINGLES_ROIs_heatmap_XZplane.svg'), dpi = 600)

# %%
