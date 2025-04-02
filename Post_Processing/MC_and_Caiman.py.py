# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:54:32 2022

@author: Nicolas Velez
"""

import natsort
import skimage.io
import pickle
import glob
import logging
import os
import scipy
import logging
import matplotlib.pyplot as plt
import numpy as np
import time

t0 = time.time()

logging.basicConfig(format=
                          "%(relativeCreated)12d [%(filename)s:%(funcName)10s():%(lineno)s] [%(process)d] %(message)s",
                    # filename="/tmp/caiman.log",
                    level=logging.WARNING)

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr, nb_inspect_correlation_pnr
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
import cv2

try:
    cv2.setNumThreads(0)
except:
    pass
import bokeh.plotting as bpl
import holoviews as hv

#bpl.output_notebook()
#hv.notebook_extension('bokeh')

####################################################################################################################################################
################################################################################################################################################################
#PARAMETERS
# dataset dependent parameters
# dataset dependent parameters
frate = 2.3                      # movie frame rate
decay_time = 3                 # length of a typical transient in seconds

# motion correction parameters
motion_correct = True    # flag for performing motion correction
pw_rigid = False         # flag for performing piecewise-rigid motion correction (otherwise just rigid)
gSig_filt = (7, 7)       # size of high pass spatial filtering, used in 1p data
max_shifts = (30, 30)      # maximum allowed rigid shift
strides = (48, 48)       # start a new patch for pw-rigid motion correction every x pixels
overlaps = (24, 24)      # overlap between pathes (size of patch strides+overlaps)
max_deviation_rigid = 5  # maximum deviation allowed for pat32ch with respect to rigid shifts
border_nan = 'copy'      # replicate values along the boundaries
bord_px = 1

# parameters for source extraction and deconvolution
# parameters for source extraction and deconvolution
p = 1            # order of the autoregressive system
K = None            # upper bound on number of components per patch, in general None
gSig = (2, 2)       # gaussian width of a 2D gaussian kernel, which approximates a neuron
gSiz = (6, 6)     # average diameter of a neuron, in general 4*gSig+1
Ain = None          # possibility to seed with predetermined binary masks
merge_thr = .9     # merging threshold, max correlation allowed
rf = 120          # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
stride_cnmf = 40    # amount of overlap between the patches in pixels
#                     (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
tsub = 1          # downsampling factor in time for initialization,
#                     increase if you have memory problems
ssub = 1            # downsampling factor in space for initialization,
#                     increase if you have memory problems
#                     you can pass them here as boolean vectors
low_rank_background = True  # None leaves background of each patch intact,
#                     True performs global low-rank approximation if gnb>0
gnb = 1            # number of background components (rank) if positive,
#                     else exact ring model with following settings
#                         gnb= 0: Return background as b and W
#                         gnb=-1: Return full rank background B
#                         gnb<-1: Don't return background
nb_patch = 1        # number of background components (rank) per patch if gnb>0,
#                     else it is set automatically
min_corr = .3  # min peak value from correlation image
min_pnr =  4.0  # min peak to noise ration from PNR image
ssub_B = 1          # additional downsampling factor in space for background
ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor      # additional downsampling factor in space for background

##################################################################################################################################
#MOTION CORRECT

def motionCorrect(i, fnames, opts, dview, bord_px, pw_rigid, motion_correct, border_nan):
    
    """
    Applies norm-corre motion correction on a single plane of timelapse
    """
    if motion_correct:
        # do motion correction rigid
        mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
        mc.motion_correct(save_movie=True)
        fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
        if pw_rigid:
            bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                     np.max(np.abs(mc.y_shifts_els)))).astype(int)
        else:
            bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(int)
            #plt.subplot(1, 2, 1); plt.imshow(mc.total_template_rig)  # % plot template
            #plt.subplot(1, 2, 2); plt.plot(mc.shifts_rig)  # % plot rigid shifts
            #plt.legend(['x shifts', 'y shifts'])
            #plt.xlabel('frames')
            #plt.ylabel('pixels')

        bord_px = 0 if border_nan == 'copy' else bord_px
        base_name = str(i) + 'memmap_'
        fname_new = cm.save_memmap(fname_mc, base_name=base_name, order='C',
                               border_to_0=bord_px)
    else:  # if no motion correction just memory map the file
        fname_new = cm.save_memmap(fnames, base_name=base_name,
                               order='C', border_to_0=0, dview=dview)
    
    return fname_new, mc

#########################################################################################################################
##############################################################################################################################
####### save objects
def save_object(obj, filename):
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
################################################################################################################################################################
##################################################################################################################################################################
def main():

    """
    Iterates through an directory of folders/each containing the images from a fish brain and does segmentation on each
    XY slice of the fish's volume across time
    """

    pass 

    try:
        cm.stop_server()  # stop it if it was running
    except():
        pass
            
    dview = None

    directory = r'D:\Data\50_ms\New folder'
    filelist = os.listdir(directory)
    
    for j in range(len(filelist)):
        path = os.path.join(directory, filelist[j])# 'avgs')
        #files = natsort.natsorted(glob.glob(os.path.join(path, '*.TIFF')))
        print("getting averages")
        #avgImages(path, files)
        print("finished averages")
        
        #######################################
        ###parallel processing ################
        #%% start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
        #%% start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)

        files = natsort.natsorted(glob.glob(os.path.join(path, 'avgs', '*.tiff')))
        
        if os.path.exists(os.path.join(path, 'avgs', 'py_objects')) == False: 
            os.mkdir(os.path.join(path, 'avgs', 'py_objects'))
    
        if os.path.exists(os.path.join(path, 'avgs', 'cn_filters')) == False: 
            os.mkdir(os.path.join(path, 'avgs', 'cn_filters'))
        
        
        for i in range(len(files)):
            fnames = [os.path.join(path, files[i])]
            
            mmap_file_name = str(i) + "memmap__d1_299_d2_651_d3_1_order_C_frames_1403_.mmap"
            mmap_file_name2 = str(i) + "memmap__d1_299_d2_651_d3_1_order_C_frames_1455_.mmap"
            if os.path.exists(os.path.join(path, 'avgs', mmap_file_name)) == True:
                print(i)
                continue
            
            if os.path.exists(os.path.join(path, 'avgs', mmap_file_name2)) == True:
                print(i)
                continue
            
            if 'dview' in locals():    cm.stop_server(dview=dview)
            c, dview, n_processes = cm.cluster.setup_cluster(
                backend='local', n_processes=60, single_thread=False)
            
            print(fnames)
            
            #########################################
            ###params dict###########################
            mc_dict = {
            'fnames': fnames,
            'fr': frate,
            'decay_time': decay_time,
            'pw_rigid': pw_rigid,
            'max_shifts': max_shifts,
            'gSig_filt': gSig_filt,
            'strides': strides,
            'overlaps': overlaps,
            'max_deviation_rigid': max_deviation_rigid,
            'border_nan': border_nan
            }
        
            opts = params.CNMFParams(params_dict=mc_dict)
            
            
            opts.change_params(params_dict={'method_init': 'corr_pnr',  # use this for 1 photon
                                        'K': K,
                                        'gSig': gSig,
                                        'gSiz': gSiz,
                                        'merge_thr': merge_thr,
                                        'p': p,
                                        'tsub': tsub,
                                        'ssub': ssub,
                                        'rf': rf,
                                        'stride': stride_cnmf,
                                        'only_init': True,    # set it to True to run CNMF-E
                                        'nb': gnb,
                                        'nb_patch': nb_patch,
                                        'method_deconvolution': 'oasis',       # could use 'cvxpy' alternatively # double check this. 
                                        'low_rank_background': low_rank_background,
                                        'update_background_components': False,  # sometimes setting to False improve the results
                                        'min_corr': min_corr,
                                        'min_pnr': min_pnr,
                                        'normalize_init': False,               # just leave as is
                                        'center_psf': True,                    # leave as is for 1 photon
                                        'ssub_B': ssub_B,
                                        'ring_size_factor': ring_size_factor,
                                        'del_duplicates': True,                # whether to remove duplicates from initialization
                                        'border_pix': bord_px})
            
            fname_new, mc = motionCorrect(i, fnames, opts, dview, bord_px, pw_rigid, motion_correct, border_nan)
            print("motion corrected")
            # load memory mappable file
            Yr, dims, T = cm.load_memmap(fname_new)
            images = Yr.T.reshape((T,) + dims, order='F')
            
            # compute some summary images (correlation and peak to noise)
            cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=gSig[0], swap_dim=False) # change swap dim if output looks weird, it is a problem with tiffile
            
            new_filename = str(i) + ".pkl"
            new_cnmfe = str(i) + ".hdf5"
            save_path = os.path.join(path, 'avgs', "cn_filters", new_filename)
            save_object(cn_filter, save_path)
            # inspect the summary images and set the parameters
            #nb_inspect_correlation_pnr(cn_filter, pnr)
            
            cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=Ain, params=opts)
            cnm.fit(images)
            
            #%% COMPONENT EVALUATION
            # the components are evaluated in two ways:
            #   a) the shape of each component must be correlated with the data
            #   b) a minimum peak SNR is required over the length of a transient
            # Note that here we do not use the CNN based classifier, because it was trained on 2p not 1p data
        
            min_SNR = 1.4           # adaptive way to set threshold on the transient size
            r_values_min = 0.5 # threshold on space consistency (if you lower more components
            min_cnn_t = 0.99          # threshold for CNN based classifier
            cnn_lowest = 0.1 
            cnm.params.set('quality', {'min_SNR': min_SNR,
                                       'rval_thr': r_values_min,
                                       'use_cnn': False})
            cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
        
            print(' ***** ')
            print('Number of total components: ', len(cnm.estimates.C))
            #print('Number of accepted components: ', len(cnm.estimates.idx_component))
            save_path = os.path.join(path, 'avgs', "py_objects", new_filename)
            save_object(cnm.estimates, save_path)
            
            cnm.save(os.path.join(path, 'avgs', "py_objects", new_cnmfe))
        ###############################################################################################


# %% This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
    
    