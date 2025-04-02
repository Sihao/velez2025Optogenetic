import logging
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import os

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
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour, nb_inspect_correlation_pnr
import cv2

def eliminate_duplicate(C_all, centroids_all, centroids_sums_all, raw_dff_all):
        
        corrs = np.corrcoef(np.asarray(raw_dff_all))
        corrs[corrs == 1] = 0

        from scipy.spatial.distance import pdist, squareform
        dists = squareform(pdist(centroids_all[:,:2]))
        zdists = squareform(pdist(np.expand_dims(centroids_all[:,2], axis=1)))

        merge_thr = 0.9
        tau = 6
        ztau = 2
        edges = np.where(np.triu(np.all(np.asarray((dists<tau, corrs>merge_thr, zdists<ztau)), axis=0),k=1))

        import networkx as nx
        G = nx.Graph()
        G.add_edges_from(np.asarray(edges).T)
        cc = list(nx.connected_components(G))

        for i in range(len(cc)):
            x = list(cc[i])
            Totals = centroids_sums_all[x]
            C_all[x[0],:] = np.asarray(Totals.T * C_all[x,:] / np.sum(Totals,axis=0))[0]
            raw_dff_all[x[0],:] = np.asarray(Totals.T * raw_dff_all[x,:] / np.sum(Totals,axis=0))[0]
            centroids_all[x[0],:] = np.asarray(Totals.T * centroids_all[x,:] / np.sum(Totals,axis=0))[0]
            centroids_sums_all[x] = np.sum(Totals, axis=0)[0]

            C_all[x[1:],:] = np.nan
            centroids_all[x[1:],:] = np.nan
            raw_dff_all[x[1:],:] = np.nan

            #if A_all is not None:
            #    from scipy.sparse import csc_matrix
            #    curr = csc_matrix((A_all[0].shape[0], A_all[0].shape[1]),dtype=np.float16)
              #  for ii in range(len(idxs)):
              #      curr += SNRs[0,ii] * A_all[idxs[ii]]
               # A_all[idxs[0]] = curr 
                #for ii in range(1,len(idxs)): 
                 #   A_all[idxs[ii]] = None

        keep = np.where(np.logical_not(np.any(np.isnan(C_all),axis=1)))[0]

        
        C_all_keep = C_all[keep,:]
        centroids_all_keep = centroids_all[keep,:]
        centroids_sums_all_keep = centroids_sums_all[keep]
        raw_dff_all_keep = raw_dff_all[keep,:]

        #if A_all is not None:
            #A_all = [x for x in A_all if x is not None]
            
        return C_all_keep, centroids_all_keep, centroids_sums_all_keep, raw_dff_all_keep

import skimage.io
import glob
import natsort
import pickle
from caiman.base.rois import com
import caiman as cnm
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import caiman.source_extraction.cnmf.utilities as cmn

def uploadCaiman(path, i):
    
    """Uploads cn_filter image and ROI objects from CaIman
    
    Params: 
    Path: Directoy path
    i: File
    
    Return: cn_filter image and ROI object. 
    
    """
    
    cn_filters_path = os.path.join(path, 'cn_filters', (str(i) + ".pkl"))
    NMF_object_path = os.path.join(path, 'py_objects', (str(i) + ".pkl"))
        
    with open(cn_filters_path, 'rb') as f:
        cn_filter = pickle.load(f)
        
    with open(NMF_object_path, 'rb') as f:
        NMF_object = pickle.load(f)
        
    return cn_filter, NMF_object
    
def collate_planes(path, load_masks=True, splits=10):
    
    """
    Adapated from Jason Manley
    
    collates neurons in adjacent places that are highly correlated to each other. 
    
    path: directory with videos of each plane of a brain volume
    
    return: 
    C_all: Unnormalized convolved neuronal time series: Matrix (neurons x time points)
    centroids_all: Centroids of each neuron on x,y,z coordinates: Matrix (neurons x 3)
    centroid_sums_all: sum of collated centroids for each neuron 
    F_dff_all: detrended and convolved F/dff time series: Matrix (neurons x time points)
    raw_dff_all: detrended F/dff time series: Matrix (neurons x time points)
    A_all: 3D masks for all segmented neurons
    """
    
    files = natsort.natsorted(glob.glob(os.path.join(path, 'py_objects','*pkl')))
    
    nplanes = len(files)
    print(nplanes)
    sect = int(np.round(nplanes/splits))
    print(files)
    #nplanes = 
    nplanes = len(files)
    sect = int(np.round(nplanes/splits))
    r=0
    C_total = []
    for k in range(splits):

        x = sect
        if k == splits-1:
            x = sect - (sect*splits-nplanes)-1
        C_all = []
        centroids_all = []
        SNR_all = []
        centroids_sums_all = []
        raw_dFF = []
        A_all = []
        
        
        for i in range(x):
            print(sect*k+i)
            
            if os.path.exists(os.path.join(path, 'py_objects', (str(sect*k+i) + ".pkl")))==True:
                cn_filter, NMF_object = uploadCaiman(path, sect*k+i)
            else:
                continue
                
            idx_components = NMF_object.idx_components 
            
            C = NMF_object.C[idx_components,:]
            YrA = NMF_object.YrA[idx_components,:]
            dims = cn_filter.shape
            coms= com(NMF_object.A, dims[0],dims[1])
            X = coms[:,0]
            Y = coms[:,1]
            A = np.vstack((X.T, Y.T, (np.zeros((NMF_object.A.shape[1])) + sect*k+i))).T
            A = A[idx_components, :]
            centroid_sums = np.sum(NMF_object.A, axis=0).T
            centroid_sums = centroid_sums[idx_components, :]
            print(C.shape)
            if C.shape[0] > 0:

                raw_dff= cmn.detrend_df_f(NMF_object.A[:, idx_components], NMF_object.b, C, NMF_object.f, YrA=YrA, quantileMin=8, frames_window=1000, flag_auto=True, use_fast=False, detrend_only=False)
                #raw_dff=C
                C_all.append(C)
                centroids_all.append(A)
                centroids_sums_all.append(centroid_sums)
                raw_dFF.append(raw_dff) 

            #if load_masks:

                #from scipy.sparse import csc_matrix
                #import warnings
                #warnings.simplefilter('ignore')
                #for ii in range(A.shape[1]):
                #    curr = csc_matrix((nplanes,A.shape[0]), dtype=A.dtype)
                #    curr[i,:] = A[:,ii].T
                #    A_all.append(curr.astype(np.float16))
                #warnings.simplefilter('default')
            #else:
            #    A_all = None
        
        if C_all== []:
            continue
        C_all = np.concatenate(C_all)
        centroids_all = np.concatenate(centroids_all, axis=0)
        centroids_sums_all = np.concatenate(centroids_sums_all, axis=0)
        raw_dff_all = np.concatenate(raw_dFF, axis=0)
            
        C_all_keep, centroids_all_keep, centroids_sum_all, raw_dff_all_keep = eliminate_duplicate(C_all, centroids_all, centroids_sums_all, raw_dff_all)
        print(k) 
        if C_total ==[]:
            C_total = C_all_keep
            centroids_total = centroids_all_keep
            centroids_sums_total = centroids_sums_all
            raw_data_total = raw_dff_all_keep
            
        else:
            C_total = np.vstack((C_total, C_all_keep))
            centroids_total = np.vstack((centroids_total, centroids_all_keep))
            centroids_sums_total = np.vstack((centroids_sums_total, centroids_sums_all))
            raw_data_total = np.vstack((raw_data_total, raw_dff_all_keep))
        print("x")
        print(C_total.shape)
    #C_total, centroids_total, centroids_sum_total, raw_data_total = eliminate_duplicate(C_total, centroids_total, centroids_sums_total, raw_data_total)

    return C_total, centroids_total, centroids_sums_total, raw_data_total
    
def save_object(obj, filename):
    
    """saves caiman object using pickle
    
    params:
    obj: NMF object
    filename: filename path to save to
    
    """
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL) 

directory = r"E:\output\Fuuuuuuck"

for i in os.listdir(directory):
    
    path = os.path.join(directory, i)
    if os.path.exists(os.path.join(path, 'merged')) == False: 
        
        os.mkdir(os.path.join(path, 'merged'))
        
    C_all, centroids_all, centroids_sums_all, raw_dff_all= collate_planes(path)
    
    save_object(C_all, os.path.join(path, "merged", "merged_traces.pkl"))
    save_object(centroids_all, os.path.join(path, "merged", "merged_centroids.pkl"))
    save_object(raw_dff_all, os.path.join(path, "merged", "merged_raw.pkl"))
    #save_object(A_all, os.path.join(path, "merged", "merged_A.pkl"))
    
