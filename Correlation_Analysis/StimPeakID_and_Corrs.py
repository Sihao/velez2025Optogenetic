#!/usr/bin/env python
# coding: utf-8

# In[92]:


def get_train_2(path_stim, vps, num_stims=8, stim_cycles=10, sampling_rate=50000, distance=250000,  height=0.05, crop=90, stim_length=100):
    '''
    Using stimulus train gathered from the movement of the galvanometer mirrors identify 
    peaks of train and categorize them depending on the location/neuromast they targeted. Uses 
    trial information 
    
    Params:
    Path Stim: Stim file 
    VPS: Imaging volume rate
    Num_Stims: Number of neuromast targeted + 1   /number of stimuli
    Stim_Cycles: Number of stimulus repeats
    sampling_rate: Stim train sampling rate in Hz
    distance: Estimated interval between peaks for scipys find_peaks function
    height: Estimated height of peaks for scipys find_peaks function. Height should be above noise threshold
    crop: Timepoints cropped at beginning of time series - needs to match
    
    Returns: Peak timepoints as indices. Matrix: num_stims x length of time series
    '''
    
    # load stimulus train gathered from galvanomeneter displacement
    stims = np.loadtxt(path_stim, delimiter=",", dtype=float)
    # identify peaks based on sampling_rate and distance parameters
    from scipy.signal import find_peaks
    stims = np.abs(stims[2,:])*10
    array_of_stims=[]
    
    peakIdxs = (find_peaks(stims, height=height, distance=distance)[0])
    
    # find time difference between first and second peak to define as interstimulus interval
    firstPeak = (peakIdxs[0]/(sampling_rate/vps))
    secondPeak = (peakIdxs[1]/(sampling_rate/vps))
    ISI = secondPeak-firstPeak
    print(ISI)
    
    # Iterate through num_stims. For each stim location identify the index of a peak at the interval of
    # ISI*num_stims. Create a zero array of the length of the experimental trial and log stim indices as 1. 
    # Crop and Append each stim train.
    for i in range(num_stims):
        arr = np.zeros((np.floor(stims.shape[0]/(sampling_rate/vps))).astype('int'))
        init = firstPeak + i*ISI
        end = init + (stim_cycles-1)*num_stims*ISI
        idxs = (np.linspace(init, end, num=stim_cycles)).astype('int')
        print(idxs)
        arr[idxs] = 1
        
        arr = arr[crop:]
        array_of_stims.append(arr)
    
    return array_of_stims, stims

def get_train(path_stim, vps, sampling_rate=50000, height=0.05, crop=90, distance=250000, stim_length=100):
    
    '''
    Using stimulus train gathered from the movement of the galvanometer mirrors identify 
    peaks of train and categorize them depending on the location/neuromast they targeted. Uses 
    trial information 
    
    Params:
    Path Stim: Stim file 
    VPS: Imaging volume rate
    Num_Stims: Number of neuromast targeted + 1   /number of stimuli
    Stim_Cycles: Number of stimulus repeats
    sampling_rate: Stim train sampling rate in Hz
    distance: Estimated interval between peaks for scipys find_peaks function
    height: Estimated height of peaks for scipys find_peaks function. Height should be above noise threshold
    crop: Timepoints cropped at beginning of time series - needs to match
    
    Returns: Peak timepoints as indices. Matrix: num_stims x length of time series
    '''
    
    
    stims = np.loadtxt(path_stim, delimiter=",", dtype=float)
    from scipy.signal import find_peaks
    stims= stims[2,:]*10 
    array_of_stims=[]
    
    peakIdxs = (find_peaks(stims, height=height, distance=distance)[0])
    peakValues = list(np.unique(np.floor(stims[peakIdxs])).astype('int'))
    peakIdxs_neg = (find_peaks(-1*stims, height=height, distance=distance)[0])
    peakValues_neg = list(np.unique(np.floor(stims[peakIdxs_neg])).astype('int'))
    peakValues.extend(peakValues_neg)
    
    for i in peakValues:
    
        arr = np.zeros((np.floor(stims.shape[0]/(sampling_rate/vps))).astype('int'))
        x = np.where(np.floor(stims[peakIdxs])==i)[0]
        idx = np.floor(peakIdxs[x]/(sampling_rate/vps)).astype("int")
        arr[idx] = 1
        
        x = np.where(np.floor(stims[peakIdxs_neg])==i)[0]
        idx = np.floor(peakIdxs_neg[x]/(sampling_rate/vps)).astype("int")
        arr[idx] = 1
        
        arr = arr[crop:]
        array_of_stims.append(arr)

        
    return array_of_stims, stims

def conv(stims,  tau=6):
    """
    Convolves stim train for each location/neuromast with a exponential decay kernel
    
    Stims: Stim trains. Matrix: num_stims x length of time serie
    Tau: Decay time constant of calcium indicator
    
    Return: Convolved stim train. Matrix: num_stims x length of time series
    
    """
    # define calcium indicator kernel 
    kernel = (1/tau)*np.exp(-(np.linspace(-40,40,80)/tau))*np.heaviside(np.linspace(-40,40,80), 1)

    # convolve each stim
    import scipy.signal as signal 
    
    convs = []
    for i in range(len(stims)):
        conv = scipy.signal.convolve(stims[i], kernel)[39:-40]
        convs.append(conv/np.max(conv))
    
    return convs

def uploadCaiman(path, i):
    
    """
    Uploads cn_filter image and ROI objects from CaIman
    
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
    
def get_pearson(z, convs):
    """
    z: f/dff time series for each neuron. Matrix: (neurons x time series length)
    convs: convolved stim trains. Matrix: ()
    
    returns: Pearson correlation matrix between f/dff or and respective convs. Matrix: Convs x neurons.
    """
    
    traceSum = np.sum(z, axis=1)
    nonzero_idxs = np.where(traceSum !=0)[0]
    d = z[nonzero_idxs,:]
    d = d/(np.max(d, axis=0)-np.min(d, axis=0))
    x = np.vstack((d, convs))
    y = np.corrcoef(x)
    
    return np.transpose(y[-convs.shape[0]:, :d.shape[0]])

import seaborn as sns

def plot_traces(path, z, idxs, stim, x):
    
    """Plot correlated traces either as raw traces or as a heatmap."""

    fig, axs = plt.subplots(nrows=len(idxs)+1, ncols=1, figsize=(20, 35))
    
    for i in range(len(idxs)):
        y = z[idxs[i],:]
        axs[i].plot(y/np.max(y))
        axs[i].axis('off')
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
    print(len(idxs))
    axs[len(idxs)] = plt.plot(stim)  
    
    plot_name = "plot" + str(x) + ".png"
    plt.savefig(os.path.join(path, 'merged', plot_name), bbox_inches='tight')
    plt.close()
    
    fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 5]})
    
    a0 = plt.plot(stim)
    
    a1 = sns.heatmap(z[idxs,:])
    plot_name = "plot" + str(x) + "heatmap.png"
    plt.savefig(os.path.join(path, 'merged', plot_name), bbox_inches='tight')
    plt.close()

import scipy.interpolate

def uploadImages(path):
    
    """
    Upload image stack for max projection
    """
    
    files = natsort.natsorted(glob.glob(os.path.join(path, '*.TIFF')))
    
    dstack = []
    for i in range(len(files)):
        dstack.append(skimage.io.imread(files[i])[1,:,:])
        print(skimage.io.imread(files[i])[1,:,:].shape)
    dstack = np.stack(dstack, axis=2)
    dstack = np.asarray(dstack)
    print(dstack.shape)
    return dstack
    
def createMIPs(path):
    
    """
    Make max projection in XY, ZX, ZY axis. ZX and ZY axis are interpolated.
    """
    
    dstack = uploadImages(path)
    max_stack = dstack
    max_XY = np.max(max_stack, axis=2)
    max_ZY = np.max(max_stack, axis=1)
    #xzy, yzy = np.indices(np.shape(max_ZY))
    xzy = np.linspace(0, max_ZY.shape[0], max_ZY.shape[0])
    yzy = np.linspace(0, max_ZY.shape[1], max_ZY.shape[1])
    maxZY = scipy.interpolate.RectBivariateSpline(xzy, yzy, max_ZY)
    max_ZX = np.max(max_stack, axis=0)
    #xzx, yzx = np.indices(np.shape(max_ZY))
    xzx = np.linspace(0, max_ZX.shape[0], max_ZX.shape[0])
    yzx = np.linspace(0, max_ZX.shape[1], max_ZX.shape[1])
    maxZY = scipy.interpolate.RectBivariateSpline(xzx, yzx, max_ZX)

    return max_XY, max_ZY, max_ZX

def LocationFig(path, As, XY, ZY, ZX, idxs, i, corrs): 
    
    """
    Plot the location of correlated neurons on the XY, ZY, ZX mark projections of a zebrafish brain volume
    """
    
    Ax = As[idxs, 0]
    Ay = As[idxs, 1]
    Az = As[idxs, 2]
    corrs = corrs[idxs]
    #cmap = sns.cubehelix_palette(as_cmap=True)

    plt.figure()
    plt.imshow(XY, cmap='gray', vmin=100, vmax=180)
    plt.scatter(Ay, Ax, s=0.8)
    plt.savefig(os.path.join(path, 'merged', (str(i) + "_XY.pdf")), dpi=1260)
    plt.close()

    plt.figure()
    plt.imshow(ZY, cmap='gray', vmin=100, vmax=180)
    plt.scatter(Az, Ax, s=0.8) #alpha=corrs)
    plt.savefig(os.path.join(path, 'merged', (str(i) + "_ZY.pdf")), dpi=1260)
    plt.close()

    plt.figure()
    plt.imshow(ZX, cmap='gray', vmin=100, vmax=180)
    plt.scatter(Az, Ax,  s=0.8)# alpha=corrs)
    plt.savefig(os.path.join(path, 'merged', (str(i) + "_ZX.pdf")), dpi=1260)
    plt.close()

def assess_pearson(dFs, corr1, numStimuli=8, numIters=500):
    
    """
    Identifies neurons correlated to each neuromasts' stim train by creating a null distribution
    of randomly delta functions and then comparing this distribution to the distribution of correlations of
    the actual neurons. Any neuron with a correlation higher than the mean + 3*stds from the null distribution
    gets accepted as significant.
    
    params: 
    dFs: dF/F
    corr1: Pearson correlation matrix of num_stims x neurons
    """

    Rs = np.zeros((numIters, dFs.shape[0]))

    zeroTrain = np.zeros((numIters, dFs.shape[1]))
    for i in range(numIters):
        randomIdxs = np.random.randint(dFs.shape[1], size=numStimuli)
        zeroTrain[i,randomIdxs] = 1

    conv_train = conv(zeroTrain, tau=6)
    conv_train = np.array(conv_train)
    Rs = get_pearson(dFs, conv_train)

    return Rs
    
def plot_corr_hist(corr_control, corr1, i):
    
    """
    Plots historgram of correlation values
    """
    plt.figure()
    bins = np.linspace(-1, 1, 200)
    plt.hist(corr_control, bins, alpha=0.5, label='Control Corrs')
    plt.hist(corr1, bins, alpha=0.5, label='Neuron Corrs')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(path, 'merged', (str(i) + "_hist.pdf")), dpi=1260)
    
def save_object(obj, filename):
    
    """saves caiman object using pickle
    
    params:
    obj: NMF object
    filename: filename path to save to
    
    """
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL) 

def upload_data(path): 
    
    with open(os.path.join(path, 'merged', "data.pkl"), 'rb') as f:
        data_dicts = pickle.load(f)

    convs = data_dicts['convs']
    stims = data_dicts['stims']
    corrs = data_dicts['corrs']
    corrs_control = data_dicts['control_corrs']
    
    return convs, stims, corrs, corrs_control


import glob
import natsort
from caiman.base.rois import com
import os
import numpy as np
import scipy
import skimage.io
import pickle
import matplotlib.pyplot as plt
import caiman.source_extraction.cnmf.utilities as cmn

path_stim = r"D:\Data\041323_opto\Stims\fishE_50ms_run3_stim_data.csv"
path = r"D:\Data\50_ms\Fish_E_04132023\avgs"
filelist = natsort.natsorted(glob.glob(os.path.join(path, '*.TIFF')))

stims, stims_x = get_train_2(path_stim, 2.2280)
convs = conv(stims)
convs = np.asarray(convs)

#with open(os.path.join(path, 'merged', "merged_Fdff.pkl"), 'rb') as f:
#    dFs = pickle.load(f)

with open(os.path.join(path, 'merged', "merged_centroids.pkl"), 'rb') as f:
    As = pickle.load(f)

with open(os.path.join(path, 'merged', "merged_raw.pkl"), 'rb') as f:
    Raws = pickle.load(f)

#these two should be the same, if not VPS error
print(len(stims[0]))
print(Raws.shape)

corr2 = get_pearson(Raws, convs)
corrs_control = assess_pearson(Raws, corr2)


plt.figure()
for i in range(len(stims)): 
    plt.plot(stims[i], label=i)
plt.legend()
plt.savefig(os.path.join(path, 'merged', "stims_train"), dpi=1260)

data_dict = {'convs': convs, 'stims': stims, 'corrs': corr2, 'control_corrs': corrs_control}
with open(os.path.join(path, 'merged', "data.pkl"), 'wb') as handle:
    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

for i in range(len(raw_y)):
    
    if len(raw_y[i]) > 0:
        sort_idxs = np.argsort(raw_y[i])
        sorted_Raws = Raws[sort_idxs, :]

        plot_traces(path, Raws, raw_y[i], convs[i], i)
        LocationFig(path, As, XY, ZY, ZX, raw_y[i], i, corr2[:,i])  
        plot_corr_hist(corrs_control[:,i], corr2[:,i], i)




