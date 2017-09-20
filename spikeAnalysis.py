from elephant.statistics import *
import numpy as np
from scipy.io.matlab import loadmat,savemat
from neo.core import SpikeTrain
from quantities import ms,s
import neo
import quantities as pq
import elephant
import sys
from elephant import kernels
from elephant.spike_train_correlation import cross_correlation_histogram,corrcoef
from elephant.conversion import *
import quantities as pq
from mpl_toolkits.mplot3d import Axes3D


dat = loadmat(fname,struct_as_record=False)
# access the analog data
vars = dat['vars'][0,0]
filtvars = dat['filtvars'][0,0]
rawvars = dat['rawvars'][0,0]
C = dat['C']
cc = convertC(C)
sp = dat['sp'][0,0]

ISI = isi(sp,0)
ISI = ISI[ISI<100]
CV = cv(ISI)
LV = lv(ISI.squeeze())

def iterate_filtvar(filtvars,cc):
    ''' this loops through every variable in a structure (generally filtvars)
    and returns a dict of mean and minmax for each variable in each contact'''
    mean_filtvars = {}
    minmax_filtvars = {}

    for attr in filtvars._fieldnames:
        mean_filtvars[attr], minmax_filtvars[attr] = get_analog_contact(getattr(filtvars,attr),cc)
    return mean_filtvars,minmax_filtvars

def get_analog_contact(var, cc):
    ''' this gets the mean and min-max of a given analog signal in each contact interval'''
    mean_var = np.empty([cc.shape[0], var.shape[1]])
    minmax_var = np.empty([cc.shape[0], var.shape[1]])

    for ii, contact in enumerate(cc):
        var_slice = var[contact[0]:contact[1], :]
        mean_var[ii, :] = np.mean(var_slice, 0)
        minmax_idx = np.argmax(np.abs(var_slice), 0)
        minmax_var[ii, :] = var_slice[minmax_idx, np.arange(len(minmax_idx))]

    return mean_var,minmax_var

def get_fr_by_contact(sp,cc,pre_onset=0,post_offset=0):
    ''' this is OK. Mean FR is trustworthy, the rest is maybe not exactly what we want to do.'''
    mean_fr = np.empty(cc.shape[0])
    locked_st=[]
    for ii, contact in enumerate(cc):
        idx = np.logical_and(sp>=(contact[0]-pre_onset),sp<(contact[1]+post_offset))
        sp_contact = sp[idx]-contact[0]
        mean_fr[ii] = mean_firing_rate(sp_contact*ms)
        locked_st.append(sp_contact)

    return mean_fr,locked_st

def join_locked_st(locked_st):
    ''' useful for creating a PSTH'''
    all_spikes = np.array([])
    for contact in locked_st:
        all_spikes = np.concatenate((all_spikes,contact))
    all_spikes.sort()
    return np.array(all_spikes)

def get_PSTH(sp,cc,pre_onset=10):
    ''' I dont think the PSTH is good because the number of 
    contacts occuring dimnishes as time goes on, 
    so a reduction in spike rate could be due to that. 
    Maybe need to normalize time?'''

    locked_st = get_fr_by_contact(sp,cc,pre_onset=pre_onset,post_offset=0)[1]
    all_spikes = join_locked_st(locked_st)

def get_autocorr(sp_neo):
    return(cross_correlation_histogram(BinnedSpikeTrain(sp_neo, binsize=ms), BinnedSpikeTrain(sp_neo, binsize=ms)))

def replaceNaNs(var,mode='zero'):
    if mode=='zero':
        var[np.isnan(var)]=0
    elif mode=='median':
        m = np.nanmedian(var,0)
        idx = np.any(np.isnan(var))
        var[np.any(np.isnan(var),1),:] = m
    else:
        raise ValueError('Wrong mode indicated. May want to impute NaNs in some instances')

def correlate_to_stim(sp_neo,var,kernel_sigmas,mode='g'):
    corr_ = np.empty(kernel_sigmas.shape[0])
    for ii,sigma in enumerate(kernel_sigmas):
        if mode=='r':
            kernel = kernels.RectangularKernel(sigma=sigma * ms)
        else:
            kernel = kernels.GaussianKernel(sigma=sigma * ms)
        r = instantaneous_rate(sp_neo, sampling_period=ms, kernel=kernel)
        corr_[ii] = np.corrcoef(r.squeeze(), var)[0,1]
    plt.plot(kernel_sigmas,corr_)
    return(corr_,kernel_sigmas)


