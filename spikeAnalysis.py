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


def get_fr_by_contact(blk):
    ''' calculates the mean firing rate
    of each contact interval
    for each cell
    across all segments in a block'''
    FR = {}
    for unit in blk.channel_indexes[-1].units:
        FR[unit.name]=np.array([]).reshape(0,0)*1/s
        for seg,train in zip(blk.segments,unit.spiketrains):
            seg_fr = np.zeros([len(seg.epochs[0]),1],dtype='f8')*1/s
            for ii,(start, dur) in enumerate(zip(seg.epochs[0],seg.epochs[0].durations)):
                sp = train.time_slice(start,start+dur)
                seg_fr[ii] = mean_firing_rate(sp)
            FR[unit.name] = np.append(FR[unit.name],seg_fr)
    print('Calculated per contact firing rate for {}'.format(FR.keys()))
    return FR


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


