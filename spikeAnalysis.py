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


def get_contact_sliced_trains(blk):
    '''returns mean_fr,ISIs,spiketrains for each contact interval for each cell'''
    ISI = {}
    ISI_units = pq.ms
    contact_trains = {}
    FR = {}
    FR_units = 1/pq.s
    for unit in blk.channel_indexes[-1].units:
        FR[unit.name] = np.array([]).reshape(0, 0)*1/pq.s
        tempPSTH =[]
        tempISI = []
        for train,seg in zip(unit.spiketrains,blk.segments):
            epoch = seg.epochs[0]
            seg_fr = np.zeros([len(epoch), 1], dtype='f8')*FR_units
            for ii,(start,dur) in enumerate(zip(epoch.times,epoch.durations)):
                train_slice = train.time_slice(start, start + dur)
                if len(train_slice)>0:
                    seg_fr[ii] = mean_firing_rate(train_slice)
                if len(train_slice)>2:
                    intervals = isi(np.array(train_slice)*ISI_units)
                else:
                    intervals = [np.nan]*ISI_units
                    # b = binarize(train_slice, sampling_rate=pq.kHz)
                tempPSTH.append(train_slice)
                tempISI.append(intervals)
            FR[unit.name] = np.append(FR[unit.name], seg_fr) * FR_units
        contact_trains[unit.name] = tempPSTH
        ISI[unit.name] = tempISI
    return FR,ISI,contact_trains

def get_binary_trains(trains,norm_length=True):
    '''takes a list of spike trains and computes binary spike trains for all
    can return a matrix with the number of columns equal to the longest train.

    Might be useful to normalize based on the number of observations of each time point'''


    b = []
    if norm_length:
        durations = np.zeros(len(trains),dtype='int64')
        for ii,train in enumerate(trains):
            duration = train.t_stop - train.t_start
            duration.units = pq.ms
            durations[ii] = int(duration)
        max_duration = np.max(durations)+1
        b = np.zeros([len(trains),max_duration],dtype='bool')


    for ii,train in enumerate(trains):
        if len(train) == 0:
            duration = train.t_stop-train.t_start
            duration.units=pq.ms
            duration = int(duration)
            if norm_length:
                b[ii,:duration] = np.zeros(duration,dtype='bool')
            else:
                b.append(np.zeros(duration,dtype='bool'))

        else:
            if norm_length:
                b_temp = binarize(train, sampling_rate=pq.kHz)
                b[ii,:len(b_temp)]=b_temp
            else:
                b.append(binarize(train,sampling_rate=pq.kHz))
    if norm_length:
        # return the durations if the length is kept consistent across all
        return b,durations
    else:
        return b
