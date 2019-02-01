import neo
import glob
import neoUtils
import os
import sys
import numpy as np
import scipy
import scipy.io.matlab as sio
import spikeAnalysis
import varTuning
import quantities as pq
import pyentropy as pye

def get_PS_given_R(blk,unit_num=0):
    if True:
        raise Exception('This doesnt work yet')

    CP = neoUtils.get_var(blk,'CP')
    S = float(blk.annotations['s'][2:-1])
    CP/=S

    FR = neoUtils.get_rate_b(blk,unit_num=unit_num,sigma=2*pq.ms)[1]
    spiked = np.logical_and(np.all(np.isfinite(CP),axis=1),FR)
    idx = np.all(np.isfinite(CP),axis=1)


    PR_S,edges = np.histogramdd(CP.magnitude[spiked,:],bins=50)
    PS,edges = np.histogramdd(CP.magnitude[idx,:],bins=50)

    return(post)





def ent_analyses(blk,X_disc=128,Y_disc=64):
    CP = neoUtils.get_var(blk,'CP')
    S = float(blk.annotations['s'][2:-1])
    CP/=S
    CP = CP.magnitude
    idx = np.all(np.isfinite(CP),axis=1)
    s = np.empty_like(CP)
    s[:] = np.nan
    s[idx, :] = pye.quantise(CP[idx, :], X_disc, uniform='bins')[0]
    FR = neoUtils.get_rate_b(blk,unit_num=unit_num,sigma=2*pq.ms)[0]
    FR = pye.quantise(FR,Y_disc,uniform='bins')[0]

    idx = np.all(np.isfinite(s),axis=1)
    X = s.astype('int64').T[:,idx]
    Y = FR[np.newaxis,idx]
    DS = pye.DiscreteSystem(X, (X.shape[0], bins), Y, (1, bins))
    DS.calculate_entropies()

    #TODO: I have created a discrete FR and Stimulus, now I need to perform the actual entropy calcs
    if True:
        raise Exception('This is not done')





