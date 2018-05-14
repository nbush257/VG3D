import pyNN
import quantities as pq
import elephant
import spikeAnalysis
import neoUtils
import os
import glob
import GLM
import numpy as np
import scipy
import continuous_model
import sklearn

"""
This module is based on Dong,..Bensmaia et al. 2013
"""

def get_X_y(fname,p_smooth,unit_num=0):
    varlist = ['M', 'F', 'TH', 'PHIE']
    blk = neoUtils.get_blk(fname)
    blk_smooth = GLM.get_blk_smooth(fname,p_smooth)

    cbool = neoUtils.get_Cbool(blk)
    X = GLM.create_design_matrix(blk,varlist)
    Xdot,Xsmooth = GLM.get_deriv(blk,blk_smooth,varlist,[9])

    X = np.concatenate([X,Xdot],axis=1)
    X = neoUtils.replace_NaNs(X,'pchip')
    X = neoUtils.replace_NaNs(X,'interp')
    scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
    X = scaler.fit_transform(X)

    y = neoUtils.get_rate_b(blk,unit_num)[1][:,np.newaxis]
    return(X,y,cbool)


def calc_IInj(X,K):
    pass

def saturation(I,params):
    """ Caclulate the saturation of the injected current"""
    I_0 = params['I_0']
    return(np.divide(
        np.multiply(I,I_0),
        np.sum(I_0,np.abs(I))
    ))

def cost_function(y,yhat,tau=5*pq.ms):
    if type(y) is not neo.core.spiketrain.SpikeTrain:
        y = spikeAnalysis.binary_to_neo_train(y)
    if type(yhat) is not neo.core.spiketrain.SpikeTrain:
        yhat = spikeAnalysis.binary_to_neo_train(yhat)

    D = elephant.spike_train_dissimilarity.van_rossum_dist([y,yhat],tau)

def init_params(X,nfilts):
    params = {'K':np,#free
              'tau':0,#free
              'A0':0,#free
              'A1':0,#free
              'a':0,#free
              'b':10.,# 1/s
              'C':150.,#pF
              'Vrest':-70.,#mV
              'THETA_inf':-30., # mV
             }
    return(params)

def run_IF(I_inj,params):
    T = len(I_inj)
    I_ind = np.zeros_like(I_inj)
    i0 = np.zeros_like(I_inj)
    i1 = np.zeros_like(I_inj)
    tau_0 = 5 # in ms
    tau_1 = 50 # in ms

    V = np.ones_like(I_inj)*params['Vrest']
    THETA = np.ones_like(I_inj)*params['THETA_inf']
    spikes=[]
    for t in range(1,T):
        #calculate Iind
        di0 = -i0[t]/tau_0
        di1 = -i1[t]/tau_1

        # update spike induced currents
        i0[t] = i0[t-1]+di0
        i1[t] = i1[t-1]+di1
        I_ind[t] = i0[t]+i1[t]

        # V'(t) = -1/tau[V(t)-V_rest]+(I(t)+I_ind(t))/C)
        dV = -np.divide(V[t]-params['Vrest'],params['tau']) + np.divide(np.sum(I_inj[t],I_ind[t]),params['C'])
        # THETA'(t) = a[V(t)-V_rest]-b[THETA(t)-THETA_inf]
        dTHETA = np.multiply(params['a'],(V[t]-params['Vrest'])) - np.multiply(params['b'],THETA[t]- params['THETA_inf'])

        # update voltge
        V[t] = V[t-1]+dV
        # update Threshold?
        THETA[t] = THETA[t-1] + dTHETA

           # If exceed threshold
        if V[t]>THETA[t]:
            i0[t] = i0[t]+A0
            i1[t] = i1[t]+A1
            I_ind[t] = i0[t]+i1[t]
            V[t] = params['Vrest']
            THETA[t] = np.max(THETA[t],params['THETA_inf'])
            spikes.append(t)


    return(V,spikes)



p_smooth = r'C:\Users\nbush257\Box Sync\__VG3D\_deflection_trials\_NEO\smooth'
fname = r'C:\Users\nbush257\Box Sync\__VG3D\_deflection_trials\_NEO\rat2017_08_FEB15_VG_D1_NEO.h5'
