"""
This is a script to run the STM without binning of the input.
It will incorporate derivatives, and spike history.
It will implement crossvalidation
It will implement variable dropout
"""
from bin_model import get_params, get_blk_smooth
import GLM
import neoUtils
import elephant
import cmt.models
import cmt.nonlinear
import sklearn
import numpy as np
import quantities as pq
import scipy
import os
import sys
import glob
import pandas as pd
import cmt.tools
import neo
def get_X_y(fname,p_smooth,unit_num,pca_tgl=False,n_pcs=3):
    varlist = ['M', 'F', 'TH', 'PHIE']
    blk = neoUtils.get_blk(fname)
    blk_smooth = get_blk_smooth(fname,p_smooth)

    cbool = neoUtils.get_Cbool(blk)
    X = GLM.create_design_matrix(blk,varlist)
    Xdot,Xsmooth = GLM.get_deriv(blk,blk_smooth,varlist,[0,5,9])
    # if using the PCA decomposition of the inputs:
    if pca_tgl:

        X = neoUtils.replace_NaNs(X,'pchip')
        X = neoUtils.replace_NaNs(X,'interp')

        Xsmooth = neoUtils.replace_NaNs(Xsmooth,'pchip')
        Xsmooth = neoUtils.replace_NaNs(Xsmooth,'interp')

        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        X = scaler.fit_transform(X)

        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        Xsmooth = scaler.fit_transform(Xsmooth)

        pca = sklearn.decomposition.PCA()
        X_pc = pca.fit_transform(X)[:,:n_pcs]
        pca = sklearn.decomposition.PCA()
        Xs_pc = pca.fit_transform(Xsmooth)[:,:n_pcs]
        zero_pad = np.zeros([1,n_pcs])
        Xd_pc = np.diff(np.concatenate([zero_pad,Xs_pc],axis=0),axis=0)
        X = np.concatenate([X_pc,Xd_pc],axis=1)

        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        X = scaler.fit_transform(X)
    else:
        X = np.concatenate([X,Xdot],axis=1)
        X = neoUtils.replace_NaNs(X,'pchip')
        X = neoUtils.replace_NaNs(X,'interp')
        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        X = scaler.fit_transform(X)


    y = neoUtils.get_rate_b(blk,unit_num)[1][:,np.newaxis]
    # Xc = X[cbool,:]
    # yc = y[cbool]
    yhat = np.zeros_like(y)
    return(X,y,cbool)

def run_STM_CV(X, y, cbool,params,n_sims=1):
    num_components = 4
    num_features = 20
    k =10  
    KF = sklearn.model_selection.KFold(k,shuffle=True)
    yhat = np.zeros(y.shape[0])
    MODELS=[]
    count=0
    y = y.astype('f8')
    y[np.invert(cbool),:]=0
    yhat_sim = np.zeros([y.shape[0],n_sims])

    for train_index,test_index in KF.split(X):
        count+=1
        print('\t{} of {} crossvalidations'.format(count,k))
        model = cmt.models.STM(X.shape[1], 0,
                               num_components,
                               num_features,
                               cmt.nonlinear.LogisticFunction,
                               cmt.models.Bernoulli)
        retval = model.train(X[train_index].T, y[train_index].T, parameters=params)

        if not retval:
            print('Max_iter ({:.0f}) reached'.format(params['max_iter']))
        MODELS.append(model)
        yhat[test_index] = model.predict(X[test_index].T)

    model = cmt.models.STM(X.shape[1], 0,
                           num_components,
                           num_features,
                           cmt.nonlinear.LogisticFunction,
                           cmt.models.Bernoulli)
    retval = model.train(X.T, y.T, parameters=params)
    yhat_sim = np.array([cmt.tools.sample_spike_train(X.T,
                                                        model,
                                                        spike_history=-5)
                           for x in range(n_sims)]).squeeze().T


    yhat = yhat.T

    return(yhat,yhat_sim,model)

def run_dropout(fname,p_smooth,unit_num,params):
    X,y,cbool = get_X_y(fname,p_smooth,unit_num)

    no_M = np.array([0,0,0,1,1,1,1,1]*4,dtype='bool')
    no_F = np.array([1,1,1,0,0,0,1,1]*4,dtype='bool')
    no_R = np.array([1,1,1,1,1,1,0,0]*4,dtype='bool')
    
    just_M = np.array([1,1,1,0,0,0,0,0]*4,dtype='bool')
    just_F = np.array([0,0,0,1,1,1,0,0]*4,dtype='bool')
    just_R = np.array([0,0,0,0,0,0,1,1]*4,dtype='bool')

    X_noM = X[:,no_M]
    X_noF = X[:,no_F]
    X_noR = X[:,no_R]
    
    X_just_M = X[:,just_M]
    X_just_F = X[:,just_F]
    X_just_R = X[:,just_R]

    # save outputs
    yhat={} # cross validated
    yhat_sim = {} # not cross validated
    model={}
    print('Running Full')
    yhat['full'],yhat_sim['full'],model['full'] = run_STM_CV(X,y,cbool,params)

    # Run one drops
    print('Running No Derivative')
    yhat['noD'], yhat_sim['noD'],model['noD'] = run_STM_CV(X[:,:8],y,cbool,params)
    print('Running No Moment')
    yhat['noM'], yhat_sim['noM'],model['noM'] = run_STM_CV(X_noM,y,cbool,params)
    print('Running No Force')
    yhat['noF'], yhat_sim['noF'],model['noF'] = run_STM_CV(X_noF,y,cbool,params)
    print('Running No Rotation')
    yhat['noR'], yhat_sim['noR'],model['noR'] = run_STM_CV(X_noR,y,cbool,params)
    # Run two drops
    print('Running just Derivative') 
    yhat['justD'], yhat_sim['justD'],model['justD'] = run_STM_CV(X[:,8:],y,cbool,params)
    print('Running just Moment')
    yhat['justM'], yhat_sim['justM'],model['justD'] = run_STM_CV(X_just_M,y,cbool,params)
    print('Running just Force')
    yhat['justF'], yhat_sim['justF'],model['justF'] = run_STM_CV(X_just_F,y,cbool,params)
    print('Running just Rotation')
    yhat['justR'], yhat_sim['justR'],model['justR'] = run_STM_CV(X_just_R,y,cbool,params)

    return(yhat,yhat_sim,model)
def get_correlations(y,yhat,yhat_sim,cbool,kernels=np.power(2,range(1,10))):
    if yhat.ndim==1:
        yhat = yhat[np.newaxis]
    if yhat_sim.ndim==1:
        yhat_sim = yhat_sim[:,np.newaxis]
    spt = neo.SpikeTrain(np.where(y)[0]*pq.ms,sampling_rate=pq.kHz,t_stop=y.shape[0]*pq.ms)

    rate = [elephant.statistics.instantaneous_rate(spt,
                                                   sampling_period=pq.ms,
                                                   kernel=elephant.kernels.GaussianKernel(x*pq.ms))
            for x in kernels]
    R = {}
    R['yhat'] = [scipy.corrcoef(x.magnitude.ravel()[cbool],yhat.ravel()[cbool])[0,1] for x in rate]
    sim_rate = np.mean(yhat_sim,axis=1)
    R['yhat_sim'] = [scipy.corrcoef(x.magnitude.ravel()[cbool], sim_rate.ravel()[cbool])[0, 1] for x in rate]
    return(R,kernels)

if __name__=='__main__':
    fname = sys.argv[1]
    p_smooth = r'/projects/p30144/_VG3D/deflections/_NEO'
    p_save = r'/projects/p30144/_VG3D/deflections/_NEO/results'
    blk = neoUtils.get_blk(fname)
    params = {'verbosity': 0,
              'threshold': 1e-8,
              'max_iter': 1e5,
              'regularize_weights': {
                  'strength': 1e-3,
                  'norm': 'L2'}
              }
    for unit_num in range(len(blk.channel_indexes[-1].units)):
        R = {}
        yhat,yhat_sim,model = run_dropout(fname,p_smooth,unit_num,params)
        y = neoUtils.get_rate_b(blk,unit_num)[1]
        X, y, cbool = get_X_y(fname, p_smooth, unit_num)
        root = neoUtils.get_root(blk,unit_num)
        for key in yhat.iterkeys():
            R[key],kernel_sizes = get_correlations(y,yhat[key],yhat_sim[key],cbool)


        root = neoUtils.get_root(blk,unit_num)
        np.savez(os.path.join(p_save,'{}_STM_continuous_20feats.npz'.format(root)),
                 X=X,
                 y=y,
                 yhat=yhat,
                 yhat_sim=yhat_sim,
                 cbool=cbool,
                 R = R,
                 kernel_sizes=kernel_sizes,
                 params=params,
                 models=model)

