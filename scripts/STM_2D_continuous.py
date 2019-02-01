"""
This is a script to run the STM on the 2D data 
It will incorporate derivatives, and spike history.
It will implement crossvalidation
It will implement variable dropout
"""
from bin_model import get_params
import GLM
from GLM import get_blk_smooth
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
    
def get_X_y(fname,unit_num=0):
    varlist = ['M','FX','FY','TH']
    blk = neoUtils.get_blk(fname)
    cbool = neoUtils.get_Cbool(blk)
    X = GLM.create_design_matrix(blk,varlist)
    Xdot,Xsmooth = GLM.get_deriv(blk,blk,varlist,[0,5,9])

    X = np.concatenate([X,Xdot],axis=1)
    X = neoUtils.replace_NaNs(X,'pchip')
    X = neoUtils.replace_NaNs(X,'interp')
    scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
    X = scaler.fit_transform(X)
    y = neoUtils.get_rate_b(blk,unit_num)[1][:,np.newaxis]
    yhat = np.zeros_like(y)
    return(X,y,cbool)

def run_STM_CV(X, y, cbool,params,n_sims=1):
    num_components = 4
    num_features = 5
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
def run_dropout(fname,unit_num,params):
    X,y,cbool = get_X_y(fname,unit_num)
    num_derivs = 3 # this is a hardcoded numer of derivative smoothings to use 
    M = np.tile(np.array([1,0,0,0],dtype='bool'),num_derivs+1)
    F = np.tile(np.array([0,1,1,0],dtype='bool'),num_derivs+1)
    R = np.tile(np.array([0,0,0,1],dtype='bool'),num_derivs+1)

    # save outputs
    yhat={} # cross validated
    yhat_sim = {} # not cross validated
    model={}

    # =========== #
    # =========== #
    # Run Full
    print('Running Full')
    yhat['full'],yhat_sim['full'],model['full'] = run_STM_CV(X,y,cbool,params)
    # =========== #
    # =========== #
    # Run one Drop
    # Run noM
    print('Running no derivative')
    yhat['noD'],yhat_sim['noD'],model['noD'] = run_STM_CV(X[:,:4],y,cbool,params)
    # Run noM
    print('Running no moment')
    yhat['noM'],yhat_sim['noM'],model['noM'] = run_STM_CV(X[:,np.invert(M)],y,cbool,params)
    # Run noF
    print('Running no force')
    yhat['noF'],yhat_sim['noF'],model['noF'] = run_STM_CV(X[:,np.invert(F)],y,cbool,params)
    # Run noR
    print('Running no rotation')
    yhat['noR'],yhat_sim['noR'],model['noR'] = run_STM_CV(X[:,np.invert(R)],y,cbool,params)

    # =========== #
    # =========== #
    # Run 2 Drops

    print('Running just Derivative')
    yhat['justD'], yhat_sim['justD'],model['justD'] = run_STM_CV(X[:,4:],y,cbool,params)
    print('Running just Moment')
    yhat['justM'],yhat_sim['justM'],model['justM'] = run_STM_CV(X[:,M],y,cbool,params)
    print('Running just Force')
    yhat['justF'], yhat_sim['justF'],model['justF'] = run_STM_CV(X[:,F],y,cbool,params)
    print('Running just Rotation')
    yhat['justR'],yhat_sim['justR'],model['justR'] = run_STM_CV(X[:,R],y,cbool,params)

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
    p_save = r'/projects/p30144/_VG3D/deflections/_NEO_2D/results'
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
        yhat,yhat_sim,model = run_dropout(fname,unit_num,params)
        y = neoUtils.get_rate_b(blk,unit_num)[1]
        X, y, cbool = get_X_y(fname,  unit_num)
        root = neoUtils.get_root(blk, unit_num)
        for key in yhat.iterkeys():
            R[key],kernel_sizes = get_correlations(y,yhat[key],yhat_sim[key],cbool)


        root = neoUtils.get_root(blk,unit_num)
        np.savez(os.path.join(p_save,'{}_STM_continuous_2D.npz'.format(root)),
                 X=X,
                 y=y,
                 yhat=yhat,
                 yhat_sim=yhat_sim,
                 cbool=cbool,
                 R = R,
                 kernel_sizes=kernel_sizes,
                 params=params,
                 models=model)








