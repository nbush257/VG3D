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
def get_X_y(fname,p_smooth,unit_num):
    varlist = ['M', 'F', 'TH', 'PHIE']
    blk = neoUtils.get_blk(fname)
    blk_smooth = get_blk_smooth(fname,p_smooth)

    cbool = neoUtils.get_Cbool(blk)
    X = GLM.create_design_matrix(blk,varlist)
    Xdot = GLM.get_deriv(blk,blk_smooth,varlist,[0,5,9])

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

def run_STM_CV(X, y, cbool):
    num_components = 4
    num_features = 5
    n_sims=10
    k = 10
    KF = sklearn.model_selection.KFold(k,shuffle=True)
    yhat = np.zeros(y.shape[0])
    MODELS=[]
    count=0
    y = y.astype('f8')
    y[np.invert(cbool),:]=0
    yhat_sim = np.zeros([y.shape[0],n_sims])
    params = {'verbosity':1,
              'threshold':1e-7,
              'max_iter':1e3,
              'regularize_weights':{
                  'strength': 0,
                  'norm':'L2'}
              }

    # TODO: simulate all!! DO we want to crossvalidate it?
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
            print('Max_iter ({:.0f}) reached'.format(get_params()['max_iter']))
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

    print('\t\t corrcoef = {}'.format(r))
    return(yhat,yhat_sim)

def run_dropout(fname,p_smooth,unit_num):
    X,y,cbool = get_X_y(fname,p_smooth,unit_num)

    no_M = np.array([0,0,0,1,1,1,1,1]*4,dtype='bool')
    no_F = np.array([1,1,1,0,0,0,1,1]*4,dtype='bool')
    no_R = np.array([1,1,1,1,1,1,0,0]*4,dtype='bool')
    # TODO: remove c

    X_noM = X[:,no_M]
    X_noF = X[:,no_F]
    X_noR = X[:,no_R]

    # save outputs
    yhat={} # cross validated
    yhat_sim = {} # not cross validated
    print('Running Full')
    yhat['full'],yhat_sim['full'] = run_STM_CV(X,y,cbool)
    print('Running No Derivative')
    yhat['noD'], yhat_sim['noD'] = run_STM_CV(X[:,:8],y,cbool)
    print('Running No Moment')
    yhat['noM'], yhat_sim['noM'] = run_STM_CV(X_noM,y,cbool)
    print('Running No Force')
    yhat['noF'], yhat_sim['noF'] = run_STM_CV(X_noF,yc,cbool)
    print('Running No Rotation')
    yhat['noR'], yhat_sim['noR'] = run_STM_CV(X_noR,yc,cbool)

    return(yhat,yhat_sim)

if __name__=='__main__':
    fname = sys.argv[1]
    p_smooth = r'/projects/p30144/_VG3D/deflections/_NEO'
    blk = neoUtils.get_blk(fname)
    df_head = pd.DataFrame(columns=['id','full','noD','noM','noF','noR'])
    csv_file = os.path.join(p_smooth,'continuous_model.csv')
    df_head.to_csv(csv_file,index=None)
    for unit_num in range(len(blk.channel_indexes[-1].units)):
        yhat,yhat_sim = run_dropout(fname,p_smooth,unit_num)
        root = neoUtils.get_root(blk,unit_num)
        df = pd.DataFrame([R],columns=['id','full','noD','noM','noF','noR'])
        with open(csv_file,'a') as f:
            df.to_csv(f,header=False,index=False)
