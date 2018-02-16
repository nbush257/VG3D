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
def get_Xc_yc(fname,p_smooth,unit_num):
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
    y = neoUtils.get_rate_b(blk,unit_num)[1]
    Xc = X[cbool,:]
    yc = y[cbool]
    yhat = np.zeros_like(y)
    return(Xc,yc,cbool,yhat)

def run_STM_CV(Xc,yc,cbool,yhat):
    num_components = 4
    num_features = 5
    k = 10
    KF = sklearn.model_selection.KFold(k)
    yhat_model = np.zeros(yc.shape[0])
    MODELS=[]
    count=0
    for train_index,test_index in KF.split(Xc):
        print('\t{} of {} crossvalidations'.format(count,k))
        model = cmt.models.STM(Xc.shape[1],0,
                               num_components,
                               num_features,
                               cmt.nonlinear.LogisticFunction,
                               cmt.models.Bernoulli)
        retval = model.train(Xc[train_index].T,yc[train_index].T,parameters=get_params())

        if not retval:
            print('Max_iter ({:.0f}) reached'.format(get_params()['max_iter']))
        MODELS.append(model)
        yhat_model[test_index] = model.predict(Xc[test_index].T)
    yhat[cbool] =yhat_model
    r = scipy.corrcoef(yhat[cbool].ravel(),yc.ravel())[0,1]
    print('\t\t corrcoef = {}'.format(r))
    return(r)

def run_dropout(fname,p_smooth,unit_num):
    Xc,yc,cbool,yhat = get_Xc_yc(fname,p_smooth,unit_num)

    no_M = np.array([0,0,0,1,1,1,1,1]*4,dtype='bool')
    no_F = np.array([1,1,1,0,0,0,1,1]*4,dtype='bool')
    no_R = np.array([1,1,1,1,1,1,0,0]*4,dtype='bool')

    X_noM = Xc[:,no_M]
    X_noF = Xc[:,no_F]
    X_noR = Xc[:,no_R]

    R ={}
    print('Running Full')
    R['full'] = run_STM_CV(Xc,yc,cbool,yhat)
    print('Running No Derivative')
    R['noD'] = run_STM_CV(Xc[:,:8],yc,cbool,yhat)
    print('Running No Moment')
    R['noM'] = run_STM_CV(X_noM,yc,cbool,yhat)
    print('Running No Force')
    R['noF'] = run_STM_CV(X_noF,yc,cbool,yhat)
    print('Running No Rotation')
    R['noR'] = run_STM_CV(X_noR,yc,cbool,yhat)
    return(R)

if __name__=='__main__':
    fname = sys.argv[1]
    p_smooth = r'/projects/p30144/_VG3D/deflections/_NEO'
    blk = neoUtils.get_blk(fname)
    df_head = pd.DataFrame(columns=['id','full','noD','noM','noF','noR'])
    csv_file = os.path.join(p_smooth,'continuous_model.csv')
    df_head.to_csv(csv_file,index=None)
    for unit_num in range(len(blk.channel_indexes[-1].units)):
        R = run_dropout(fname,p_smooth,unit_num)
        root = neoUtils.get_root(blk,unit_num)
        df = pd.DataFrame([R],columns=['id','full','noD','noM','noF','noR'])
        with open(csv_file,'a') as f:
            df.to_csv(f,header=False,index=False)
