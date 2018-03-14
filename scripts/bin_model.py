""" This is a script to run an STM on the binned version of
the input variables. It will incorporate derivatives, but no spike
history.
It will implement cross validation
It will implement variable dropout
"""
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
# Init ========= #
# fname = r'/media/nbush257/5446399F46398332/Users/nbush257/Box Sync/__VG3D/_deflection_trials/_NEO/rat2017_08_FEB15_VG_D1_NEO.h5'
# unit_num = 0
# binsize = 10 # number of ms to bin over
# k = 10 # number of cross validations splits
# num_components=3
# num_features=20
# varlist = ['M','F','TH','PHIE']
# p_smooth = r'/media/nbush257/5446399F46398332/Users/nbush257/Box Sync/__VG3D/_deflection_trials/_NEO/smooth'
# ============== #
def get_params():
    params = {'verbosity':0,
              'threshold':1e-9,
              'max_iter':1e6}
    return(params)


def get_blk_smooth(fname,p_smooth):
    root = os.path.splitext(os.path.basename(fname))[0]
    smooth_file = glob.glob(os.path.join(p_smooth,root+'*smooth*.h5'))
    if len(smooth_file)>1:
        raise ValueError('More than one smooth file found')
    elif len(smooth_file)==0:
        raise ValueError('No Smooth file found')

    blk = neoUtils.get_blk(smooth_file[0])

    return(blk)


def get_Xc_yc(fname,p_smooth,unit_num,binsize):
    varlist = ['M', 'F', 'TH', 'PHIE']
    blk = neoUtils.get_blk(fname)
    blk_smooth = get_blk_smooth(fname,p_smooth)

    cbool = neoUtils.get_Cbool(blk)
    X = GLM.create_design_matrix(blk,varlist)
    Xdot = GLM.get_deriv(blk,blk_smooth,varlist,[0,5,9]) #maybe only want one derivative?

    X = np.concatenate([X,Xdot],axis=1)
    X = neoUtils.replace_NaNs(X,'pchip')
    X = neoUtils.replace_NaNs(X,'interp')

    Xbin = GLM.bin_design_matrix(X,binsize=binsize)
    scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
    Xbin = scaler.fit_transform(Xbin)
    cbool_bin= GLM.bin_design_matrix(cbool[:,np.newaxis],binsize=binsize).ravel()

    y = neoUtils.concatenate_sp(blk)['cell_{}'.format(unit_num)]
    ybin = elephant.conversion.BinnedSpikeTrain(y,binsize=binsize*pq.ms).to_array().T.astype('f8')
    Xbin = Xbin[:ybin.shape[0],:]
    cbool_bin = cbool_bin[:ybin.shape[0]]
    yhat = np.zeros(ybin.shape[0])

    Xc = Xbin[cbool_bin,:]
    yc = ybin[cbool_bin,:]
    return(Xc,yc,cbool_bin,yhat)


def run_STM_CV(Xc,yc,cbool_bin,yhat):
    num_components = 3
    num_features = 24
    k = 20

    KF = sklearn.model_selection.KFold(k,shuffle=True)

    yhat_model = np.zeros(yc.shape[0])
    MODELS =[]
    count=0
    for train_index,test_index in KF.split(Xc):
        count+=1
        print('\t{} of {} crossvalidations'.format(count,k))
        model = cmt.models.STM(Xc.shape[1],0,
                               num_components,
                               num_features,
                               cmt.nonlinear.ExponentialFunction,
                               cmt.models.Poisson)
        retval = model.train(Xc[train_index,:].T,yc[train_index,:].T,parameters=get_params())
        if not retval:
            print('Max_iter ({:.0f}) reached'.format(get_params()['max_iter']))
        MODELS.append(model)
        yhat_model[test_index] = model.predict(Xc[test_index].T)

    yhat[cbool_bin] =yhat_model
    yhat[yhat>binsize]=binsize
    r = scipy.corrcoef(yhat[cbool_bin].ravel(),yc.ravel())[0,1]
    print('\t\t corrcoef = {}'.format(r))
    return(r)


def run_dropout(fname,p_smooth,unit_num,binsize=10):
    Xc,yc,cbool_bin,yhat = get_Xc_yc(fname,p_smooth,unit_num,binsize)

    no_M = np.array([0,0,0,1,1,1,1,1]*4,dtype='bool')
    no_F = np.array([1,1,1,0,0,0,1,1]*4,dtype='bool')
    no_R = np.array([1,1,1,1,1,1,0,0]*4,dtype='bool')

    X_noM = Xc[:,no_M]
    X_noF = Xc[:,no_F]
    X_noR = Xc[:,no_R]

    R =[]
    print('Running Full')
    R.append(run_STM_CV(Xc,yc,cbool_bin,yhat))
    print('Running No Derivative')
    R.append(run_STM_CV(Xc[:,:8],yc,cbool_bin,yhat))
    print('Running No Moment')
    R.append(run_STM_CV(X_noM,yc,cbool_bin,yhat))
    print('Running No Force')
    R.append(run_STM_CV(X_noF,yc,cbool_bin,yhat))
    print('Running No Rotation')
    R.append(n_STM_CV(X_noR,yc,cbool_bin,yhat))
    return(R)


if __name__=='__main__':
    fname = sys.argv[1]
    binsize = int(sys.argv[2])
    p_smooth = r'/projects/p30144/_VG3D/deflections/_NEO'
    blk = neoUtils.get_blk(fname)
    #df_head = pd.DataFrame(columns=['id','full','noD','noM','noF','noR'])
    csv_file = os.path.join(p_smooth,'{}_bin_model_correlations.csv'.format(binsize))
    #df_head.to_csv(csv_file,index=None)
    for unit_num in range(len(blk.channel_indexes[-1].units)):
        R = run_dropout(fname,p_smooth,unit_num,binsize)
        root = neoUtils.get_root(blk,unit_num)
        df = pd.DataFrame([R],columns=['id','full','noD','noM','noF','noR'])
        with open(csv_file,'a') as f:
            df.to_csv(f,header=False,index=False)
