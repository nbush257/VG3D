from continuous_model import run_STM_CV,get_X_y,get_correlations
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

def run_dropout(fname,p_smooth,unit_num,params,n_pcs=6):
    X,y,cbool = get_X_y(fname,p_smooth,unit_num,pca_tgl=True,n_pcs=n_pcs)

    yhat={} # cross validated
    yhat_sim = {} # not cross validated
    model={}
    for ii in range(n_pcs):
        print('Running Full{}'.format(ii))
        idx = range(ii,X.shape[1],n_pcs) 
        yhat['full{}'.format(ii)],yhat_sim['full{}'.format(ii)],model['full{}'.format(ii)]= run_STM_CV(X,y,cbool,params,n_sims=1)
        print('Running no derivative{}'.format(ii))
        yhat['noD{}'.format(ii)],yhat_sim['noD{}'.format(ii)],model['noD{}'.format(ii)] = run_STM_CV(X[:,:ii],y,cbool,params,n_sims=1)

    return(yhat,yhat_sim,model)
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
        X, y, cbool = get_X_y(fname, p_smooth,
                              unit_num,pca_tgl=True,n_pcs=6)#change to 6postest
        root = neoUtils.get_root(blk,unit_num)
        for key in yhat.iterkeys():
            R[key],kernel_sizes = get_correlations(y,yhat[key],yhat_sim[key],cbool)

        root = neoUtils.get_root(blk,unit_num)
        np.savez(os.path.join(p_save,'{}_STM_PCA.npz'.format(root)),
                 X=X,
                 y=y,
                 yhat=yhat,
                 yhat_sim=yhat_sim,
                 cbool=cbool,
                 R = R,
                 kernel_sizes=kernel_sizes,
                 params=params,
                 models=model)
