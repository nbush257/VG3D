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

def run_dropout(fname,p_smooth,unit_num,params):
    X,y,cbool = get_X_y(fname,p_smooth,unit_num,pca_tgl=True,n_pcs=3)

    yhat={} # cross validated
    yhat_sim = {} # not cross validated
    print('Running Full')
    yhat['full'],yhat_sim['full'] = run_STM_CV(X,y,cbool,params)
    print('Running no derivative')
    yhat['noD'],yhat_sim['noD'] = run_STM_CV(X[:,:3],y,cbool,params)

    return(yhat,yhat_sim)
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
        yhat,yhat_sim = run_dropout(fname,p_smooth,unit_num,params)
        y = neoUtils.get_rate_b(blk,unit_num)[1]
        X, y, cbool = get_X_y(fname, p_smooth, unit_num,pca_tgl=True,n_pcs=3)
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
                 params=params)


