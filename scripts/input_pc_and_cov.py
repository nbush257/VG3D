import glob
import os
import neoUtils
import numpy as np

def get_pc(blk):
    # get data
    use_flags = neoUtils.concatenate_epochs(blk)
    cbool = neoUtils.get_Cbool(blk)
    M = neoUtils.get_var(blk, 'M').magnitude
    F = neoUtils.get_var(blk, 'F').magnitude
    TH = neoUtils.get_var(blk, 'TH').magnitude
    PH = neoUtils.get_var(blk, 'PHIE').magnitude

    # center angles
    deltaTH = neoUtils.center_var(TH, use_flags)
    deltaPH = neoUtils.center_var(PH, use_flags)
    deltaTH[np.invert(cbool)] = np.nan
    deltaPH[np.invert(cbool)] = np.nan
    X = np.concatenate([M,F,deltaTH,deltaPH],axis=1)
    pc = neoUtils.applyPCA(X, cbool)[1]

    return(pc)
def batch_pc(p_load,p_save):
    ID = []
    COV=[]
    EXP_VAR=[]
    for f in glob.glob(os.path.join(p_load,'*.h5')):
        blk = neoUtils.get_blk(f)
        id = neoUtils.get_root(blk,0)[:-2]
        print('Working on {}'.format(id))
        pc = get_pc(blk)
        cov = pc.get_covariance()
        exp_var = pc.explained_variance_ratio_
        COV.append(cov)
        EXP_VAR.append(exp_var)
        ID.append(id)
    np.savez(os.path.join(p_save,'cov_exp_var.npz'),cov=COV,exp_var=EXP_VAR,id=ID)
    print('Saved PCA descriptions!')
    return None

# TODO: Nonlinear Manifold fitting
