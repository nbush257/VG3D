import glob
import sys
import os
import neoUtils
import numpy as np
import sklearn
try:
    import tensorflow
    tensorflow_installed=True
except ImportError:
    tensorflow_installed=False
    pass

def get_X(blk):
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
    X = np.concatenate([M, F, deltaTH, deltaPH], axis=1)
    return(X)


def get_pc(blk):
    '''
    apply PCA to the input data
    :param blk: 
    :return: principal components structure
    '''    
    cbool = neoUtils.get_Cbool(blk)
    X = get_X(blk)
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
    ID = np.array(ID)
    COV = np.array(COV)
    EXP_VAR = np.array(EXP_VAR)
    COV = np.moveaxis(COV,[0,1,2],[2,0,1])
    EXP_VAR = EXP_VAR.T
    var_labels = ['Mx','My','Mz','Fx','Fy','Fz','TH','PHI']

    np.savez(os.path.join(p_save,'cov_exp_var.npz'),
             cov=COV,
             exp_var=EXP_VAR,
             id=ID,
             var_labels=var_labels)

    print('Saved PCA descriptions!')
    return None

def manifold_fit(blk,sub_samp=4,n_components=2,method='ltsa'):
    '''
    Fit the input data to a LLE manifold
    :param blk:
    :return:
    '''
    X = get_X(blk)
    cbool=neoUtils.get_Cbool(blk)
    X[np.invert(cbool),:]=np.nan
    idx = np.all(np.isfinite(X),axis=1)
    scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
    X[idx,:] = scaler.fit_transform(X[idx,:])

    LLE = sklearn.manifold.LocallyLinearEmbedding(n_neighbors=10,method=method,n_jobs=-1,n_components=n_components,eigen_solver='dense')
    X_sub = X[idx,:]
    # samp = np.random.choice(X_sub.shape[0],n_pts,replace=False)
    samp = np.arange(0,X_sub.shape[0],sub_samp)
    X_sub = X_sub[samp,:]

    LLE.fit(X_sub)
    Y = np.empty([X.shape[0],n_components],dtype='f8')
    Y[:] = np.nan
    Y[idx,:] = LLE.transform(X[idx,:])

    return(LLE,Y)


if tensorflow_installed:
    def auto_encoder(blk):
        pass

if __name__=='__main__':
    fname = sys.argv[1]
    blk = neoUtils.get_blk(fname)
    X = get_X(blk)
    LLE,Y = manifold_fit(blk)
    outname = os.path.splitext(fname)[0]+'_manifold.npz'
    print(outname)
    np.savez(outname,LLE=LLE,Y=Y,X=X)

