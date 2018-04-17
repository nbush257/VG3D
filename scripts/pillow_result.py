import sklearn
import pandas as pd
import elephant
import scipy
import quantities as pq
import numpy as np
import spikeAnalysis
import glob
import scipy.io.matlab as sio
import get_whisker_PCA
import os

def get_corr(fname,kernel_sizes):
    """
    Get the correlation between the model prection
    and the observed spike rates
    :param fname:
    :return:
    """
    dat = sio.loadmat(fname,struct_as_record=False,squeeze_me=True)
    y = dat['y'].ravel()
    cbool = dat['cbool'].astype('bool').ravel()
    X = dat['X']
    yhat = dat['yhat'].ravel()
    spt = spikeAnalysis.binary_to_neo_train(y)

    rates = [elephant.statistics.instantaneous_rate(spt,pq.ms,
                                                   elephant.kernels.GaussianKernel(sigma*pq.ms)
                                                   ) for sigma in kernel_sizes]
    R = [scipy.corrcoef(x.magnitude.ravel()[cbool],yhat[cbool])[0,1] for x in rates]
    return(R,rates)


def get_canonical_angles(fname):
    """
    Calculate the canonical angles between the PCA decomposition
    of the input space and the Pillow filter vectors
    :param fname: name of the pillow mid mat file
    :return:
    """
    dat = sio.loadmat(fname,struct_as_record=False,squeeze_me=True)
    X = dat['X']
    cbool = dat['cbool'].astype('bool')
    weights = {}
    K = dat['ppcbf_avg'].k
    Ko = scipy.linalg.orth(K)
    pc = sklearn.decomposition.PCA()
    pc.fit(X[cbool,:])
    eigenvectors = pc.components_[:3,:]
    canonical_angles = np.linalg.svd(np.dot(eigenvectors,Ko))[1]
    # Pack the vars
    weights['K'] = K
    weights['Ko'] = Ko
    weights['pc'] = pc
    weights['eigenvectors'] = eigenvectors
    weights['canonical_angles']=canonical_angles
    return(weights)


def batch_mat_to_dataframes():
    p_load = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\pillowX')
    kernel_sizes = np.power(2,np.arange(1,10))
    varnames = ['Mx','My','Mz','Fx','Fy','Fz','TH','PHI',
                'Mxdot', 'Mydot', 'Mzdot', 'Fxdot', 'Fydot', 'Fzdot', 'THdot', 'PHIdot']

    DF_canonical = pd.DataFrame()
    DF_performance = pd.DataFrame()
    DF_pillow_weights = pd.DataFrame()
    DF_pillow_weights_normed = pd.DataFrame()
    pca_all_whiskers = []
    for f in glob.glob(os.path.join(p_load,'*MID.mat')):
        id = os.path.basename(f)[:10]
        print('Working on {}'.format(id))
        R,rates = get_corr(f,kernel_sizes)
        weight_dict = get_canonical_angles(f)
        canonical_angles = weight_dict['canonical_angles']
        df_r = pd.DataFrame()
        df_r['R'] = R
        df_r['kernel_sizes'] = kernel_sizes
        df_r['id'] = id

        df_canonical = pd.DataFrame()
        df_canonical['Angle0'] = [canonical_angles[0]]
        df_canonical['Angle1'] = [canonical_angles[1]]
        df_canonical['Angle2'] = [canonical_angles[2]]
        df_canonical['id'] = id

        df_pillow_weights = pd.DataFrame()
        df_pillow_weights_normed = pd.DataFrame()
        for ii in range(3):
            df_pillow_weights['Filter_{}'.format(ii)] = weight_dict['K'][:,ii]
        df_pillow_weights['var'] = varnames
        df_pillow_weights['id'] = id

        df_pillow_weights_normed = pd.DataFrame()
        for ii in range(3):
            df_pillow_weights_normed['Filter_{}'.format(ii)] = weight_dict['Ko'][:,ii]
        df_pillow_weights_normed['var'] = varnames
        df_pillow_weights_normed['id'] = id

        DF_pillow_weights = DF_pillow_weights.append(df_pillow_weights)
        DF_pillow_weights_normed = DF_pillow_weights_normed.append(df_pillow_weights_normed)
        DF_canonical = DF_canonical.append(df_canonical)
        DF_performance = DF_performance.append(df_r)
        pca_all_whiskers.append(weight_dict['pc'])


    DF_performance.to_csv(os.path.join(p_load,'pillow_MID_performance.csv'))
    DF_canonical.to_csv(os.path.join(p_load,'pillow_MID_canonical_angles_weights_against_pca.csv'))
    DF_pillow_weights.to_csv(os.path.join(p_load,'pillow_MID_weights_raw.csv'))
    DF_pillow_weights_normed.to_csv(os.path.join(p_load,'pillow_MID_weights_orthogonalized.csv'))

    np.save(os.path.join(p_load,'input_pca.npy'),pca_all_whiskers)

if __name__=='__main__':
    batch_mat_to_dataframes()

