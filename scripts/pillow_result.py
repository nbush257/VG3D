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


def get_corr(y,yhat,cbool,kernel_sizes=None):
    """
    :param y: the observed spike train as a binary vector
    :param yhat: the predicted spike train as a float vector
    :param cbool: a boolean vector of contact times
    :param kernel_sizes: the sizes of the Gaussian smoothing kernel to test.
                                Only pass an argument if you have a very good reason.
    :return R, rates
    """
    if kernel_sizes==None:
        kernel_sizes = np.power(2, np.arange(1, 10))

    spt = spikeAnalysis.binary_to_neo_train(y)
    rates = [elephant.statistics.instantaneous_rate(spt,pq.ms,
                                                   elephant.kernels.GaussianKernel(sigma*pq.ms)
                                                   ) for sigma in kernel_sizes]
    R = [scipy.corrcoef(x.magnitude.ravel()[cbool],yhat[cbool])[0,1] for x in rates]

    return(R,rates,kernel_sizes)


def best_deriv_drops_r(fname):
    """
    Takes a matfile that has fitted pillow drops,
        calculates the correlation between the predicted spiking and the
        observed rate at different smoothing, and outputs a pandas dataframe
        with the rates and kernels
    :param fname: a matfile with the pillow results
    :return df: a dataframe where each column is a different model correlation,
                    along with kernel sizes and id
    """
    dat = sio.loadmat(fname,struct_as_record=False,squeeze_me=True)
    output = dat['output']
    y = dat['y'].ravel()
    cbool = dat['cbool'].astype('bool').ravel()
    R = {}
    model_map = {'Full':'full',
                 'Drop_derivative':'noD',
                 'Drop_force':'noF',
                 'Drop_moment':'noM',
                 'Drop_rotation':'noR',
                 'Just_derivative':'justD',
                 'Just_force':'justF',
                 'Just_moment':'justM',
                 'Just_rotation':'justR',
                 }
    for model in output._fieldnames:
        print('\t'+model)
        yhat = output.__getattribute__(model).yhat
        R[model],_,kernels = get_corr(y,yhat,cbool)

    R['kernels'] = kernels
    df = pd.DataFrame(R)
    df['id'] = os.path.basename(fname)[:10]
    df = df.rename(columns=model_map)
    return(df)


def best_deriv_2D_r(fname):
    """
    Calculates the correlation between the 2D pillow models
    and the observed spiking
    :param fname:
    :return df: Dataframe containinf the pearson correlation for the 2D models
    """
    dat = sio.loadmat(fname,struct_as_record=False,squeeze_me=True)
    y = dat['y'].ravel()
    cbool = dat['cbool'].astype('bool').ravel()
    yhat = dat['yhat'].ravel()
    df = pd.DataFrame()
    df['2D'],_,df['kernels'] = get_corr(y,yhat,cbool)
    df['id'] = os.path.basename(fname)[:10]

    return(df)


def batch_best_deriv_drops():
    """
    Get the Pearson correlations for the pillow drop analyses for all files
    Saves to a csv in the results directory
    :return None:

    """

    p_load = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\pillowX\best_smoothing_deriv')
    p_save = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')
    DF = pd.DataFrame()
    for f in glob.glob(os.path.join(p_load,'*drops.mat')):
        print('Working on {}'.format(os.path.basename(f)[:10]))
        df = best_deriv_drops_r(f)
        DF = DF.append(df)
    DF.to_csv(os.path.join(p_save,'pillow_best_deriv_drop_correlations.csv'),index=False)
    return 0


def batch_best_deriv_2D():
    """
    Get the Pearson correlations for the pillow 2D analyses for all files
    Saves to a csv in the results directory
    :return None:

    """

    p_load = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\pillowX\2d_best_smoothing')
    p_save = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')
    DF = pd.DataFrame()
    for f in glob.glob(os.path.join(p_load,'*MID.mat')):
        print('Working on {}'.format(os.path.basename(f)[:10]))
        df = best_deriv_2D_r(f)
        DF = DF.append(df)
    DF.to_csv(os.path.join(p_save,'pillow_best_deriv_2D_correlations.csv'),index=False)
    return 0


def batch_mat_to_dataframes():
    p_load = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\pillowX\55ms_smoothing_deriv')
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

        # Load data
        dat = sio.loadmat(f,struct_as_record=False,squeeze_me=True)
        y = dat['y'].ravel()
        cbool = dat['cbool'].astype('bool').ravel()
        yhat = dat['yhat'].ravel()

        # calculate R
        R,rates,kernels = get_corr(y,yhat,cbool)
        weight_dict = get_canonical_angles(f)
        canonical_angles = weight_dict['canonical_angles']
        df_r = pd.DataFrame()
        df_r['R'] = R
        df_r['kernels'] = kernels
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


def orthogonality_of_K(fname,p_save=None):
    """
    This function calculates how orthogonal each of the individual
    filters of the pillow model are from each other
    :return:
    """
    if p_save is None:
        p_save = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')

    df = pd.read_csv(fname,index_col=0)
    is_stim = pd.read_csv(os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results\cell_id_stim_responsive.csv'))
    df = df.merge(is_stim,on='id')
    df = df[df.stim_responsive]
    DF_OUT = pd.DataFrame()
    ORTHO_MAT = np.empty([3,3,len(df.id.unique())])

    for cell_id,cell in enumerate(df.id.unique()):
        df_out = pd.DataFrame()
        sub_df = df[df.id==cell]
        X = sub_df[['Filter_0','Filter_1','Filter_2']].as_matrix()
        norms = np.linalg.norm(X,2,axis=0)
        X = X/norms
        ortho_mat=np.empty((X.shape[1],X.shape[1]))
        for ii,x in enumerate(X.T):
            for jj,y in enumerate(X.T):
                ortho_mat[ii,jj]=np.abs(np.dot(x,y))
        ORTHO_MAT[:,:,cell_id] = ortho_mat
        df_out['norm0'] = [norms[0]]
        df_out['norm1'] = [norms[1]]
        df_out['norm2'] = [norms[2]]
        df_out['id'] = cell
        DF_OUT = DF_OUT.append(df_out)
    DF_OUT.to_csv(os.path.join(p_save,'MID55ms_weight_vector_norms.csv'),index=False)
    np.save(os.path.join(p_save,'MID55ms_weight_dot_products_normed.npy'),ortho_mat)





def neural_participation_ratios(fname,p_save=None):
    """
    This function will calulate the participation ratios
    for each of the fit neural vectors from the Pillow models
    :return:
    """
    if p_save is None:
        p_save = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')

    df = pd.read_csv(fname,index_col=0)
    is_stim = pd.read_csv(os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results\cell_id_stim_responsive.csv'))
    df = df.merge(is_stim,on='id')
    df = df[df.stim_responsive]
    for cell_id,cell in enumerate(df.id.unique()):
        sub_df = df[df.id==cell]
        X = sub_df[['Filter_0','Filter_1','Filter_2']].as_matrix()
        norms = np.linalg.norm(X,2,axis=0)
        X = X/norms
        for u in X.T:
            pass
        #TODO: waiting for SARA











if __name__=='__main__':
    batch_mat_to_dataframes()

