import neo
import entropy
import pyentropy as pye
import neoUtils
import numpy as np
import pandas as pd
import os
import glob
import GLM
import scipy
import quantities as pq
import elephant
def joint_pr_given_s(var1,var2,sp,cbool,bins=None,min_obs=5):

    ''' Returns the noramlized response histogram of two variables
    INPUTS:     var1,var2 -- the two variables on which to plot the joint histogram. Must be either 1D numpy or column vector
                sp -- either a neo spike train, or a numpy array. The numpy array can be a continuous rate estimate
                nbins -- number of bins, or boundaries of bins to make the histograms
                min_obs -- minimum number of observations of the prior to count as an instance. If less than min obs, returns nan for that bin
    OUTPUS:     bayesm -- a masked joint histogram hiehgts
                var1_edges = bin edges on the first variable
                var2_edges = bin edges on the second variable
    '''
    # handle bins -- could probably be cleaned up NEB
    if type(bins)==int:
        bins = [bins,bins]
    elif type(bins)==list:
        pass
    else:
        bins = [50,50]

    if var1.ndim==2:
        if var1.shape[1] == 1:
            var1 = var1.ravel()
        else:
            raise Exception('var1 must be able to be unambiguously converted into a vector')
    if var2.ndim==2:
        if var2.shape[1] == 1:
            var2 = var2.ravel()
        else:
            raise Exception('var2 must be able to be unambiguously converted into a vector')
    pass
    # TODO: Fix this math
    pass
    var1[np.invert(cbool)] = np.nan
    var2[np.invert(cbool)] = np.nan
    not_nan_mask = np.logical_and(np.isfinite(var1), np.isfinite(var2))
    not_nan_mask = np.logical_and(not_nan_mask,cbool)
    not_nan = np.where(not_nan_mask)[0]

    ps,var1_edges,var2_edges= np.histogram2d(var1[not_nan_mask],var2[not_nan_mask],bins=bins)
    ps/=np.sum(not_nan_mask)

    spt = sp.times.magnitude.astype('int')
    idx = [x for x in spt if x in not_nan]
    ps_r1 = np.histogram2d(var1[idx], var2[idx], bins=[var1_edges,var2_edges])[0]/len(idx)
    pr1_s = ps_r1/ps

def pr_given_s(var_in,sp,cbool,bins=None,min_obs=5):
    if bins is None:
        bins = 50
    else:
        pass
    EDGES = []
    PS = []
    PS_R1 = []
    PR1_S = []
    for ii in range(var_in.shape[1]):
        var = var_in[:,ii].copy()

        var[np.invert(cbool)] = np.nan
        not_nan_mask = np.isfinite(var)
        not_nan = np.where(not_nan_mask)[0]
        ps,var1_edges= np.histogram(var[not_nan_mask],bins=bins)
        ps = ps.astype('f8')
        PS.append(ps)
        # ps/=float(np.sum(not_nan_mask))

        spt = sp.times.magnitude.astype('int')
        idx = [x for x in spt if x in not_nan]
        ps_r1 = np.histogram(var[idx], bins=var1_edges)[0]/float(len(idx))
        keep = ps_r1>0
        pr1_s = ps_r1[keep]/ps[keep]
        EDGES.append(var1_edges[keep])
        PS_R1.append(ps_r1[keep])
        PR1_S.append(pr1_s/np.sum(pr1_s))
    # entropy = [-np.mean(np.log(PR1_S[x])) for x in range(var_in.shape[1])]
    entropy = [-np.mean(np.log(PR1_S[x])) for x in range(var_in.shape[1])]

    return(entropy)


def calc_ent(fname,p_smooth,unit_num):
    blk = neoUtils.get_blk(fname)
    blk_smooth = GLM.get_blk_smooth(fname,p_smooth)
    varlist = ['M', 'F', 'TH', 'PHIE']
    root = neoUtils.get_root(blk,unit_num)
    print('Working on {}'.format(root))
    Xdot = GLM.get_deriv(blk,blk_smooth,varlist)[0]
    Xdot = np.reshape(Xdot,[-1,8,10])

    sp = neoUtils.concatenate_sp(blk)['cell_{}'.format(0)]
    cbool = neoUtils.get_Cbool(blk)
    entropy = []
    for ii in range(Xdot.shape[1]):
        var_in = Xdot[:,ii,:].copy()
        entropy.append(pr_given_s(var_in,sp,cbool,bins=50))
    return(entropy)

def batch_calc_entropy():
    p_load =os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO')
    p_save = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')
    p_smooth = r'K:\VG3D\_rerun_with_pad\_deflection_trials\_NEO\smooth'

    DF = pd.DataFrame()
    for ii,f in enumerate(glob.glob(os.path.join(p_load,'*.h5'))):
        if ii==0:
            continue
        blk = neoUtils.get_blk(f)
        num_units = len(blk.channel_indexes[-1].units)
        for unit_num in range(num_units):
            id =  neoUtils.get_root(blk,unit_num)
            print('Working on {}'.format(id))
            try:
                entropy = calc_ent(f,p_smooth,unit_num)
                df = pd.DataFrame()
                for ii,var in enumerate(['Mx','My','Mz','Fx','Fy','Fz','TH','PHI']):
                    df[var] = entropy[ii]
                df['id'] = id
                df['smoothing'] = np.arange(5,100,10)
                DF = DF.append(df)
            except:
                print('problem on {}'.format(id))
                continue
    DF.to_csv(os.path.join(p_save,'entropy_by_smoothing.csv'),index=False)

def get_min_entropy():
    """
    Get the minimum entropy smoothing for all variables.
    Save a csv in results
    :return:
    """
    p_load = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')
    df = pd.read_csv(os.path.join(p_load,'entropy_by_smoothing.csv'))
    is_stim = pd.read_csv(os.path.join(p_load,'cell_id_stim_responsive.csv'))
    df = df.merge(is_stim,on='id')

    min_smoothing = []
    mean_smoothing = []
    mode_smoothing = []
    for cell in df.id.unique():
       sub_df = df[df.id==cell]
       idx = sub_df.drop(['id', 'smoothing','stim_responsive'], axis=1).idxmin(0)
       min_smoothing.append(sub_df.smoothing[idx])

    mean_smoothing = [np.mean(x) for x in min_smoothing]
    mode_smoothing = [scipy.stats.mode(x)[0][0] for x in min_smoothing]

    min_smoothing = np.concatenate(min_smoothing)
    min_smoothing = np.reshape(min_smoothing,[len(df.id.unique()),8])
    order = np.argsort(np.mean(min_smoothing,axis=1))
    df_entropy =  pd.DataFrame()
    df_entropy['mode_smoothing'] = mode_smoothing
    df_entropy['mean_smoothing'] = mean_smoothing
    df_entropy['id'] = df.id.unique()
    df_entropy_all = pd.DataFrame(np.array(min_smoothing),
                                  index=df.id.unique(),
                                  columns=['Mx dot','My dot','Mz dot','Fx dot','Fy dot','Fz dot','TH dot','PH dot'])
    df_entropy_all = df_entropy_all.reset_index()
    df_entropy_all = df_entropy_all.rename(columns={'index': 'id'})
    df_entropy_all.to_csv(os.path.join(p_load,'min_smoothing_entropy.csv'),index=False)


def tuning_curve_MSE(var_in,sp,cbool,bins=None,min_obs=5):
    """
    get the residual from the mean for the tuning curve (i.e., how steep is the tuning curve)

    :param var_in:
    :param sp:
    :param cbool:
    :param bins:
    :param min_obs:
    :return:
    """
    if bins is None:
        bins = 50
    else:
        pass
    EDGES = []
    PS = []
    PS_R1 = []
    PR1_S = []
    MSE = []
    for ii in range(var_in.shape[1]):
        var = var_in[:,ii].copy()

        var[np.invert(cbool)] = np.nan
        not_nan_mask = np.isfinite(var)
        not_nan = np.where(not_nan_mask)[0]
        ps,var1_edges= np.histogram(var[not_nan_mask],bins=bins)
        ps = ps.astype('f8')
        PS.append(ps)
        # ps/=float(np.sum(not_nan_mask))

        spt = sp.times.magnitude.astype('int')
        idx = [x for x in spt if x in not_nan]
        ps_r1 = np.histogram(var[idx], bins=var1_edges)[0]/float(len(idx))
        keep = ps_r1>0
        pr1_s = ps_r1[keep]/ps[keep]
        EDGES.append(var1_edges[:-1][keep])
        PS_R1.append(ps_r1[keep])
        PR1_S.append(pr1_s)
        m = np.mean(pr1_s)
        mse = np.mean((pr1_s-m)**2)
        MSE.append(mse)

    return(MSE)

def calc_MSE(fname,p_smooth,unit_num):
    blk = neoUtils.get_blk(fname)
    blk_smooth = GLM.get_blk_smooth(fname,p_smooth)
    varlist = ['M', 'F', 'TH', 'PHIE']
    root = neoUtils.get_root(blk,unit_num)
    print('Working on {}'.format(root))
    Xdot = GLM.get_deriv(blk,blk_smooth,varlist)[0]
    Xdot = np.reshape(Xdot,[-1,8,10])

    sp = neoUtils.concatenate_sp(blk)['cell_{}'.format(0)]
    cbool = neoUtils.get_Cbool(blk)
    mse = []
    for ii in range(Xdot.shape[1]):
        var_in = Xdot[:,ii,:].copy()
        mse.append(tuning_curve_MSE(var_in,sp,cbool,bins=50))
    return(mse)

def batch_calc_MSE():
    p_load =os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO')
    p_save = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')
    p_smooth = r'E:\VG3D\_rerun_with_pad\_deflection_trials\_NEO\smooth'

    DF = pd.DataFrame()
    for ii,f in enumerate(glob.glob(os.path.join(p_load,'*.h5'))):
        if ii==0:
            continue
        blk = neoUtils.get_blk(f)
        num_units = len(blk.channel_indexes[-1].units)
        for unit_num in range(num_units):
            id =  neoUtils.get_root(blk,unit_num)
            print('Working on {}'.format(id))
            try:
                mse = calc_MSE(f,p_smooth,unit_num)
                df = pd.DataFrame()
                for ii,var in enumerate(['Mx','My','Mz','Fx','Fy','Fz','TH','PHI']):
                    df[var] = mse[ii]
                df['id'] = id
                df['smoothing'] = np.arange(5,100,10)
                DF = DF.append(df)
            except:
                print('Problem on {}'.format(id))

    DF.to_csv(os.path.join(p_save,'MSE_by_smoothing.csv'),index=False)

def get_max_MSE():
    """
    Get the maximum MSE smoothing for all variables.
    Save a csv in results
    :return:
    """
    p_load = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')
    df = pd.read_csv(os.path.join(p_load,'MSE_by_smoothing.csv'))
    is_stim = pd.read_csv(os.path.join(p_load,'cell_id_stim_responsive.csv'))
    df = df.merge(is_stim,on='id')

    max_smoothing= []
    mean_smoothing = []
    mode_smoothing = []
    for cell in df.id.unique():
        sub_df = df[df.id==cell]
        idx = sub_df.drop(['id', 'smoothing','stim_responsive'], axis=1).idxmax(0)
        max_smoothing.append(sub_df.smoothing[idx])

    mean_smoothing = [np.mean(x) for x in max_smoothing]
    mode_smoothing = [scipy.stats.mode(x)[0][0] for x in max_smoothing]

    max_smoothing = np.concatenate(max_smoothing)
    max_smoothing = np.reshape(max_smoothing,[len(df.id.unique()),8])
    order = np.argsort(np.mean(max_smoothing,axis=1))
    df_entropy =  pd.DataFrame()
    df_entropy['mode_smoothing'] = mode_smoothing
    df_entropy['mean_smoothing'] = mean_smoothing
    df_entropy['id'] = df.id.unique()
    df_entropy_all = pd.DataFrame(np.array(max_smoothing),
                                  index=df.id.unique(),
                                  columns=['Mx dot','My dot','Mz dot','Fx dot','Fy dot','Fz dot','TH dot','PH dot'])
    df_entropy_all = df_entropy_all.reset_index()
    df_entropy_all = df_entropy_all.rename(columns={'index': 'id'})
    df_entropy_all.to_csv(os.path.join(p_load,'max_smoothing_MSE.csv'),index=False)

def calc_corr(fname,p_smooth,unit_num):
    blk = neoUtils.get_blk(fname)
    blk_smooth = GLM.get_blk_smooth(fname,p_smooth)
    varlist = ['M', 'F', 'TH', 'PHIE']
    component_list = ['{}_dot'.format(x) for x in [
        'Mx','My','Mz','Fx','Fy','Fz','TH','PHI'
    ]]
    root = neoUtils.get_root(blk,unit_num)
    Xdot = GLM.get_deriv(blk,blk_smooth,varlist)[0]
    Xdot = np.reshape(Xdot,[-1,8,10])
    windows = np.arange(5,100,10)

    sp = neoUtils.concatenate_sp(blk)['cell_{}'.format(0)]
    cbool = neoUtils.get_Cbool(blk)
    corr = []
    R = []
    # loop over variables
    for ii in range(Xdot.shape[1]):
        var_in = Xdot[:,ii,:].copy()
        # loop over smoothing
        r = []
        for jj in range(var_in.shape[1]):
            kernel = elephant.kernels.GaussianKernel(pq.ms*windows[jj])
            FR = elephant.statistics.instantaneous_rate(sp,pq.ms,kernel=kernel)
            idx = np.isfinite(var_in[:,jj])
            r.append(scipy.corrcoef(var_in[:,jj].ravel()[idx],
                               FR.magnitude.ravel()[idx])[0,1])
        R.append(r)
    R = np.array(R)
    df = pd.DataFrame(data=R, columns=['{}ms'.format(x) for x in windows])
    df.index=component_list
    return(df)

def get_corr_with_FR():
    p_load =os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO')
    p_save = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')
    p_smooth = r'K:\VG3D\_rerun_with_pad\_deflection_trials\_NEO\smooth'

    DF = pd.DataFrame()
    for ii,f in enumerate(glob.glob(os.path.join(p_load,'*.h5'))):
        if ii==0:
            continue
        blk = neoUtils.get_blk(f)
        num_units = len(blk.channel_indexes[-1].units)
        for unit_num in range(num_units):
            id =  neoUtils.get_root(blk,unit_num)
            print('Working on {}'.format(id))
            try:
                df = calc_corr(f,p_smooth,unit_num)
                df['id'] = id
                DF = DF.append(df)
            except:
                print('Problem on {}'.format(id))
    DF.to_csv(os.path.join(p_save,'derivative_corr_by_smoothing.csv'),index=True)

if __name__ == '__main__':
    get_corr_with_FR()

