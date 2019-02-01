import GLM
import pandas as pd
import neoUtils
import analyze_by_deflection
import os
import glob
import spikeAnalysis
import numpy as np
def get_first_spike_vals(fname,p_smooth,unit_num):
    """
    Return a dataframe with length Ncontacts and the value of
    relevant stimulus features at that time

    :param blk:         neo block
    :param unit_num:    int
    :return: pandas dataframe
    """
    # get the blocks
    blk = neoUtils.get_blk(fname)
    blk_smooth = GLM.get_blk_smooth(fname,p_smooth)
    # get the trains and times of first spikes
    _,_,trains = spikeAnalysis.get_contact_sliced_trains(blk,unit_num)
    t_idx = [train[0].magnitude if len(train)>0 else np.nan for train in trains]
    t_idx = np.array(t_idx)
    t_idx = t_idx[np.isfinite(t_idx)].astype('int')
    # get the stimuli
    varlist = ['M','F','TH','PHIE']
    X = GLM.create_design_matrix(blk,varlist)
    Xsmooth = GLM.get_deriv(blk,blk_smooth,varlist,smoothing=[9])[1]
    MB = np.sqrt(X[:,1]**2+X[:,2]**2)[:,np.newaxis]
    FB = np.sqrt(X[:,4]**2+X[:,5]**2)[:,np.newaxis]
    RB = np.sqrt(X[:,6]**2+X[:,7]**2)[:,np.newaxis]
    # use smooth to calculate derivative
    MBsmooth = np.sqrt(Xsmooth[:,1]**2+Xsmooth[:,2]**2)[:,np.newaxis]
    FBsmooth = np.sqrt(Xsmooth[:,4]**2+Xsmooth[:,5]**2)[:,np.newaxis]
    RBsmooth = np.sqrt(Xsmooth[:,6]**2+Xsmooth[:,7]**2)[:,np.newaxis]

    X = np.concatenate([MB,FB,RB],axis=1)
    Xsmooth = np.concatenate([MBsmooth,FBsmooth,RBsmooth],axis=1)
    Xdot = np.diff(np.concatenate([np.zeros([1,3]),Xsmooth]),axis=0)
    X = np.concatenate([X,Xdot],axis=1)

    #extract stimulus at time of first spike and output to a dataframe
    vals = X[t_idx]
    vallist = ['MB','FB','RB','MBdot','FBdot','RBdot']
    df = pd.DataFrame()
    for ii in range(len(vallist)):
        df[vallist[ii]] = vals[ii,:]
    df['id'] = neoUtils.get_root(blk,unit_num)
    return(df)
def batch_get_first_spike_val(p_load,p_save,p_smooth):
    DF = pd.DataFrame()
    for f in glob.glob(os.path.join(p_load,'*.h5')):
        blk = neoUtils.get_blk(f)
        num_units = len(blk.channel_indexes[-1].units)
        for unit_num in num_units:
            df = get_first_spike_vals(f,p_save,unit_num)
            DF = DF.append(df)
    DF.to_csv(os.path.join(p_save,'first_spike_data.csv'),index=False)

