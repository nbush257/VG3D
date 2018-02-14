import glob
import os
import sys
import neoUtils
import spikeAnalysis
import pandas as pd
import worldGeometry
import numpy as np
def create_threshold_DF(blk,unit_num=0,time_win=20,max_spikes=3):
    # TODO: I don't know if this is super meaningful.
    # If we use a constant time window then we are not looking at the magnitude, but th derivative
    # which could be what we want...
    use_flags = neoUtils.concatenate_epochs(blk)

    id = neoUtils.get_root(blk,unit_num)
    if len(use_flags)<10:
        print('{} has too few contacts'.format(id))
        return -1

    onset,offset = neoUtils.get_contact_apex_idx(blk,mode='time_win',time_win=time_win)
    all_var_mag = np.empty([len(onset),0])
    for varname in ['M','F','TH','PHIE']:
        var = neoUtils.get_var(blk,varname)
        if varname in ['TH','PHIE']:
            var = neoUtils.center_var(var,use_flags)
        var_sliced = neoUtils.get_analog_contact_slices(var,use_flags)

        var_onset = worldGeometry.get_onset(var_sliced,onset,to_array=False)

        var_mag = np.array([x[-1] if len(x)>0 else np.zeros(var_sliced.shape[2]) for x in var_onset ])
        all_var_mag = np.concatenate([all_var_mag,var_mag],axis=1)

    c_idx = np.empty(var_sliced.shape[1],dtype='f8')
    c_idx[:] = np.nan
    for n_spikes in range(max_spikes):
        temp_idx = spikeAnalysis.get_onset_contacts(blk,onset,num_spikes=n_spikes)
        c_idx[temp_idx]=n_spikes
    X =np.concatenate([all_var_mag,c_idx[:,np.newaxis]],axis=1)
    df = pd.DataFrame(X)
    df = df.rename(columns={0:'Mx',1:'My',2:'Mz',3:'Fx',4:'Fy',5:'Fz',6:'TH',7:'PHI',8:'n_spikes'})

    dir_idx,med_dir = worldGeometry.get_contact_direction(blk,False)

    df['dir_idx'] = dir_idx
    df['med_dir'] = df.dir_idx.map({x:med_dir[x] for x in range(len(med_dir))})
    df['id'] = [id for x in range(df.shape[0])]
    df['time_win'] = [time_win for x in range(df.shape[0])]
    return(df)

def batch_thresh_response(p_load,p_save):
    DF = pd.DataFrame()
    for f in glob.glob(os.path.join(p_load,'*.h5')):
       blk = neoUtils.get_blk(f)
       print('Working on {}'.format(os.path.basename(f)))
       num_units = len(blk.channel_indexes[-1].units)
       for unit_num in range(num_units):
           df = create_threshold_DF(blk,unit_num)
           if df is not -1:
               DF = DF.append(df)
    DF.to_csv(os.path.join(p_save,'threshold_variable_response.csv'),index=False)
    return(0)
if __name__ == '__main__':
    # blk = neoUtils.get_blk(r'C:\Users\guru\Box Sync\__VG3D\_deflection_trials\_NEO\rat2016_12_MAR02_VG_B3_NEO.h5')
    # create_threshold_DF(blk,0)
    p_load = r'C:\Users\guru\Box Sync\__VG3D\_deflection_trials\_NEO'
    p_save = r'C:\Users\guru\Box Sync\__VG3D\_deflection_trials\_NEO\results'
    batch_thresh_response(p_load,p_save)
