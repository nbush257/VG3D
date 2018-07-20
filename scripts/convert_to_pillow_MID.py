import glob
import os
import neoUtils
import numpy as np
import pandas as pd
import GLM
import scipy.io.matlab as sio
if 'BOX_PATH' in os.environ:
    p_load = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO')
    p_smooth =r'K:\VG3D\_rerun_with_pad\_deflection_trials\_NEO\smooth'
    p_save = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\pillowX')
    min_entropy = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results\min_smoothing_entropy.csv')
    p_load_2d = r'K:\VG3D\_rerun_with_pad\_deflection_trials\_NEO_2D'
else:
    p_load ='/projects/p30144/_VG3D/deflections/_NEO'
    p_smooth = p_load
    p_save =os.path.join(p_load,'pillow')
    p_load_2d = '/projects/p30144/_VG3D/deflections/_NEO_2D'


def get_arclength_bool(blk,unit_num,fname=None):
    # fname is the name of the csv file with arclength groupings
    if fname is None:
        if 'BOX_PATH' in os.environ:
            fname = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results\direction_arclength_FR_group_data.csv')
        else:
            fname =os.path.join('/projects/p30144/_VG3D/deflections/direction_arclength_FR_group_data.csv')
    df = pd.read_csv(fname)
    id = neoUtils.get_root(blk,unit_num)
    sub_df = df[df.id==id]
    arclength_list = sub_df.Arclength.tolist()
    use_flags = neoUtils.concatenate_epochs(blk)
    if len(sub_df)!= len(use_flags):
        raise ValueError('The number of contacts in the block {} do not match the number of contacts in the csv {}'.format(len(use_flags),len(sub_df)))
    cbool = neoUtils.get_Cbool(blk)
    distal_cbool = np.zeros_like(cbool)
    medial_cbool = np.zeros_like(cbool)
    proximal_cbool = np.zeros_like(cbool)
    # loop through each contact and set the appropriate arclength boolean
    for ii in range(len(use_flags)):
        start = use_flags[ii].magnitude.astype('int')
        dur = use_flags.durations[ii].magnitude.astype('int')
        if arclength_list[ii] == 'Proximal':
            proximal_cbool[start:start+dur]=1
        elif arclength_list[ii] == 'Distal':
            distal_cbool[start:start+dur]=1
        elif arclength_list[ii] == 'Medial':
            medial_cbool[start:start+dur]=1
    arclengths = {'Distal':distal_cbool,
                  'Medial':medial_cbool,
                  'Proximal':proximal_cbool}

    return(arclengths)


def smoothed(smooth_idx=9):
    smooth_vals = np.arange(5,100,10)
    sub_p_save = os.path.join(p_save,'{}ms_smoothing_deriv'.format(smooth_vals[smooth_idx]))
    if not os.path.isdir(sub_p_save):
        os.mkdir(sub_p_save)
    for f in glob.glob(os.path.join(p_load,'*NEO.h5')):
        try:
            blk = neoUtils.get_blk(f)
            blk_smooth = GLM.get_blk_smooth(f,p_smooth)
            num_units = len(blk.channel_indexes[-1].units)
            for unit_num in range(num_units):
                varlist = ['M', 'F', 'TH', 'PHIE']
                root = neoUtils.get_root(blk,unit_num)
                print('Working on {}'.format(root))
                outname =os.path.join(sub_p_save,'{}ms_{}_pillowX.mat'.format(smooth_vals[smooth_idx],root))

                X = GLM.create_design_matrix(blk,varlist)
                Xdot = GLM.get_deriv(blk,blk_smooth,varlist,[smooth_idx])[0]
                X = np.concatenate([X,Xdot],axis=1)
                sp = neoUtils.concatenate_sp(blk)['cell_{}'.format(unit_num)]
                y = neoUtils.get_rate_b(blk,unit_num)[1]
                cbool = neoUtils.get_Cbool(blk)
                arclengths = get_arclength_bool(blk,unit_num)

                sio.savemat(outname,{'X':X,'y':y,'cbool':cbool,'arclengths':arclengths})
        except Exception as ex:
            print('Problem with {}:{}'.format(os.path.basename(f),ex))


def smoothed_mechanics():
    """
    use this function to grab the data from the smoothed mechanics and the
    derivative of the same
    """

    f_arclength = '/projects/p30144/_VG3D/deflections/direction_arclength_FR_group_data.csv'
    f_list =glob.glob(os.path.join(p_load,'*NEO.h5'))
    f_list.sort()

    for f in f_list:
        try:
            blk = neoUtils.get_blk(f)
            blk_smooth = GLM.get_blk_smooth(f,p_smooth)
            num_units = len(blk.channel_indexes[-1].units)
            for unit_num in range(num_units):
                varlist = ['M', 'F', 'TH', 'PHIE']
                root = neoUtils.get_root(blk,unit_num)
                print('Working on {}'.format(root))
                outname = os.path.join(p_save,'{}_smooth_mechanicsX.mat'.format(root))

                Xdot,X = GLM.get_deriv(blk,blk_smooth,varlist,smoothing=[5])
                X = np.concatenate([X,Xdot],axis=1)
                y = neoUtils.get_rate_b(blk,unit_num)[1]
                cbool = neoUtils.get_Cbool(blk)
                arclengths = get_arclength_bool(blk,unit_num,fname=f_arclength)

                sio.savemat(outname,{'X':X,'y':y,'cbool':cbool,'smooth':55,'arclengths':arclengths})
        except Exception as ex:
            print('Problem with {}:{}'.format(os.path.basename(f),ex))

def smoothed_best():
    df = pd.read_csv(min_entropy,index_col='id')
    smooth_vals = np.arange(5,100,10).tolist()
    best_smooth = df.mode(axis=1)[0]
    best_idx = [smooth_vals.index(x) for x in best_smooth]
    best_idx = pd.DataFrame({'idx':best_idx},index=best_smooth.index)

    for f in glob.glob(os.path.join(p_load,'*NEO.h5')):
        try:
            blk = neoUtils.get_blk(f)
            blk_smooth = GLM.get_blk_smooth(f,p_smooth)
            num_units = len(blk.channel_indexes[-1].units)
            for unit_num in range(num_units):
                varlist = ['M', 'F', 'TH', 'PHIE']
                root = neoUtils.get_root(blk,unit_num)
                print('Working on {}'.format(root))
                if root not in best_idx.index:
                    print('{} not found in best smoothing derivative data'.format(root))
                    continue
                outname = os.path.join(p_save,'best_smoothing_deriv\\{}_best_smooth_pillowX.mat'.format(root))
                X = GLM.create_design_matrix(blk,varlist)
                smoothing_to_use = best_idx.loc[root][0]

                Xdot = GLM.get_deriv(blk,blk_smooth,varlist,smoothing=[smoothing_to_use])[0]
                X = np.concatenate([X,Xdot],axis=1)
                y = neoUtils.get_rate_b(blk,unit_num)[1]
                cbool = neoUtils.get_Cbool(blk)
                arclengths = get_arclength_bool(blk,unit_num)

                sio.savemat(outname,{'X':X,'y':y,'cbool':cbool,'smooth':best_smooth.loc[root],'arclengths':arclengths})
        except Exception as ex:
            print('Problem with {}:{}'.format(os.path.basename(f),ex))


def get_2D():
    df = pd.read_csv(min_entropy,index_col='id')
    smooth_vals = np.arange(5,100,10).tolist()
    best_smooth = df.mode(axis=1)[0]
    best_idx = [smooth_vals.index(x) for x in best_smooth]
    best_idx = pd.DataFrame({'idx':best_idx},index=best_smooth.index)

    for f in glob.glob(os.path.join(p_load_2d,'*smoothed.h5')):
        try:
            blk = neoUtils.get_blk(f)
            num_units = len(blk.channel_indexes[-1].units)
            for unit_num in range(num_units):
                varlist = ['M', 'FX', 'FY','TH']
                root = neoUtils.get_root(blk,unit_num)
                print('Working on {}'.format(root))
                if root not in best_idx.index:
                    print('{} not found in best smoothing derivative data'.format(root))
                    continue
                outname = os.path.join(p_save,'2d_best_smoothing\\{}_best_smooth_2d_pillowX.mat'.format(root))
                X = GLM.create_design_matrix(blk,varlist)
                smoothing_to_use = best_idx.loc[root][0]

                Xdot = GLM.get_deriv(blk,blk,varlist,smoothing=[smoothing_to_use])[0]
                X = np.concatenate([X,Xdot],axis=1)
                y = neoUtils.get_rate_b(blk,unit_num)[1]
                cbool = neoUtils.get_Cbool(blk)
                arclengths = get_arclength_bool(blk,unit_num)

                sio.savemat(outname,{'X':X,'y':y,'cbool':cbool,'smooth':best_smooth.loc[root],'arclengths':arclengths})
        except:
            print('Problem with {}'.format(os.path.basename(f)))

