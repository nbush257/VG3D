import glob
import os
import neoUtils
import numpy as np
import pandas as pd
import GLM
import scipy.io.matlab as sio
p_load = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO')
p_smooth =r'F:\VG3D\_rerun_with_pad\_deflection_trials\_NEO\smooth'
p_save = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\pillowX')
min_entropy = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results\min_smoothing_entropy.csv')
p_load_2d = r'F:\VG3D\_rerun_with_pad\_deflection_trials\_NEO_2D'
def smoothed_55ms():
    for f in glob.glob(os.path.join(p_load,'*.h5')):
        try:
            blk = neoUtils.get_blk(f)
            blk_smooth = GLM.get_blk_smooth(f,p_smooth)
            num_units = len(blk.channel_indexes[-1].units)
            for unit_num in range(num_units):
                varlist = ['M', 'F', 'TH', 'PHIE']
                root = neoUtils.get_root(blk,unit_num)
                print('Working on {}'.format(root))
                outname = os.path.join(p_save,'55ms_smoothing_deriv\\{}_pillowX.mat'.format(root))
                X = GLM.create_design_matrix(blk,varlist)
                Xdot = GLM.get_deriv(blk,blk_smooth,varlist,[5])[0]
                X = np.concatenate([X,Xdot],axis=1)
                sp = neoUtils.concatenate_sp(blk)['cell_{}'.format(unit_num)]
                y = neoUtils.get_rate_b(blk,unit_num)[1]
                cbool = neoUtils.get_Cbool(blk)

                sio.savemat(outname,{'X':X,'y':y,'cbool':cbool})
        except:
            print('Problem with {}'.format(os.path.basename(f)))

def smoothed_best():
    df = pd.read_csv(min_entropy,index_col='id')
    smooth_vals = np.arange(5,100,10).tolist()
    best_smooth = df.mode(axis=1)[0]
    best_idx = [smooth_vals.index(x) for x in best_smooth]
    best_idx = pd.DataFrame({'idx':best_idx},index=best_smooth.index)

    for f in glob.glob(os.path.join(p_load,'*.h5')):
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

                sio.savemat(outname,{'X':X,'y':y,'cbool':cbool,'smooth':best_smooth.loc[root]})
        except:
            print('Problem with {}'.format(os.path.basename(f)))

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

                sio.savemat(outname,{'X':X,'y':y,'cbool':cbool,'smooth':best_smooth.loc[root]})
        except:
            print('Problem with {}'.format(os.path.basename(f)))
