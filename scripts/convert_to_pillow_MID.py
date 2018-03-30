import glob
import os
import neoUtils
import numpy as np
import GLM
import scipy.io.matlab as sio
p_load = r'C:\Users\guru\Box Sync\__VG3D\_deflection_trials\_NEO'
p_smooth =r'K:\VG3D\_rerun_with_pad\_deflection_trials\_NEO\smooth'
p_save = r'C:\Users\guru\Box Sync\__VG3D\_deflection_trials\_NEO\results'
for f in glob.glob(os.path.join(p_load,'*.h5')):
    try:
        blk = neoUtils.get_blk(f)
        blk_smooth = GLM.get_blk_smooth(f,p_smooth)
        num_units = len(blk.channel_indexes[-1].units)
        for unit_num in range(num_units):
            varlist = ['M', 'F', 'TH', 'PHIE']
            root = neoUtils.get_root(blk,unit_num)
            print('Working on {}'.format(root))
            outname = os.path.join(p_save,'{}_pillowX.mat'.format(root))
            X = GLM.create_design_matrix(blk,varlist)
            Xdot = GLM.get_deriv(blk,blk_smooth,varlist,[5])[0]
            X = np.concatenate([X,Xdot],axis=1)
            sp = neoUtils.concatenate_sp(blk)['cell_{}'.format(unit_num)]
            y = neoUtils.get_rate_b(blk,unit_num)[1]
            cbool = neoUtils.get_Cbool(blk)

            sio.savemat(outname,{'X':X,'y':y,'cbool':cbool})
    except:
        print('Problem with {}'.format(os.path.basename(f)))
