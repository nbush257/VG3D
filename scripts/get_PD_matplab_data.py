import glob
import os
from scipy.io.matlab import savemat
from bayes_analyses import *
from neo_utils import *
from mechanics import *
p_out = r'C:\Users\guru\Desktop\test'
p_in = r'C:\Users\guru\Box Sync\__VG3D\deflection_trials\data'

def matlab_get_smooth_circ():
    for f in glob.glob(os.path.join(p_in,'rat*.pkl')):
        print(os.path.basename(f))
        fid = PIO(f)
        blk = fid.read_block()
        M = get_var(blk)
        MB, MD = get_MB_MD(M)
        for unit in blk.channel_indexes[-1].units:
            unit_num = int(unit.name[-1])
            r,b = get_rate_b(blk,unit_num)
            root = get_root(blk,unit_num)
            output_fname = 'circ_stats_{}.mat'.format(root)


            w,edges=PD_fitting(MD,r)
            save_dict = {
                'w':w,
                'alpha':edges[:-1],
                'root':root
            }
            savemat(os.path.join(p_out,output_fname),save_dict)


