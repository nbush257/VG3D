import glob
import os
from scipy.io.matlab import savemat
from bayes_analyses import *
from neo_utils import *
from mechanics import *
p_out = r'C:\Users\guru\Desktop\test'
p_in = r'C:\Users\guru\Box Sync\__VG3D\deflection_trials\data'

for f in glob.glob(os.path.join(p_in,'rat*.pkl')):
    print(os.path.basename(f))
    fid = PIO(f)
    blk = fid.read_block()
    for unit in blk.channel_indexes[-1].units:
        sp = concatenate_sp(blk)[unit.name]
        root = get_root(blk,int(unit.name[-1]))
        output_fname = 'circ_stats_{}.mat'.format(root)
        M = get_var(blk)
        MB,MD = get_MB_MD(M)

        w,edges=PD_fitting(MD,sp)
        save_dict = {
            'w':w,
            'alpha':edges[:-1],
            'root':root
        }
        savemat(os.path.join(p_out,output_fname),save_dict)
