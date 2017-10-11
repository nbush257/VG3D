from spikeAnalysis import *
from bayes_analyses import *
from neo_utils import *
from mechanics import *
from neo.io import PickleIO as PIO
import os
import glob

p = r'C:\Users\guru\Box Sync\__VG3D\deflection_trials\data'
p_save = r'C:\Users\guru\Box Sync\__VG3D\deflection_trials\figs'
files = glob.glob(os.path.join(p,'*.pkl'))
for f in files:
    print os.path.basename(f)
    fid = PIO(f)
    blk = fid.read_block()
    for unit in blk.channel_indexes[-1].units:
        cell_num = int(unit.name[-1])
        root = get_root(blk,cell_num)
        M = get_var(blk)[0]
        replace_NaNs(M)
        sp = concatenate_sp(blk)
        st = sp[unit.name]
        kernel = elephant.kernels.GaussianKernel(5 * pq.ms)
        r = np.array(instantaneous_rate(st,sampling_period=pq.ms,kernel =kernel)).ravel()
        Mdot = get_deriv(M)


        fig = plt.figure()
        axy = fig.add_subplot(121)
        axz = fig.add_subplot(122)

        bayes_plots(M[:, 1], Mdot[:, 1], r, 20, ax=axy)
        bayes_plots(M[:, 2], Mdot[:, 2], r, 20, ax=axz)

        axy.set_title('M$_y$ Phase Plane')
        axz.set_title('M$_z$ Phase Plane')

        axy.set_xlabel('M$_y$ (N-m)')
        axy.set_ylabel('M$_\dot{y}$ (N-m/ms)')

        axz.set_xlabel('M$_z$ (N-m)')
        axz.set_ylabel('M$_\dot{z}$ (N-m/ms)')

        axy.grid('off')
        axy.set_facecolor([0.3, 0.3, 0.3])
        axz.grid('off')
        axz.set_facecolor([0.3, 0.3, 0.3])
        axy.axis('normal')
        axz.axis('normal')
        fig.suptitle(root)
        plt.tight_layout()
        plt.savefig(os.path.join(p_save, root + '_phase_planes.png'), dpi=300)