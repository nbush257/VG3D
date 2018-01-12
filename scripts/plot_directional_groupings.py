import neoUtils
import worldGeometry
import plotVG3D
import matplotlib.pyplot as plt
import glob
import os

# ========================================== #
#          plot directional PSTHs            #
# ========================================== #
for file in glob.glob(R'C:\Users\guru\Box Sync\__VG3D\_deflection_trials\_NEO\*.h5'):
    blk = neoUtils.get_blk(file)
    num_units = len(blk.channel_indexes[-1].units)
    for unit_num in xrange(num_units):
        root = neoUtils.get_root(blk,unit_num)
        outname = r'C:\Users\guru\Box Sync\__VG3D\_deflection_trials\_NEO\results\{}_directional_normed.png'.format(root)
        if os.path.isfile(outname):
            print('File {} already found, skipping'.format(os.path.basename(outname)))
            continue
        print('Working on {}'.format(root))
        out = plotVG3D.plot_spike_trains_by_direction(blk,unit_num,norm_dur=True)
        if out is -1:
            continue

        plt.savefig(outname,dpi=300)
        plt.close('all')

# ========================================== #
#          plot arclength groups             #
# ========================================== #
for file in glob.glob(R'C:\Users\guru\Box Sync\__VG3D\_deflection_trials\_NEO\*.h5'):
    blk = neoUtils.get_blk(file)
    num_units = len(blk.channel_indexes[-1].units)
    for unit_num in xrange(num_units):
        root = neoUtils.get_root(blk,unit_num)
        outname = r'C:\Users\guru\Box Sync\__VG3D\_deflection_trials\_NEO\results\{}_arclength_groupings.png'.format(root)
        if os.path.isfile(outname):
            print('File {} already found, skipping'.format(os.path.basename(outname)))
            continue
        print('Working on {}'.format(root))
        try:
            idx = worldGeometry.get_radial_distance_group(blk,plot_tgl=True)
        except:
            print('Warning,{} Failed'.format(root))
        if idx is -1:
            continue

        plt.savefig(outname,dpi=300)
        plt.close('all')