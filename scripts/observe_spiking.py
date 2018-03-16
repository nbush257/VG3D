import sys
import neoUtils
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()
sns.set_style('ticks')

blk = neoUtils.get_blk(sys.argv[1])
M = neoUtils.get_var(blk).magnitude
sp = neoUtils.concatenate_sp(blk)
cc = neoUtils.concatenate_epochs(blk,-1)
Cbool = neoUtils.get_Cbool(blk)
c_idx = np.where(Cbool)[0]
# M[np.invert(Cbool),:] = 0

ymax = np.nanmax(M)/4
ymin = np.nanmin(M)/4


def shadeVector(cc,color='k'):
    ax = plt.gca()
    ylim = ax.get_ylim()
    for start,dur in zip(cc.times.magnitude,cc.durations.magnitude):
        ax.fill([start,start,start+dur, start+dur],[ylim[0],ylim[1],ylim[1],ylim[0]],color,alpha=0.1)


for ii in xrange(len(sp)):
    unit = sp['cell_{}'.format(ii)]
    # spt = unit.times.as_array().astype('int')
    # spt = np.intersect1d(spt, c_idx)

    plt.plot(M)
    shadeVector(cc)
    root = neoUtils.get_root(blk,ii)
    plt.vlines(unit.times,ymin,ymax,'k',alpha=0.4)
    plt.title('{}'.format(root))
    plt.show()
    plt.close('all')
