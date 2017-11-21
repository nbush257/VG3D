import os
import glob
from spikeAnalysis import *
from bayes_analyses import *
from neo_utils import *
p_save = r'C:\Users\nbush257\Box Sync\__VG3D\deflection_trials\figs'
p_load = r'C:\Users\nbush257\Box Sync\__VG3D\deflection_trials\data'
# ============================= #
# CONCATENATE PD CURVES         #
# ============================= #
THETA = []
L_dir = []
bin_edges = []
bin_heights = []
selectivity_signifigance = []
ID = []

fig_summary = plt.figure()
ax_summary = fig_summary.add_subplot(111,projection = 'polar')


for f in glob.glob(os.path.join(p_load,'rat*.pkl')):
    print(os.path.basename(f))

    blk = get_blk(f)
    M = get_var(blk)
    MB, MD = get_MB_MD(M)
    for unit in blk.channel_indexes[-1].units:
        sp = concatenate_sp(blk)[unit.name]
        root = get_root(blk,int(unit.name[-1]))
        ID.append(root)
        rate, theta_k, L_dir_, theta_preferred = PD_fitting(MD, sp)
        # p = direction_test(rate, theta_k)
        # selectivity_signifigance.append(p)
        # plot histograms

        THETA.append(theta_preferred)
        L_dir.append(L_dir_)
        bin_edges.append(theta_k)
        bin_heights.append(rate)

np.savez(os.path.join(p_save,'preferred_directions.npz'),
        THETA=THETA,
        L_dir=L_dir,
        bin_edges=bin_edges,
        bin_heights=bin_heights
        )


# ========================== #
# PLOT ALL #
# ========================== #
ID = []
for file in glob.glob(r'C:\Users\nbush257\Box Sync\__VG3D\deflection_trials\figs\PD*.png'):
    root = os.path.splitext(os.path.basename(file))[0][3:]
    ID.append(root)


p_save = r'C:\Users\nbush257\Box Sync\__VG3D\deflection_trials\figs'
THETA = dat['THETA']
L_dir = dat['L_dir']
rate = dat['bin_heights']
theta_k = dat['bin_edges']
for theta_preferred,L_dir_,th,r,root in zip(THETA,L_dir,theta_k,rate,ID):

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='polar')
    ax.plot(th[:-1], r*1000, 'ko', alpha=0.7)
    ax.annotate('',
                xy=(theta_preferred, L_dir_ * np.max(r*1000)),
                xytext=(0, 0),
                arrowprops={'arrowstyle': 'simple,head_width=1','linewidth':1,'color':[0.6,0.7,0.5]})
    plt.savefig(os.path.join(p_save,'PD_{}.svg'.format(root)),dpi=300)

    plt.close(fig)


# ============================= #
# ============================= #
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='polar')
    for ii in xrange(len(THETA)):

        ax.annotate('',
                    xy=(THETA[ii], L_dir[ii]),
                    xytext=(0, 0),
                    arrowprops={'arrowstyle': 'simple,head_width=1','linewidth': 1, 'facecolor':'k','alpha': 0.3}
                    )
plt.figure()
plt.hist(L_dir[np.isfinite(L_dir)],30,color='k',alpha=0.5)

plt.figure()
plt.hist(THETA[np.isfinite(THETA)])

np.savez(r'C:\Users\nbush257\Box Sync\___hartmann_lab\presentations\posters\sfn_2017\figs\direction_tuning.npz',
         ID =ID,
         THETA=THETA,
         L_dir=L_dir,
         theta_k=theta_k,
         rate=rate,
         )