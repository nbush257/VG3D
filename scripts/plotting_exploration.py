from spikeAnalysis import *
from bayes_analyses import *
from neo_utils import *
from mechanics import *
from neo.io import PickleIO as PIO
import os
import glob
from sklearn.preprocessing import imputation
from matplotlib.ticker import FormatStrFormatter
import subprocess
import os
from mpl_toolkits.mplot3d import Axes3D


p = r'C:\Users\guru\Box Sync\__VG3D\deflection_trials\data'
files = glob.glob(os.path.join(p,'*.pkl'))
f = r'rat2017_01_JAN05_VG_B2_NEO.pkl'
cell_num = 0

fid = PIO(os.path.join(p,f))
blk  = fid.read_block()
root = get_root(blk,cell_num)
M = get_var(blk)[0]
F = get_var(blk,'F')[0]
sp = concatenate_sp(blk)
Cbool = get_Cbool(blk)
M[np.invert(Cbool),:]=0
F[np.invert(Cbool),:]=0

M = replace_NaNs(M,mode='pchip')
M = replace_NaNs(M,mode='interp')

F = replace_NaNs(F,mode='pchip')
F = replace_NaNs(F,mode='interp')

b = binarize(sp['cell_{}'.format(cell_num)],sampling_rate=pq.kHz)[:-1]
st = np.array(sp['cell_{}'.format(cell_num)]).astype('f8')
# ============================================== #
# ====== Plot Mech and spikes over time ======== #
# ============================================== #
fig = plt.figure()
ax = fig.add_subplot(111)

# plt.plot(b*M.ptp()/10,'k')
ax.vlines(sp['cell_0'],np.min(M),np.max(M),alpha=0.4)
ax.plot(M,linewidth=2)
ax.legend(['M$_x$','M$_y$','M$_z$','spikes'],bbox_to_anchor=(1, 0.5),loc='center left',fontsize=14)
ax.set_xlabel('Time (ms)',fontsize=20)
ax.set_ylabel('Moments (N-m)',fontsize=20)
ax.set_title('{}'.format(get_root(blk,0)),fontsize=20)
ax.tick_params(labelsize=16)

# position one
# ax.set_ylim([-1.06e-8,1.54e-7])
# ax.set_xlim([4.29211e5,4.32e5])


# =====================================================================#
# ======= Make a rotating plot of the mechanics input space ===========#
# =====================================================================#

p_vid_save = r'C:\Users\guru\Desktop\test'
f_vid_save = root
fig = plt.figure(figsize=(8,5))
ax3M = fig.add_subplot(121,projection='3d')
ax3M.plot(M[:,0],M[:,1],M[:,2],'k.',alpha=0.01)
ax3M.set_title('Moments')
ax3M.set_xlabel('M$_x$ (N-m)',labelpad=15)
ax3M.set_ylabel('M$_y$ (N-m)',labelpad=15)
ax3M.set_zlabel('M$_z$ (N-m)',labelpad=15)
ax3M.patch.set_facecolor('w')
ax3M.axis('equal')

ax3F = fig.add_subplot(122,projection='3d')
ax3F.plot(F[:,0],F[:,1],F[:,2],'k.',alpha=0.01)
ax3F.set_title('Forces')
ax3F.set_xlabel('F$_x$ (N)',labelpad=15)
ax3F.set_ylabel('F$_y$ (N)',labelpad=15)
ax3F.set_zlabel('F$_z$ (N)',labelpad=15)
ax3F.patch.set_facecolor('w')

ax3M.w_xaxis.set_pane_color((0., 0., 0., 0.3))
ax3M.w_yaxis.set_pane_color((0., 0., 0., 0.3))
ax3M.w_zaxis.set_pane_color((0., 0., 0., 0.3))

ax3F.w_xaxis.set_pane_color((0., 0., 0., 0.3))
ax3F.w_yaxis.set_pane_color((0., 0., 0., 0.3))
ax3F.w_zaxis.set_pane_color((0., 0., 0., 0.3))



ax3M.xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
ax3M.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))
ax3M.zaxis.set_major_formatter(FormatStrFormatter('%.0e'))

ax3F.xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
ax3F.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))
ax3F.zaxis.set_major_formatter(FormatStrFormatter('%.0e'))

ax3F.axis('equal')
ax3M.axis('equal')
count=0
plt.tight_layout()
for ii in xrange(0,360,1):
    ax3M.azim=ii
    ax3F.azim = ii

    fig.set_size_inches(11, 6)
    plt.draw()
    plt.savefig(os.path.join(p_vid_save,f_vid_save+'{:05d}.png'.format(count)),dpi=300)

    count+=1

subprocess.call(['ffmpeg',
                 '-f','image2',
                 '-r','10',
                 '-i',os.path.join(p_vid_save,f_vid_save+'%05d.png'),
                 '-c:v','mpeg4',
                 '-q','2',
                 '-y',
                 os.path.join(p_vid_save,f_vid_save+'.mp4')
                 ])
for item in os.listdir(p_vid_save):
    if item.endswith('.png') or item.endswith('.wav'):
        os.remove(os.path.join(p_vid_save,item))

