import neoUtils
import numpy as np
import varTuning
import spikeAnalysis
import quantities as pq
import os
import matplotlib.pyplot as plt

# blk_smooth = neoUtils.get_blk(r"C:\Users\nbush\Documents\rat2017_08_FEB15_VG_D1_NEO_smooth_dat.h5")
# blk = neoUtils.get_blk()

blk_smooth = neoUtils.get_blk(r"F:\VG3D\_rerun_with_pad\_deflection_trials\_NEO\smooth\rat2017_09_FEB17_VG_C4_NEO_smooth_dat.h5")
blk = neoUtils.get_blk(os.path.join(r'D:\Users\NBUSH\Box Sync\Box Sync\__VG3D\_deflection_trials\_NEO','rat2017_09_FEB17_VG_C4_NEO.h5'))
def mult_join_plots(var):
    pass



def plot_smooth_hists(blk,blk_smooth,unit_num=0):
    use_flags = neoUtils.concatenate_epochs(blk)
    cbool = neoUtils.get_Cbool(blk)
    r,b =neoUtils.get_rate_b(blk,unit_num,2*pq.ms)

    M = neoUtils.get_var(blk_smooth,'M_smoothed').magnitude
    M[np.invert(cbool),:]=np.nan
    Mdot = neoUtils.get_deriv(M)

    F = neoUtils.get_var(blk_smooth,'F_smoothed').magnitude
    F[np.invert(cbool),:]=np.nan
    Fdot = neoUtils.get_deriv(F)

    PHI = neoUtils.get_var(blk_smooth,'PHIE_smoothed').magnitude
    PHI = neoUtils.center_var(PHI.squeeze(),use_flags)
    PHI[np.invert(cbool),:]=np.nan
    PHIdot = neoUtils.get_deriv(PHI)

    TH = neoUtils.get_var(blk_smooth,'TH_smoothed').magnitude
    TH = neoUtils.center_var(TH.squeeze(),use_flags)
    TH[np.invert(cbool),:]=np.nan
    THdot = neoUtils.get_deriv(TH)

    # ROT = np.sqrt(np.add(np.power(PHI,2),np.power(TH,2)))
    # ROTdot = neoUtils.get_deriv(ROT)

    colormax = np.nanmax(r)/2
    f = plt.figure()
    nbins=50
    for loc,ii in enumerate(range(0,10,2)):
        R,edges1,edges2 = varTuning.joint_response_hist(Mdot[:,1,ii],Mdot[:,2,ii],r,cbool,bins=nbins)
        ax = f.add_subplot(3,M.shape[-1]/2,loc+1)
        ax.pcolormesh(edges1[:-1],edges2[:-1], R, cmap='OrRd', edgecolors='None',vmin=0,vmax=colormax)
        ax.set_xlim(np.nanmin(Mdot[:,1,:]),np.nanmax(Mdot[:,1,:]))
        ax.set_ylim(np.nanmin(Mdot[:, 2,:]), np.nanmax(Mdot[:, 2, :]))
        ax.set_title('Smooth_level = {}'.format(ii))
        if ii==0:
            ax.set_ylabel('Mydot vs Mzdot')
    for loc,ii in enumerate(range(0,10,2)):
        R, edges1, edges2 = varTuning.joint_response_hist(Fdot[:, 1, ii], Fdot[:, 2, ii], r, cbool,bins=nbins)
        ax = f.add_subplot(3, M.shape[-1]/2, loc + 1+5)
        ax.pcolormesh(edges1[:-1], edges2[:-1], R, cmap='OrRd', edgecolors='None',vmin=0,vmax=colormax)
        ax.set_xlim(np.nanmin(Fdot[:, 1, :]), np.nanmax(Fdot[:, 1, :]))
        ax.set_ylim(np.nanmin(Fdot[:, 2, :]), np.nanmax(Fdot[:, 2, :]))
        if ii==0:
            ax.set_ylabel('Fydot vs Fzdot')
    for loc,ii in enumerate(range(0,10,2)):
        R, edges1, edges2 = varTuning.joint_response_hist(THdot[:, ii], PHIdot[:, ii], r, cbool,bins=nbins)
        ax = f.add_subplot(3, M.shape[-1]/2, loc + 1+10)
        ax.pcolormesh(edges1[:-1], edges2[:-1], R, cmap='OrRd', edgecolors='None',vmin=0,vmax=colormax)
        ax.set_xlim(np.nanmin(THdot), np.nanmax(THdot))
        ax.set_ylim(np.nanmin(PHIdot), np.nanmax(PHIdot))
        if ii==0:
            ax.set_ylabel('THdot vs PHIdot')

    plt.show()