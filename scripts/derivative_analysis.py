import neoUtils
import numpy as np
import varTuning
import quantities as pq
import os
import matplotlib.pyplot as plt
import glob
import sys

def mult_join_plots(var1,var2,r,cbool,bins=50):
    if type(bins)==int:
        bins = [np.linspace(np.nanmin(var1), np.nanmax(var1), bins),
                np.linspace(np.nanmin(var2), np.nanmax(var2), bins)]

    R = {}
    xx=[]
    yy=[]


    for ii in range(var1.shape[-1]):
        R[ii] = varTuning.joint_response_hist(var1[:,ii],var2[:,ii],r,cbool,bins=bins)[0]
    for mesh in R.values():
        rows = np.where(np.any(~mesh.mask, axis=0))[0]
        cols = np.where(np.any(~mesh.mask, axis=1))[0]

        xx.append([rows[0],rows[-1]])
        yy.append([cols[0], cols[-1]])


    xx = [np.min(xx), np.max(xx)]
    yy = [np.min(yy), np.max(yy)]



    return(R,bins,xx,yy)

def plot_smooth_hists(blk,blk_smooth,unit_num=0,p_save=None,nbins=75):
    DPI_RES=600
    id = neoUtils.get_root(blk, unit_num)
    fig_name = os.path.join(p_save, '{}_derivative_smoothing_compare.png'.format(id))
    if os.path.isfile(fig_name):
        print('{} found, skipping...'.format(fig_name))
        return(None)

    smoothing_windows = range(5,101,10)
    use_flags = neoUtils.concatenate_epochs(blk)
    cbool = neoUtils.get_Cbool(blk)
    r,b =neoUtils.get_rate_b(blk,unit_num,2*pq.ms)

    # catch empty smoothed data
    if len(blk_smooth.segments)==0 or len(blk_smooth.segments[0].analogsignals)==0:
        print('Smoothed data not found in {}'.format(id))
        return(-1)

    # get vars
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


    # calculate histograms
    R_Mdot, bins_Mdot, edgesx_Mdot, edgesy_Mdot = mult_join_plots(Mdot[:, 1, :], Mdot[:, 2, :], r, cbool, bins=nbins)
    newbins =[np.linspace(bins_Mdot[0][edgesx_Mdot][0],bins_Mdot[0][edgesx_Mdot][1],nbins),
              np.linspace(bins_Mdot[1][edgesy_Mdot][0], bins_Mdot[1][edgesy_Mdot][1], nbins)]
    R_Mdot, bins_Mdot, edgesx_Mdot, edgesy_Mdot = mult_join_plots(Mdot[:, 1, :], Mdot[:, 2, :], r, cbool, bins=newbins)

    R_Fdot, bins_Fdot, edgesx_Fdot, edgesy_Fdot = mult_join_plots(Fdot[:, 1, :], Fdot[:, 2, :], r, cbool,bins=nbins)
    newbins = [np.linspace(bins_Fdot[0][edgesx_Fdot][0], bins_Fdot[0][edgesx_Fdot][1], nbins),
               np.linspace(bins_Fdot[1][edgesy_Fdot][0], bins_Fdot[1][edgesy_Fdot][1], nbins)]
    R_Fdot, bins_Fdot, edgesx_Fdot, edgesy_Fdot = mult_join_plots(Fdot[:, 1, :], Fdot[:, 2, :], r, cbool, bins=newbins)

    R_ROTdot, bins_ROTdot, edgesx_ROTdot, edgesy_ROTdot = mult_join_plots(THdot, PHIdot, r, cbool,bins=nbins)
    newbins = [np.linspace(bins_ROTdot[0][edgesx_ROTdot][0], bins_ROTdot[0][edgesx_ROTdot][1], nbins),
               np.linspace(bins_ROTdot[1][edgesy_ROTdot][0], bins_ROTdot[1][edgesy_ROTdot][1], nbins)]
    R_ROTdot, bins_ROTdot, edgesx_ROTdot, edgesy_ROTdot = mult_join_plots(THdot, PHIdot, r, cbool, bins=newbins)

    FR = []
    FR.append(np.nanmax([x.max() for x in R_Mdot.values()]))
    FR.append(np.nanmax([x.max() for x in R_Fdot.values()]))
    FR.append(np.nanmax([x.max() for x in R_ROTdot.values()]))
    colormax = np.nanmax(FR)

    # Plots
    f = plt.figure()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    # hardcoded for 5 smoothing steps
    for loc,ii in enumerate(range(0,10,2)):
        ax = f.add_subplot(3,5,loc+1)
        ax.pcolormesh(bins_Mdot[0],bins_Mdot[1],R_Mdot[ii], cmap='OrRd', edgecolors='None',vmin=0,vmax=colormax)
        ax.set_xlim(bins_Mdot[0][edgesx_Mdot])
        ax.set_ylim(bins_Mdot[1][edgesy_Mdot])
        ax.set_title('Smoothing window = {}ms'.format(smoothing_windows[ii]))
        ax.axvline(color='k',linewidth=1)
        ax.axhline(color='k',linewidth=1)
        if ii==0:
            ax.set_ylabel('$\\dot{M_y}$ vs  $\\dot{M_z}$',rotation=0,labelpad=20)
    for loc,ii in enumerate(range(0,10,2)):
        ax = f.add_subplot(3, 5, loc + 1+5)
        ax.pcolormesh(bins_Fdot[0], bins_Fdot[1], R_Fdot[ii], cmap='OrRd', edgecolors='None', vmin=0, vmax=colormax)
        ax.set_xlim(bins_Fdot[0][edgesx_Fdot])
        ax.set_ylim(bins_Fdot[1][edgesy_Fdot])
        ax.axvline(color='k', linewidth=1)
        ax.axhline(color='k', linewidth=1)
        if ii==0:
            ax.set_ylabel('$\\dot{F_y}$ vs $\\dot{F_z}$',rotation=0,labelpad=20)
    for loc,ii in enumerate(range(0,10,2)):
        ax = f.add_subplot(3, 5, loc + 1+10)
        h=ax.pcolormesh(bins_ROTdot[0], bins_ROTdot[1], R_ROTdot[ii], cmap='OrRd', edgecolors='None', vmin=0, vmax=colormax)
        ax.set_xlim(bins_ROTdot[0][edgesx_ROTdot])
        ax.set_ylim(bins_ROTdot[1][edgesy_ROTdot])
        ax.axvline(color='k', linewidth=1)
        ax.axhline(color='k', linewidth=1)
        if ii==0:
            ax.set_ylabel('$\\dot{\\theta}$ vs $\\dot{\\phi}$',rotation=0,labelpad=20)
    plt.suptitle('{}'.format(id))
    plt.colorbar(h)
    plt.pause(0.1)

    if p_save is not None:
        plt.savefig(fig_name,dpi=DPI_RES)
        plt.close('all')
    return(None)

def find_smooth_match(fname,p_smooth):
    f_smooth = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(os.path.join(p_smooth,'*smooth_dat.h5'))]
    root = os.path.splitext(os.path.basename(fname))[0]
    return([s for s in f_smooth if root in s])



def main(p_raw,p_smooth,p_save):
    print('p_raw: {}\np_smooth:{}\np_save:{}'.format(p_raw,p_smooth,p_save))
    for f in glob.glob(os.path.join(p_raw,'*.h5')):
        f_smooth = find_smooth_match(f,p_smooth)
        if len(f_smooth)==0:
            print('{} smooth correspondant not found'.format(f))
            continue
        else:
            f_smooth =f_smooth[0]+'.h5'
            print('Working on {}'.format(os.path.basename(f)))
        blk = neoUtils.get_blk(os.path.join(p_raw, f))
        blk_smooth = neoUtils.get_blk(os.path.join(p_smooth, f_smooth))
        num_units = len(blk.channel_indexes[-1].units)
        for unit_num in xrange(num_units):
            try:
                res= plot_smooth_hists(blk,blk_smooth,unit_num,p_save)
            except:
                print('Failed at {}'.format(f))


if __name__ == '__main__':

    p_raw = sys.argv[1]
    p_smooth = sys.argv[2]
    p_save = sys.argv[3]
    main(p_raw,p_smooth,p_save)

