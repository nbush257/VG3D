""" Plots segments of spiking and stimulus.
Needs to be made so that the labels are in and maybe make nice colors.
Maybe plot PH and TH on same axis"""
import sys
import neo
import quantities as pq
import neoUtils
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotVG3D
import os
sns.set()
sns.set_style('ticks')
dpi_res,figsize,ext=plotVG3D.set_fig_style()
p_load = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO')
fname = os.path.join(p_load,r'rat2017_03_JAN10_VG_B1_NEO.h5')
p_save = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')
starts= [  224600 ,298800]
stops = [227000,300750]
starts = np.array(starts)*pq.ms
stops = np.array(stops)*pq.ms
blk = neoUtils.get_blk(fname)
sp= neoUtils.concatenate_sp(blk)['cell_0']
cc = neoUtils.concatenate_epochs(blk)
cbool = neoUtils.get_Cbool(blk)
def shadeVector(cc,color='k',ax=None):
    if ax is None:
        ax = plt.gca()

    ylim = ax.get_ylim()
    for start,dur in zip(cc.times.magnitude,cc.durations.magnitude):
        ax.fill([start,start,start+dur, start+dur],[ylim[0],ylim[1],ylim[1],ylim[0]],color,alpha=0.1)

wd = figsize[0]
ht = wd/2
M = neoUtils.get_var(blk,'M')
F = neoUtils.get_var(blk,'F')
TH = neoUtils.get_var(blk,'TH')
PH = neoUtils.get_var(blk,'PHIE')
TH = neoUtils.center_var(TH,cc)
PH = neoUtils.center_var(PH,cc)
TH[np.invert(cbool)] = np.nan
PH[np.invert(cbool)] = np.nan
for start,stop in zip(starts,stops):
    fig,ax = plt.subplots(4,1,figsize=(wd,ht))
    sp_slice = sp.time_slice(start,stop)
    ax[0].plot(M.time_slice(start,stop))
    ax[1].plot(F.time_slice(start,stop))
    ax[2].plot(TH.time_slice(start,stop))
    ax[3].plot(PH.time_slice(start,stop))
    for _ax in ax:
        sns.despine()
        _cc = neoUtils.cbool_to_cc(cbool[int(start):int(stop)])
        _cc = neo.core.Epoch(times = _cc[0]*pq.ms,durations = (_cc[1]-_cc[0])*pq.ms,t_start=0*pq.ms,t_stop=stop-start,units=pq.ms)
        shadeVector(_cc,ax=_ax)
        ylim = _ax.get_ylim()
        for spike in sp_slice:
            _ax.vlines(spike-sp_slice.t_start,ylim[0],ylim[1]/2,color='k',alpha=0.4)

plt.show()