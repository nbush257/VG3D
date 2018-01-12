import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import neoUtils
import spikeAnalysis
import worldGeometry
import numpy as np
import matplotlib.gridspec as gridspec
import quantities as pq
sns.set()

def plot3():
    f = plt.figure()
    return(Axes3D(f))
def polar():
    f = plt.figure()
    return(f.add_subplot(111,projection='polar'))

def plot_spike_trains_by_direction(blk,unit_num=0,norm_dur=False,binsize=5*pq.ms):
    # TODO: Normalize time dimension
    # if True:
    #     raise Exception('Need to normalize by time')
    unit = blk.channel_indexes[-1].units[unit_num]
    _,_,trains = spikeAnalysis.get_contact_sliced_trains(blk,unit)

    b,durations = spikeAnalysis.get_binary_trains(trains)

    idx,med_angle = worldGeometry.get_contact_direction(blk,plot_tgl=False)
    if idx is -1:
        return(-1)

    th_contacts,ph_contacts = worldGeometry.get_delta_angle(blk)

    cc = sns.color_palette("husl", 8)
    f = plt.figure(tight_layout=True)
    f.set_dpi(300)
    f.set_size_inches(9,9)
    gs = gridspec.GridSpec(3, 3)
    ax = plt.subplot(gs[1:-1,1:-1])
    to_rotate = -med_angle[0]
    R = np.array([[np.cos(to_rotate),-np.sin(to_rotate)],
                  [np.sin(to_rotate),np.cos(to_rotate)]])

    # plot deflections
    for ii in xrange(len(idx)):
        th = np.deg2rad(th_contacts[:, ii])
        ph = np.deg2rad(ph_contacts[:, ii])
        X = np.vstack((th,ph))
        X_rot = np.dot(R,X)
        plt.plot(np.rad2deg(X_rot[0,:]), np.rad2deg(X_rot[1,:]), '.-', color=cc[idx[ii]], alpha=0.3)
    ax.set_xlabel(r'$\theta$ (deg)')
    ax.set_ylabel(r'$\phi$ (deg)')

    axis_coords = [[1, 2],
                   [0, 2],
                   [0, 1],
                   [0, 0],
                   [1, 0],
                   [2, 0],
                   [2, 1],
                   [2,2]]

    time_lim = np.percentile(durations,90) # drop the tenth percentile of durations

    PSTH=[]
    t_edges = []
    max_fr = []
    for dir in np.arange(np.max(idx)+1):
        sub_idx = np.where(idx == dir)[0]
        sub_trains = [trains[ii] for ii in sub_idx]
        if norm_dur:
            t_edges_temp,PSTH_temp,w = spikeAnalysis.get_time_stretched_PSTH(sub_trains)
        else:
            spt = spikeAnalysis.trains2times(sub_trains,concat_tgl=True)
            PSTH_temp, t_edges_temp = np.histogram(spt, bins=np.arange(0, 500, float(binsize)))
            PSTH_temp = PSTH_temp.astype('f8') / len(durations) / pq.ms * 1000.
            w = binsize

        max_fr.append(np.max(PSTH_temp))
        PSTH.append(PSTH_temp)
        t_edges.append(t_edges_temp)
    max_fr = np.max(max_fr)

    for dir in np.arange(np.max(idx)+1):
        ax = plt.subplot(gs[axis_coords[dir][0],axis_coords[dir][1]])
        ax.set_ylim([0,1])

        if norm_dur:
            ax.set_xlim(0, 1)
        else:
            ax.set_xlim(0,time_lim)

        plt.bar(t_edges[dir][:-1],
                PSTH[dir]/max_fr,
                width=w,
                align='edge',
                alpha=1,
                color=cc[dir]
                )



    f.suptitle(neoUtils.get_root(blk,unit_num))
    return(0)


