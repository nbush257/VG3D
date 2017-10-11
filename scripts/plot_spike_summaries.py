from spikeAnalysis import *
import glob
from neo.io import PickleIO as PIO
import quantities as pq
from elephant.statistics import cv,lv
import os
import matplotlib.pyplot as plt
import seaborn as sns
import quantities as pq
sns.set()


def main(p,file,save_path):
    pre = 10*pq.ms
    post = 10*pq.ms
    fid = PIO(file)
    blk = fid.read_block()
    FR,ISI,contact_trains = get_contact_sliced_trains(blk,pre=pre,post=post)
    binsize = 2*pq.ms
    for unit in blk.channel_indexes[-1].units:
        root = blk.annotations['ratnum'] + blk.annotations['whisker'] + 'c{}'.format(unit.name[-1])
        trains = contact_trains[unit.name]
        all_isi = np.array([])
        CV_array = np.array([])
        LV_array = np.array([])
        for interval in ISI[unit.name]:
            all_isi = np.concatenate([all_isi,interval])
            if np.all(np.isfinite(interval)):
                CV_array = np.concatenate([CV_array,[cv(interval)]])
                LV_array = np.concatenate([LV_array,[lv(interval)]])

        all_isi = all_isi * interval.units
        CV_array = CV_array
        CV = np.mean(CV_array)
        LV = np.mean(LV_array)

        ## calculate data for PSTH
        b,durations = get_binary_trains(contact_trains[unit.name])
        b_times = np.where(b)[1] * pq.ms#interval.units
        b_times-=pre
        PSTH,t_edges = np.histogram(b_times,bins=np.arange(-np.array(pre),np.max(durations)+np.array(post),float(binsize)))
        plt.bar(t_edges[:-1],
                PSTH.astype('f8')/len(durations)/binsize*1000,
                width=float(binsize),
                align='edge',
                alpha=0.8
                )

        ax = plt.gca()
        thresh = 500 * pq.ms
        ax.set_xlim(-15, thresh.__int__())
        ax.set_xlabel('Time after contact (ms)')
        ax.set_ylabel('Spikes per second')
        ax.set_title('PSTH for: {}'.format(root))

        plt.savefig(os.path.join(save_path,root+'_PSTH.png'),dpi=300)
        plt.close('all')
        # ============================================

        # PLOT ISIs
        plt.figure()
        thresh = 100 * pq.ms
        if len(all_isi[np.logical_and(np.isfinite(all_isi), all_isi < thresh)])==0:
            return
        ax = sns.distplot(all_isi[np.logical_and(np.isfinite(all_isi), all_isi < thresh)],
                          bins=np.arange(0,100,1),
                          kde_kws={'color':'k','lw':3,'alpha':0.5,'label':'KDE'})
        ax.set_xlabel('ISI '+all_isi.dimensionality.latex)
        ax.set_ylabel('Percentage of all ISIs')

        a_inset = plt.axes([.55, .5, .2, .2], facecolor='w')
        a_inset.grid(color='k',linestyle=':',alpha=0.4)
        a_inset.axvline(CV,color='k',lw=0.5)
        a_inset.set_title('CV = {:0.2f}\nLV = {:0.2f}'.format(CV,LV))
        a_inset.set_xlabel('CV')
        a_inset.set_ylabel('# of Contacts')
        sns.distplot(CV_array,color='g',kde=False)
        ax.set_title('ISI distribution for {}'.format(root))
        plt.savefig(os.path.join(save_path, root + '_ISI.png'), dpi=300)
        plt.close('all')

if __name__=='__main__':
    p = r'C:\Users\guru\Box Sync\__VG3D\deflection_trials\data'
    save_path = r'C:\Users\guru\Box Sync\__VG3D\deflection_trials\figs'
    files = glob.glob(os.path.join(p, '*.pkl'))
    for file in files:
        print(file)
        try:
            main(p,file,save_path)
        except:
            pass

