import os
import neo
from neo.io import PickleIO as PIO
from elephant.statistics import isi,cv,lv
import glob
from neo_utils import *
from spikeAnalysis import *
import matplotlib.pyplot as plt
import seaborn as sns
import re


def calculate_spike_stats(p,pre=10*pq.ms,post=10*pq.ms):
    ''' takes all the pickle NEO files in a directory and computes:
    ISI,CV,LV,latency,spikes trains,firing rate
    for all the cells during individaul contact times.
    '''
    file_list = glob.glob(os.path.join(p,'*NEO.pkl'))
    all_ISI = []
    all_CV = []
    all_LV = []
    all_latencies = []
    all_trains = []
    all_FR = []
    mean_CVs = []
    mean_LVs= []
    all_id = {}
    count=0
    # Loop over files and load
    for file in file_list:
        print os.path.basename(file)
        fid = PIO(os.path.join(p,file))
        blk = fid.read_block()

        # loop over the units in a trail
        for unit in blk.channel_indexes[-1].units:

            # get the ID of the cell
            root = get_root(blk,int(unit.name[-1]))
            # get the spike trains, ISIs, and FRs from each contact interval
            FR, ISI, trains = get_contact_sliced_trains(blk, unit, pre=pre, post=post)
            # get the variation stats on each contact interval
            CV_array,LV_array = get_CV_LV(ISI)

            # get the latency to first spike of the contact.
            # This will contain negative numbers because because
            # we are considering spikes that occur in some pad before the "contact" as determined by NN
            latency = np.empty(len(trains))
            latency[:] = np.nan
            for ii,train in enumerate(trains):
                if len(train)>0:
                    latency[ii] = train[0]-train.t_start-pre
            latency = latency * train.units

            # output to lists
            all_ISI.append(ISI)
            all_CV.append(CV_array)
            all_LV.append(LV_array)
            all_latencies.append(latency)
            all_trains.append(trains)
            all_FR.append(FR)
            mean_CVs.append(np.nanmean(CV_array))
            mean_LVs.append(np.nanmean(LV_array))
            all_id[root]=count
            count+=1



    # save to npz
    np.savez(os.path.join(p,'population_spike_stats.npz'),
             all_FR=all_FR,
             all_ISI=all_ISI,
             all_CV=all_CV,
             all_LV=all_LV,
             all_latencies=all_latencies,
             all_trains=all_trains,
             all_id=all_id,
             mean_CVs=mean_CVs,
             mean_LVs=mean_LVs,
             pre=pre,
             post=post,
             )

def main(argv=None):
    if argv==None:
        argv=sys.argv
    pickle_path = argv[1]
    calculate_spike_stats(pickle_path)


if __name__=='__main__':
    sys.exit(main())