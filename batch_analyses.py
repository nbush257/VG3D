import os
import neo
from neo.io import NeoMatlabIO as NIO
from elephant.statistics import isi,cv,lv
import glob
from neo_utils import *
from spikeAnalysis import *
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ================== BEGIN SPIKE STATS ================================#
# This  section gets all the spike stats and puts them in one file
p = r'C:\Users\guru\Box Sync\__VG3D\_E3D_1K\deflection_trials'
file_list = glob.glob(os.path.join(p,'*NEO.mat'))
all_ISI = {}
all_CV = {}
all_LV = {}
all_latencies = {}
all_trains = {}
all_FR = {}

for file in file_list:
    print os.path.basename(file)
    fid = NIO(os.path.join(p,file))
    blk = fid.read_block()
    add_channel_indexes(blk)

    FRs, ISIs, trains = get_contact_sliced_trains(blk)

    for name,cell in ISIs.iteritems():
        cname = os.path.splitext(os.path.basename(file))[0]
        cname = re.search('^rat\d{4}_\d{2}_[A-Z]{3}\d{2}_VG_[A-Z]\d',cname).group()+'_'+name

        ISI = np.concatenate(cell) * cell[0].units

        CV = cv(ISI[np.isfinite(ISI)])
        if len(ISI[np.isfinite(ISI)])==0:
            LV = np.nan
        else:
            LV = lv(ISI[np.isfinite(ISI)].squeeze())
        # output
        all_ISI[cname] = ISI
        all_CV[cname] = CV
        all_LV[cname] = LV

    for name, cell in trains.iteritems():
        cname = os.path.splitext(os.path.basename(file))[0]
        cname = re.search('^rat\d{4}_\d{2}_[A-Z]{3}\d{2}_VG_[A-Z]\d', cname).group() + '_' + name

        latency = np.empty(len(cell))*pq.ms;latency[:]=np.NaN
        all_trains[cname]=[]
        for ii,contact in enumerate(cell):
            if len(contact)>0:
                latency[ii] = contact[0]-contact.t_start
            all_trains[cname].append(contact)
        all_latencies[cname] = latency
    for name,cell in FRs.iteritems():
        cname = os.path.splitext(os.path.basename(file))[0]
        cname = re.search('^rat\d{4}_\d{2}_[A-Z]{3}\d{2}_VG_[A-Z]\d', cname).group() + '_'+ name
        all_FR[cname] = cell

save_dict = {'all_FR':all_FR,
             'all_ISI':all_ISI,
             'all_CV':all_CV,
             'all_LV':all_LV,
             'all_latencies':all_latencies,
             'all_trains':all_trains}

savemat(os.path.join(p,'all_ISIs.mat'),save_dict)

# ================== END SPIKE STATS =================== #
#
#
#
#
