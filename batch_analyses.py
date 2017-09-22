import os
import neo
from neo.io import NeoMatlabIO as NIO
from elephant.statistics import isi,cv,lv
import glob
p = r'C:\Users\nbush257\Box Sync\__VG3D\_E3D_1K\deflection_trials'
for file in glob.glob(os.path.join(p,'*NEO.mat')):
    fid = NIO(os.path.join(p,file))
    blk = fid.read_block()
    all_sp = {}

    for seg in blk.segments:
        for cell_num,train in enumerate(seg.spiketrains):
            sp = all_sp[cell_num]
            sp = np.array([])
            for start,dur in zip(seg.epochs[0].times,seg.epochs[0].durations):
                sp=np.concatenate((sp,np.array(train.time_slice(start,start+dur))-np.array(start)))


    ISI = isi(sp,0)
    ISI = ISI[ISI<100]
    CV = cv(ISI)
    LV = lv(ISI.squeeze())



for seg in blk.segments:
    for train in seg.spiketrains:
