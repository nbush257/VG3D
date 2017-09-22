from neo.io import NeoMatlabIO as NIO
from neo.core import Block
from elephant.conversion import binarize
import quantities as pq
def getVar(blk,varname='M',join=True,keep_neo=False):
    ''' use this utility to access an analog variable from all segments in a block easily
    If you choose to join the segments, returns a list of '''
    varnames = ['M','F','PHIE','TH','Rcp','THcp','PHIcp']
    idx = varnames.index(varname)
    split_points = []
    var = []
    if keep_neo and join:
        raise ValueError('Cannot join and keep neo structure')

    for seg in blk.segments:
        if keep_neo:
            var.append(seg.analogsignals[idx])
        else:
            var.append(seg.analogsignals[idx].as_array())
            split_points.append(seg.analogsignals[idx].shape[0])
    if join:
        var = np.concatenate(var,axis=0)
        return (var, split_points)
    else:
        return var

def replaceNaNs(var,mode='zero'):
    if mode=='zero':
        var[np.isnan(var)]=0
    elif mode=='median':
        m = np.nanmedian(var,0)
        idx = np.any(np.isnan(var))
        var[np.any(np.isnan(var),1),:] = m
    else:
        raise ValueError('Wrong mode indicated. May want to impute NaNs in some instances')


def create_chan(blk):
    chx = neo.core.ChannelIndex(0)
    num_units = []
    for seg in blk.segments:
        num_units.append(len(seg.spiketrains))
    num_units = max(num_units)
    for ii in xrange(num_units):
        unit = neo.core.Unit(name='cell_{}'.format(ii))
        chx.units.append(unit)
    for seg in blk.segments:
        for ii,train in enumerate(seg.spiketrains):
            chx.units[ii].spiketrains.append(train)
    return chx


# This code below might be useful for contact psths
chx = create_chan(blk)
f = plt.figure()
for ii,unit in enumerate(chx.units):
    f.add_subplot(len(chx.units),1,ii+1)
    PSTH =[]
    for train,seg in zip(unit.spiketrains,blk.segments):
        epoch = seg.epochs[0]
        for start,dur in zip(epoch.times,epoch.durations):
            try:
                train_slice = train.time_slice(start, start + dur)
                # b = binarize(train_slice, sampling_rate=pq.kHz)
                # plt.plot(b,'k',alpha=0.01)
            except:
                pass
            PSTH.append(train_slice)

