from neo.io import NeoMatlabIO as NIO
from neo.core import Block,ChannelIndex,Unit
from elephant.conversion import binarize
import quantities as pq
def get_var(blk,varname='M',join=True,keep_neo=False):
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

def concatenate_sp(blk):
    sp = {}
    for unit in blk.channel_indexes[-1].units:
        sp[unit.name] = np.array([])*pq.ms
        t_start = 0.*pq.ms
        for train in unit.spiketrains:
            new_train = np.array(train)*pq.ms+t_start
            sp[unit.name] = np.append(sp[unit.name],new_train) *pq.ms
            t_start +=train.t_stop
    return sp

def replace_NaNs(var,mode='zero'):
    if mode=='zero':
        var[np.isnan(var)]=0
    elif mode=='median':
        m = np.nanmedian(var,0)
        idx = np.any(np.isnan(var))
        var[np.any(np.isnan(var),1),:] = m
    elif mode=='rm':
        idx = np.any(np.isnan(var),1)
        var=np.delete(var,np.where(idx)[0],axis=0)
    else:
        raise ValueError('Wrong mode indicated. May want to impute NaNs in some instances')

def create_unit_chan(blk):
    chx = ChannelIndex(0,name='Units')
    num_units = []
    for seg in blk.segments:
        num_units.append(len(seg.spiketrains))
    num_units = max(num_units)
    for ii in xrange(num_units):
        unit = Unit(name='cell_{}'.format(ii))
        chx.units.append(unit)
    for seg in blk.segments:
        for ii,train in enumerate(seg.spiketrains):
            chx.units[ii].spiketrains.append(train)
    blk.channel_indexes.append(chx)

def create_analog_chan(blk):
    '''maps the mechanical and kinematic signals to a channel index.'''
    varnames = ['M','F','PHIE','TH','Rcp','THcp','PHIcp']
    for ii in range(7):
        chx = ChannelIndex(0,name=varnames[ii])
        blk.channel_indexes.append(chx)
    for seg in blk.segments:
        for chx,sig in zip(blk.channel_indexes,seg.analogsignals):
            chx.analogsignals.append(sig)

def add_channel_indexes(blk):
    create_analog_chan(blk)
    create_unit_chan(blk)



# # This code below might be useful for contact psths

