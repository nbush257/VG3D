import neo
import os
import sys
from neo.core import Block,ChannelIndex,Unit,SpikeTrain,AnalogSignal
from elephant.conversion import binarize
import neo
import quantities as pq
import numpy as np
import scipy
import elephant
from neo.io import PickleIO as PIO
from sklearn.preprocessing import StandardScaler

# import my functions
proc_path =os.environ['PROC_PATH']
sys.path.append(os.path.join(proc_path,r'VG3D\modules'))
sys.path.append(os.path.join(proc_path,r'VG3D\scripts'))

def get_blk(f):
    box_path = os.environ['BOX_PATH']
    dat_path = os.path.join(box_path,r'__VG3D\deflection_trials\data')
    fid = PIO(os.path.join(dat_path,f))
    return fid.read_block()

def get_rate_b(blk,unit_num,sigma=10*pq.ms):
    sp = concatenate_sp(blk)['cell_{}'.format(unit_num)]
    kernel = elephant.kernels.GaussianKernel(sigma=sigma)
    b = elephant.conversion.binarize(sp,sampling_rate=pq.kHz)[:-1]
    r = elephant.statistics.instantaneous_rate(sp,sampling_period=pq.ms,kernel=kernel).magnitude.squeeze()
    return(r,b)


def get_var(blk,varname='M',join=True,keep_neo=True):
    ''' use this utility to access an analog variable from all segments in a block easily
    If you choose to join the segments, returns a list of '''
    varnames = ['M','F','TH','PHIE','ZETA','Rcp','THcp','PHIcp','Zcp']
    idx = varnames.index(varname)
    split_points = []
    var = []
    # Create a list of the analog signals for each segment
    for seg in blk.segments:
        if keep_neo:
            var.append(seg.analogsignals[idx])
        else:
            var.append(seg.analogsignals[idx].as_array())
            split_points.append(seg.analogsignals[idx].shape[0])

    if join:
        if keep_neo:
            data = np.empty([0,var[0].shape[-1]])
            t_start = 0.*pq.s
            t_stop = 0.*pq.s
            for seg in var:
                data = np.append(data,seg.as_array(),axis=0)
                t_stop +=seg.t_stop

            sig = AnalogSignal(data*var[0].units,
                                        t_start=t_start,
                                        sampling_rate=var[0].sampling_rate,
                                        name=var[0].name)
            return sig
        else:
            var = np.concatenate(var,axis=0)
        return (var, split_points)
    else:
        return var


def concatenate_sp(blk):
    ''' takes a block and concatenates the spiketrains for each unit across segments'''
    sp = {}
    for unit in blk.channel_indexes[-1].units:
        sp[unit.name] = np.array([])*pq.ms
        t_start = 0.*pq.ms
        for train in unit.spiketrains:
            new_train = np.array(train)*pq.ms+t_start
            sp[unit.name] = np.append(sp[unit.name],new_train) *pq.ms
            t_start +=train.t_stop

        sp[unit.name] = SpikeTrain(sp[unit.name], t_stop = t_start)
    return sp


def concatenate_epochs(blk):
    starts = np.empty([0])
    durations = np.empty([0])
    t_start = 0*pq.ms
    for seg in blk.segments:
        epoch = seg.epochs[0]
        starts = np.append(starts,epoch.times+t_start,axis=0)
        durations = np.append(durations,epoch.durations.ravel())
        t_start+=seg.t_stop
    contact = neo.core.Epoch(times=starts*pq.ms,durations=durations*pq.ms,units=pq.ms)
    return contact


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
       # linear interpolation of NaNs
        nans, x= nan_helper(y)
        y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def nan_bounds(var):
    if var.ndim>1:
        raise ValueError('input needs to be 1D')
    nans = nan_helper(var)[0].astype('int')
    d = np.diff(np.concatenate([[0],nans]))
    return np.where(d==1)[0],np.where(d==-1)[0]


def replace_NaNs(var, mode='zero'):
    if type(var)==neo.core.analogsignal.AnalogSignal:
        data = var.as_array().copy()
    else:
        data = var.copy()

    if mode=='zero':
        data[np.isnan(data)]=0
    elif mode=='median':
        m = np.nanmedian(data, 0)
        idx = np.any(np.isnan(data))
        data[np.any(np.isnan(data), 1), :] = m
    elif mode=='rm':
        idx = np.any(np.isnan(data), 1)
        data=np.delete(data, np.where(idx)[0], axis=0)
    elif mode=='interp':
        for ii in xrange(data.shape[1]):
            nans, x = nan_helper(data[:, ii])
            data[nans, ii] = np.interp(x(nans), x(~nans), data[~nans, ii])
    elif mode=='pchip':
        pad=20
        for ii in xrange(data.shape[1]):
            starts,stops = nan_bounds(data[:, ii])
            for start,stop in zip(starts,stops):
                xi = np.concatenate([np.arange(start-pad,start),np.arange(stop,stop+pad)])
                yi = data[xi, ii]

                x = np.arange(start,stop)
                y = scipy.interpolate.pchip_interpolate(xi,yi,x)
    else:
        raise ValueError('Wrong mode indicated. May want to impute NaNs in some instances')

    if type(var)==neo.core.analogsignal.AnalogSignal:
        var_out = neo.core.AnalogSignal(data*var.units,
                                        t_start=0.*pq.ms,
                                        sampling_rate=var.sampling_rate,
                                        name=var.name)
        return(var_out)
    else:
        return(data)


def create_unit_chan(blk):
    chx = neo.core.ChannelIndex(index=np.array([0]),name='electrode_0')

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


def create_analog_chan(blk):
    '''maps the mechanical and kinematic signals to a channel index.'''
    varnames = ['M','F','TH','PHIE','ZETA','Rcp','THcp','PHIcp','Zcp']
    chx_list = []
    for ii in range(len(varnames)):
        chx = neo.core.ChannelIndex(np.array([0]),name=varnames[ii])
        chx_list.append(chx)
    for seg in blk.segments:
        for chx,sig in zip(chx_list,seg.analogsignals):
            chx.analogsignals.append(sig)
    return chx_list



def append_channel_indexes(blk):
    chx_list = create_analog_chan(blk)
    units = create_unit_chan(blk)
    for chx in chx_list:
        blk.channel_indexes.append(chx)
    blk.channel_indexes.append(units)


def get_Cbool(blk):
    Cbool = np.array([],dtype='bool')

    for seg in blk.segments:
        seg_bool = np.zeros(len(seg.analogsignals[0]),dtype='bool')
        epochs = seg.epochs[0]
        for start,dur in zip(epochs,epochs.durations):
            start = int(start)
            dur = int(dur)
            seg_bool[start:start+dur]=1
        Cbool = np.concatenate([Cbool,seg_bool])
    return Cbool


def get_root(blk,cell_no):
    return(blk.annotations['ratnum'] + blk.annotations['whisker'] + 'c{:01d}'.format(cell_no))


def import_data_to_model(file,vars=['M','F'],unit_idx=0):
    ''' this is a utility to do all the common preprocessing steps I do for the modelling'''
    fid = PIO(file)
    blk = fid.read_block()
    X = np.array([])
    for var in vars:
        if len(X)==0:
            X = get_var(blk,var)[0]
        else:
            X = np.append(X,get_var(blk,var)[0],axis=1)
    unit = blk.channel_indexes[-1].units[unit_idx]
    sp = concatenate_sp(blk)[unit.name]
    b = elephant.conversion.binarize(sp, sampling_rate=pq.kHz)[:-1]
    y = b[:, np.newaxis].astype('f8')
    Cbool = get_Cbool(blk)
    X[np.invert(Cbool), :] = 0
    X = replace_NaNs(X, 'pchip')
    X = replace_NaNs(X, 'interp')
    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(X)

    return X,y,b,sp,blk,Cbool


