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
from neo.io import NixIO as NIO
from sklearn.preprocessing import StandardScaler
import sklearn


# import my functions
proc_path =os.environ['PROC_PATH']
sys.path.append(os.path.join(proc_path,r'VG3D\modules'))
sys.path.append(os.path.join(proc_path,r'VG3D\scripts'))

def get_blk(f='rat2017_08_FEB15_VG_D1_NEO.h5',fullname=False):
    '''loads in a NEO block from a pickle file. Calling without arguments pulls in a default file'''
    box_path = os.environ['BOX_PATH']
    dat_path = os.path.join(box_path,r'__VG3D\_deflection_trials\_NEO')
    if not fullname:
        fid = NIO(os.path.join(dat_path,f))
    else:
        fid = NIO(f)

    return fid.read_block()

def get_rate_b(blk,unit_num,sigma=10*pq.ms):
    '''
    convinience function to get a standard rate and binary spike vector from a block and unit number

    :param blk: neo block
    :param unit_num: int dictating which unit to load
    :param sigma: gaussian kernal smoothing parameter. Must be a quantity
    :return:    r - soothed spike rate
                b - binary spike vector
    '''
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


def concatenate_epochs(blk,epoch_idx=1):
    '''
    takes a block which may have multiple segments and returns a single Epoch which has the contact epochs concatenated and properly time aligned
    :param blk: a neo block
    :return: contact -- a single neo epoch array
    '''
    starts = np.empty([0])
    durations = np.empty([0])
    t_start = 0*pq.ms
    for seg in blk.segments:
        epoch = seg.epochs[epoch_idx]
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
    '''
    Grabs the boundaries of the NaNs (where nan sections start and stop
    :param var: Vector of a signal where we want to know where NaN sections start and stop
    :return: starts, stops -- indices of the input array where nan sections start and stop
    '''
    if var.ndim>1:
        raise ValueError('input needs to be 1D')
    nans = nan_helper(var)[0].astype('int')
    d = np.diff(np.concatenate([[0],nans]))
    return np.where(d==1)[0],np.where(d==-1)[0]


def replace_NaNs(var, mode='zero',pad=20):
    '''
    takes a Vector, array, or neo Analog signal and replaces the NaNs with desired values
    :param var: numpy array or neo analog signal to replace NaNs
    :param mode: string indicating what to replace NaNs with:
                    'zero'      --  replace all NaNs with zero
                    'median'    --  replace all NaNs with vector's median. Assumes matrix columns are different variables
                    'rm'        --  removes all NaNs CHANGES THE SHAPE OF THE ARRAY
                    'interp'    --  linear interpolation over NaN sections using end points
                    'pchip'     --  spline interpolation over NaN secitions using endpoints. Uses PAD number of samples on either side of the gap to compute the spline
    :param pad: used in pchip interpolation to determine how much data to use when creating spline
    :return:    data -- the interpolated input data, in the same type as was input.

    '''

    # copy input to new variable to prevent inplace operations and convert from NEO if needed.

    if type(var)==neo.core.analogsignal.AnalogSignal:
        data = var.magnitude.copy()
    else:
        data = var.copy()

    # Apply replacement depending on mode
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
        for ii in xrange(data.shape[1]):
            starts,stops = nan_bounds(data[:, ii])
            for start,stop in zip(starts,stops):
                xi = np.concatenate([np.arange(start-pad,start),np.arange(stop,stop+pad)])
                yi = data[xi, ii]

                x = np.arange(start,stop)
                y = scipy.interpolate.pchip_interpolate(xi,yi,x)
    # catch undefined modes
    else:
        raise ValueError('Wrong mode indicated. May want to impute NaNs in some instances')

    # map output to neo signal if needed
    if type(var)==neo.core.analogsignal.AnalogSignal:
        var_out = neo.core.AnalogSignal(data*var.units,
                                        t_start=0.*pq.ms,
                                        sampling_rate=var.sampling_rate,
                                        name=var.name)
        return(var_out)
    else:
        return(data)



def get_Cbool(blk,use_bool=True):
    '''
    Given a block,get a boolean vector of contact for the concatenated data
    :param blk: neo block
    :param use_bool: a boolean as to whether to use the original C vector of the curated use_flags vector.
    :return: Cbool - a boolean numpy vector of contact
    '''
    Cbool = np.array([],dtype='bool')
    # Get contact from all available segments and offset appropriately
    for seg in blk.segments:
        seg_bool = np.zeros(len(seg.analogsignals[0]),dtype='bool')
        if use_bool:
            epochs = seg.epochs[-1]
        else:
            epochs = seg.epochs[0]

        # Set the samples during contact to True
        for start,dur in zip(epochs,epochs.durations):
            start = int(start)
            dur = int(dur)
            seg_bool[start:start+dur]=1

        Cbool = np.concatenate([Cbool,seg_bool])
    return Cbool


def get_root(blk,cell_no):
    '''
    Utility function that gets a unique ID string for each cell
    :param blk: neo block of the data
    :param cell_no: index of the cell number
    :return: root - a unique string ID
    '''
    return(blk.annotations['ratnum'] + blk.annotations['whisker'] + 'c{:01d}'.format(cell_no))


def get_deriv(var,smooth=False):
    ''' returns the temporal derivative of a numpy array with time along the 0th axis'''
    if var.ndim==1:
        var = var[:,np.newaxis]
    if var.shape[1]>var.shape[0]:
        raise Warning('Matrix was wider than it is tall, are variables in the columns?')
    # assumes a matrix or vector where the columns are variables
    if smooth:
        var = savgol_filter(var,window_length=21)

    return(np.gradient(var,axis=0)[0])

def epoch_to_cc(epoch):
    ''' take a NEO epoch representing contacts and turn it into an Nx2 matrix which
    has contact onset in the first column and contact offset in the second.'''
    cc = np.empty([len(epoch),2])
    cc[:,0] = np.array(epoch.times).T
    cc[:,1] = cc[:,0]+np.array(epoch.durations).T

    print('cc is in {}'.format(epoch.units))
    return cc.astype('int64')

def get_MB_MD(data_in):
    '''
    return the Bending magnitude (MB) and bending direction (MD) for a given moment signal..
    Accepts neo block, neo analog signal, or numpy array
    :param data_in:
    :return: MB, MD -- either neo analog signals or numpy 1D arrays (matches input type) of bending magnitude and direction
    '''

    # accept block, analog signal, or numpy array
    if type(data_in)==neo.core.block.Block:
        M = get_var(blk,'M')
    elif type(data_in)==neo.core.analogsignal.AnalogSignal:
        dat = data_in.magnitude
    elif type(data_in)==np.ndarray:
        dat = data_in

    MD = np.arctan2(dat[:, 2], dat[:, 1])
    MB = np.sqrt(dat[:, 1] ** 2 + dat[:, 2] ** 2)
    if type(data_in)==neo.core.analogsignal.AnalogSignal or type(data_in)==neo.core.block.Block:
        MD = neo.core.AnalogSignal(MD, units=pq.radians, sampling_rate=pq.kHz)
        MB = neo.core.AnalogSignal(MB, units=pq.N*pq.m, sampling_rate=pq.kHz)
    return (MB, MD)

def applyPCA(var,Cbool):
    '''
    apply PCA to a given signal only during contact. Can accept neo analog signal.
    Useful as convenience function which takes care of contact masking and interpolating NaNs.
    :param var: either analog signal or numpy array of signal to apply PCA to
    :param Cbool: boolean contact
    :return: transformed inputs, pca object
    '''
    if type(var) == neo.core.analogsignal.AnalogSignal:
        var = var.magnitude
    var[np.invert(Cbool),:] =0
    var = replace_NaNs(var,'interp')
    scaler=StandardScaler(with_mean=False)
    var = scaler.fit_transform(var)
    pca = sklearn.decomposition.PCA()

    PC = pca.fit_transform(var)
    return(PC,pca)


def get_analog_contact_slices(var, contact, slice2array=True):
    '''
    Takes all the contact intervals and extracts a contact onset centered slice of the variable. 
    :param var:             The analog signal you want to slice. Either a numpy array, quantity, or neo analogsignal
    :param contact:         The contact times. Can either be a list of epochs or a numpy boolean vector
    :param slice2array      Boolean whether to put all the slices into an array with all entries fit into the size of the longest contact. If false, returns a list.
    
    :return var_out:  Either a list or a numpy array of the sliced input variable, depending on the input 'slice2array' flag
                        If a numpy array output, dims are: [time after contact onset, contact index, var dimension]
                            --Size of the first dimension is the length of the longest contact
    '''

    # map analog signal to numpy array
    if type(var)==neo.core.analogsignal.AnalogSignal:
        var = var.magnitude

    var_slice = []

    # if the contact input is a boolean numpy array:
    if type(contact)==np.ndarray:
        if not((contact.dtype=='bool') and (contact.ndim==1)):
            raise ValueError('If contact is an array, it must be a boolean vector')

        starts = (np.where(np.diff(contact.astype('int')) == 1)[0] + 1)
        stops = (np.where(np.diff(contact.astype('int')) == -1)[0] + 1)
        for start_idx, stop_idx in zip(starts, stops):
            var_slice.append(var[start_idx:stop_idx,:])
    # if the contact input is an epoch
    elif type(contact)==neo.core.epoch.Epoch:
        for start_idx,dur in zip(contact,contact.durations):
            var_slice.append(var[int(start_idx):int(start_idx+dur),:])


    if slice2array:
        max_l = max([len(x) for x in var_slice])
        var_out = np.empty([max_l,len(var_slice),var_slice[0].shape[-1]])
        var_out[:]=np.nan
        for ii,slice in enumerate(var_slice):
            var_out[:slice.shape[0],ii,:] = slice
    else:
        var_out = var_slice

    return var_out