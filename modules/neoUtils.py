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
import statsmodels.nonparametric.smoothers_lowess as sls
import warnings
# import my functions

proc_path =os.environ['PROC_PATH']
sys.path.append(os.path.join(proc_path,r'VG3D\modules'))
sys.path.append(os.path.join(proc_path,r'VG3D\scripts'))

def get_blk(f='rat2017_08_FEB15_VG_D1_NEO.h5'):
    '''loads in a NEO block from a pickle file. Calling without arguments pulls in a default file'''
    try:
        box_path = os.environ['BOX_PATH']
        dat_path = os.path.join(box_path,r'__VG3D\_deflection_trials\_NEO')
    except:
        print('Box path not found')
        pass


    if os.path.isfile(f):
        fid = NIO(f,mode='ro')
    else:
        fid = NIO(os.path.join(dat_path,f),mode='ro')

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

    split_points = []
    var = []
    # Create a list of the analog signals for each segment
    for seg in blk.segments:
        names = [str(x.name) for x in seg.analogsignals]
        names =[w.replace('Moment', 'M') for w in names]
        names = [w.replace('Force', 'F') for w in names]
        idx = names.index(varname)
        if keep_neo:
            var.append(seg.analogsignals[idx])
        else:
            var.append(seg.analogsignals[idx].as_array())
            split_points.append(seg.analogsignals[idx].shape[0])

    if join:
        if keep_neo:
            data = []
            t_start = 0.*pq.s
            t_stop = 0.*pq.s
            for seg in var:
                data.append(seg.as_array())
                t_stop +=seg.t_stop
            data = np.concatenate(data,axis=0)
            sig = neo.AnalogSignal(data*var[0].units,
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


def replace_NaNs(var, mode='interp',pad=20):
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
                if (stop+pad)>data.shape[0]:
                    continue
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
    s = (blk.annotations['ratnum'] + blk.annotations['whisker'] + 'c{:01d}'.format(cell_no))
    return(s.replace('_',''))


def get_deriv(var,sgolay_tgl=False,window=11):
    ''' returns the temporal derivative of a numpy array with time along the 0th axis'''
    if var.ndim==1:
        var = var[:,np.newaxis]
    if var.shape[1]>var.shape[0]:
        raise Warning('Matrix was wider than it is tall, are variables in the columns?')
    if sgolay_tgl:
        var = scipy.signal.savgol_filter(var, window, 1, axis=0)

    return(np.gradient(var,axis=0))


def epoch_to_cc(epoch):
    ''' take a NEO epoch representing contacts and turn it into an Nx2 matrix which
    has contact onset in the first column and contact offset in the second.'''
    cc = np.empty([len(epoch),2])
    cc[:,0] = np.array(epoch.times).T
    cc[:,1] = cc[:,0]+np.array(epoch.durations).T

    print('cc is in {}'.format(epoch.units))
    return cc.astype('int64')


def Cbool_to_cc(Cbool):
    '''
    Helper function to turn a binary contact boolean into indexes of starts and stops
    :param Cbool: 
    :return cc: First column is start indices, second column is stop indices     
    '''
    Cbool = np.concatenate([[False],Cbool,[False]]).astype('int')
    starts = np.where(np.diff(Cbool) == 1)[0]
    stops = np.where(np.diff(Cbool) == -1)[0]
    return starts,stops

def epoch_to_bool(epoch,t_stop):
    """
    converts a neo epoch to a boolean vector, given the length of the desired vector
    :param epoch: a neo epoch of contact onsets and durations
    :param t_stop: the length of the desired boolean vector
    :return cbool: a boolean numpy vector of times where contact occurs
    """
    if type(t_stop) == pq.quantity.Quantity:
        t_stop.units=pq.ms
        t_stop = int(t_stop)
    elif type(t_stop) is int:
        pass
    else:
        raise ValueError('t_stop should be either a quantiity or an int')
    cbool =np.zeros(t_stop,dtype='bool')
    for (start,dur) in zip(epoch.times.magnitude,epoch.durations.magnitude):
        start = int(start)
        dur = int(dur)
        cbool[start:start+dur] = True
    return(cbool)


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

    MD = np.arctan2(dat[:, 2], dat[:, 1])[:,np.newaxis]
    MB = np.sqrt(dat[:, 1] ** 2 + dat[:, 2] ** 2)[:,np.newaxis]
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
        for start_idx,dur in zip(contact.times,contact.durations):
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


def get_mean_var_contact(blk, input=None, varname='Rcp'):
    '''
    Get the mean value of a variable for each contact where use_flags is the contact indicator

    :param input:         a neo block,neo analog signal, python quantity, or numpy array 
    :param varname:     variable name in the neo block. Default = 'Rcp'

    :return var_mean:   a [num_contacts x num_dims] quantities matrix of the mean variable value for each contact
    '''
    if input is None:
        var = get_var(blk, varname)
        unit = var.units
    elif type(input) is pq.quantity.Quantity or type(input) is neo.core.analogsignal.AnalogSignal:
        var = input
        unit = var.units
    elif type(input) is np.ndarray:
        var=input
        unit = pq.dimensionless

    use_flags = concatenate_epochs(blk, epoch_idx=-1)
    var_contacts = get_analog_contact_slices(var, use_flags).squeeze()
    var_contacts = np.nanmean(var_contacts * unit, axis=0)

    if var_contacts.ndim == 1:
        var_contacts = var_contacts[:, np.newaxis]

    return (var_contacts)


def get_contact_apex_idx(blk,use_world=True,mode='apex',thresh=0.75,time_win=10):
    '''
    Use the contact point to estimate the Apex of contact
    If Stretch is passed, calls the end of onset the first point at which the deflection
    is some percent of the maximal deflection distance, and the offset beginning
    is the last point in the deflection that was at that percentage of
    maximal deflection. This is a more intuitive onset in my opinion, and is robust
    to maximal points that occur at the wrong place.
    Possible modes:
            'apex' - gets the idx of max deflection
            'thresh' - gets the first point passing a threshold and last point coming down through that threshold
            'time_win' - uses a constant window around beginning and end of contact
    :param blk: 
    :return: 
    '''
    use_flags= concatenate_epochs(blk,-1)
    if use_world:
        CP = CP_to_world(blk)
    else:
        CP = get_var(blk, 'CP')
    CP_contacts = get_analog_contact_slices(CP,use_flags)
    CP_contacts = center_var(CP_contacts)
    D = np.sqrt(CP_contacts[:,:,0]**2+CP_contacts[:,:,1]**2+CP_contacts[:,:,2]**2)
    if mode=='thresh':
        for ii in range(D.shape[-1]):
            mask = np.isfinite(D[:,ii])
            D[mask,ii] = D[mask,ii]/np.nanmax(D[mask,ii])

    # catch all nan slices
    nan_idx = np.all(np.isnan(D),axis=0)
    D[:,nan_idx]=0

    # find maximum
    if mode=='thresh':
        D_bool = D>thresh
        onset = np.empty(D.shape[-1],dtype='int')
        offset = np.empty(D.shape[-1],dtype='int')
        for ii in range(D_bool.shape[-1]):
            onset[ii] = np.where(D_bool[:,ii])[0][0]
            offset[ii] = np.where(D_bool[:,ii])[0][-1]
    elif mode=='apex':
        onset = np.nanargmax(D,axis=0)
        offset = onset.copy()

    elif mode=='time_win':
        onset = np.repeat(time_win,D.shape[1])
        c_length = np.array([np.where(np.isfinite(D[:,ii]))[0][-1] for ii in range(D.shape[1])])
        too_long = time_win>c_length
        onset[too_long]=c_length[too_long]
        offset=c_length-time_win+1
        negs = offset<0
        if np.any(negs):
            warnings.warn('Time window is larger than {} of {} contacts'.format(negs.sum(),D.shape[1]))
            offset[negs]=0

    return(onset,offset)

def get_value_at_idx(var,idx):
    '''
    use this to extract the value of a variable at a given index if you have a list or array of indices
    This is useful for getting a point in from every contact but that point may be different in each contact
    :param var: 
    :param idx: 
    :return: The value of a variable at the given index 
    '''
    var_out = [var[x, ii, :] for ii, x in enumerate(idx)]
    return(np.array(var_out))



def smooth_var_lowess(sig,window=50):
    '''
    Perform Lowess smoothing on a time series. This smoothing is very good because it 
    does not have any issues with nan discontinuities
    
    :param sig: a numpy matrix of [n_samples x n_dim], or a neo analog signal or a quantity 
    :param window: a smoothing paramter. number of smaples to cinsider during the rewighting
    
    :return out: a matrix like that of the input. Does not reconvert into the input type yet. 
    '''
    if ((type(sig) is neo.core.analogsignal.AnalogSignal) or
        (type(sig) is pq.quantity.Quantity)):
        sig = sig.magnitude

    out = np.empty_like(sig)
    out[:] = np.nan
    frac=float(window)/sig.shape[0]
    for ii in xrange(sig.shape[1]):
        endog = sig[:,ii]
        fitted = sls.lowess(endog,np.arange(len(endog)),is_sorted=True,frac=frac,delta=2)
        out[fitted[:,0].astype('int'),ii] = fitted[:,1]
    return out

def center_var(var,use_flags=None):
    ''' performs centering to contact onset'''
    if var.ndim>2 or use_flags is None:
        sliced=True
    else:
        sliced=False
    var_out = var.copy()
    if sliced:
        for ii in xrange(var.shape[1]):
            var_slice = var[:,ii,:]
            if np.all(np.isnan(var_slice)):
                continue

            first_index = np.where(np.all(np.isfinite(var_slice),axis=1))[0][0]
            var_out[:,ii,:]-=var_slice[first_index,:]
    else:
        for start,dur in zip(use_flags.times,use_flags.durations):
            start = int(start)
            dur = int(dur)
            if np.any(np.isfinite(var[start:start+dur,:])):
                first_index = np.where(np.all(np.isfinite(var[start:start+dur]),axis=1))[0][0]
                var_out[start+first_index:start+dur+first_index,:]-=var[start+first_index,:]
    return(var_out)

def CP_to_world(blk):
    '''
    Transforms the contact point back into the world reference frame,
    accounting for both the rotation and bending of the whisker.

    :param blk:
    :return CP_world:
    '''
    CP = get_var(blk,'CP',keep_neo=False)[0]
    PH = get_var(blk,'PHIE',keep_neo=False)[0]
    PH = np.deg2rad(PH)
    TH = get_var(blk, 'TH',keep_neo=False)[0]
    TH =np.deg2rad(TH)
    Z = get_var(blk,'ZETA',keep_neo=False)[0]
    # Z = np.deg2rad(Z)
    BP = get_var(blk, 'BPm',keep_neo=False)[0]
    cbool = get_Cbool(blk)

    CP_world = np.empty_like(CP); CP_world[:]=np.nan
    def RX(theta):
        c = np.cos(theta)[0]
        s = np.sin(theta)[0]
        return(np.array([[1,0,0],[0,c,-s],[0,s,c]]))
    def RY(theta):
        c = np.cos(theta)[0]
        s = np.sin(theta)[0]
        return(np.array([[c,0,s],[0,1,0],[-s,0,c]]))
    def RZ(theta):
        c = np.cos(theta)[0]
        s = np.sin(theta)[0]
        return(np.array([[c,-s,0],[s,c,0],[0,0,1]]))
    for ii in xrange(CP.shape[0]):
        if cbool[ii]:
            ROT = np.linalg.multi_dot([RX(-Z[ii]),RY(-PH[ii]),RZ(-TH[ii])])
            CP_world[ii,:] = np.dot(np.linalg.inv(ROT),CP[ii,:][:,np.newaxis]).T
            CP_world[ii,:]+=BP[ii,:]
    return CP_world

def shuffle_spiketrain(sp,use_flags):
    """
    takes a spike train and shuffles the times only during contact periods
    but allows for the total number of spikes in a contact period to be
    changed.
    :param sp: either a neo spike train, an array of spike times, or a boolean vector
                indicating when spikes occur
    :param use_flags: a neo epoch of contact times
    :return sp_shuf: a shuffled spiketrain of the same type as the input
    """
    if type(sp) == neo.core.spiketrain.SpikeTrain:
        cbool = epoch_to_bool(use_flags,sp.t_stop)
        spbool = np.zeros_like(cbool)
        spbool[sp.times.as_array().astype('int')] = 1
        spbool = np.logical_and(cbool,spbool)
    elif sp.dtype=='bool':
        cbool = epoch_to_bool(use_flags,len(sp))
        spbool = np.logical_and(sp,cbool)
    else:
        raise ValueError('Spike train is not the correct type.')
    possible_time = np.where(cbool)[0]
    n_spikes = np.sum(spbool)
    shuf_times = np.random.choice(possible_time,n_spikes,replace=False)
    shuf_times.sort()
    if type(sp) == neo.core.spiketrain.SpikeTrain:
        shuf_sp = neo.SpikeTrain(shuf_times,t_stop = sp.t_stop,units= sp.units)
    elif sp.dtype=='bool':
        shuf_sp = np.zeros_like(spbool)
        shuf_sp[shuf_times]=1
    else:
        raise Exception('This is one of those errors that makes you made at the person who wrote this. This shouldnt happen')

    return shuf_sp

