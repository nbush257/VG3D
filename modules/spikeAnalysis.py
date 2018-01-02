from elephant.statistics import *
import neo
import elephant
from elephant import kernels
from elephant.conversion import *
import quantities as pq
import neoUtils


def get_autocorr(sp_neo):
    '''
    Autocorrelate a spiketrain
    :param sp_neo: a neo spike train
    :return: the autocorrelation signal
    '''
    return(elephant.spike_train_correlation.cross_correlation_histogram(BinnedSpikeTrain(sp_neo, binsize=pq.ms), BinnedSpikeTrain(sp_neo, binsize=pq.ms)))


def correlate_to_stim(sp_neo,var,kernel_sigmas,mode='g',plot_tgl=False):
    '''
    Calculate the correlation between a spike train and an analog signal.
    Smooths the observed spike train with a user defined kernel in order to get a continuous rate estimate

    :param sp_neo:          a neo spiketrain
    :param var:             a 1D numpy array, neo analog signal, or quantity to correlate the spike rate with
    :param kernel_sigmas:   list or numpy array of sigma values defining the kernel to smooth the spike train
    :param mode:            type of kernel to smooth the spike train with ['g': gaussian,'b'/'r': box or rectangular]

    :return corr_:          A numpy array of correlation values between the spiketrain and desired variable.
                                Each entry corresponds to a different smoothing parameter as indicated in 'kernel_sigmas'
    :return kernel_sigmas:  The numpy array of kernel sigma values used
    '''

    # map var to a numpy array if needed
    if type(var)==neo.core.AnalogSignal or type(var)==quantities.quantity.Quantity:
        var = var.magnitude

    # init correlation output
    corr_ = np.empty(kernel_sigmas.shape[0])

    # loop over all sigma values
    for ii,sigma in enumerate(kernel_sigmas):
        # get the appropriate kernel with corresponding sigma
        if mode=='b' or mode=='r':
            kernel = kernels.RectangularKernel(sigma=sigma * pq.ms)
        elif mode=='g':
            kernel = kernels.GaussianKernel(sigma=sigma * pq.ms)
        else:
            raise ValueError('Kernel mode not defined')

        r = instantaneous_rate(sp_neo, sampling_period=pq.ms, kernel=kernel)
        corr_[ii] = np.corrcoef(r.squeeze(), var)[0,1]

    if plot_tgl:
        plt.plot(kernel_sigmas,corr_,'k')

    return(corr_,kernel_sigmas)


def get_contact_sliced_trains(blk,unit,pre=0.,post=0.):
    '''
    returns mean_fr,ISIs,spiketrains for each contact interval for a given unit
    May want to refactor to take a unit name rather than index? N
    Need a block input to get the contact epochs
    pre is the number of milliseconds prior to contact onset to grab: particularly useful for PSTH
    post is the number of milliseconds after contact to grab

    :param blk:     a neo block of the data to slice
    :param unit:    a neo unit which carries the desired neuron's spiketrains
    :param pre:     the time prior to the contact onset estimate to include in the contact slice
    :param post:    the time after the contact offset estimate to include in the contact slice

    :return FR:             A numpy array of average firing rates where each entry is the FR for a contact interval
    :return ISI:            A list of numpy arrays where each item in the list is the vector of ISIs for a given contact interval
    :return contact_trains: A list of neo spiketrains where each entry in the list is the spike train for a given contact interval
    '''
    if type(pre)!=pq.quantity.Quantity:
        pre *= pq.ms
    if type(post) != pq.quantity.Quantity:
        post *= pq.ms

    # init units
    ISI_units = pq.ms
    FR_units = 1/pq.s

    # init outputs
    ISI = []
    contact_trains = []
    FR = np.array([]).reshape(0, 0)*1/pq.s

    # loop over each segment
    for train,seg in zip(unit.spiketrains,blk.segments):
        epoch = seg.epochs[-1]
        # initialize the mean firing rates to zero
        seg_fr = np.zeros([len(epoch), 1], dtype='f8')*FR_units
        # loop over each contact epoch
        for ii,(start,dur) in enumerate(zip(epoch.times,epoch.durations)):
            # grab the spiketrain from contact onset to offset, with a pad
            train_slice = train.time_slice(start-pre, start + dur+post)

            # need one spike to calculate FR
            if len(train_slice)>0:
                seg_fr[ii] = mean_firing_rate(train_slice)

            # need 3 spikes to get isi's
            if len(train_slice)>2:
                intervals = isi(np.array(train_slice)*ISI_units)
            else:
                intervals = [np.nan]*ISI_units
            # add to lists
            contact_trains.append(train_slice)
            ISI.append(intervals)

        FR = np.append(FR, seg_fr) * FR_units

    return FR,ISI,contact_trains


def get_binary_trains(trains,norm_length=True):
    '''
    takes a list of spike trains and computes binary spike trains for all
    can return a matrix with the number of columns equal to the longest train.

    Might be useful to normalize based on the number of observations of each time point

    :param trains:          a list of neo spiketrains
    :param norm_length:     boolean indicating whether to normalize the length of the output
                                If True: return a numpy array of fixed size where each row is a contact and each column is a ms.
                                    Fills the shorter contacts with zeros
                                If False: return a list of boolean vectors indicate spike occurrence for each contact and a list of the durations

    :return:                Binary spike trains for each contact
    '''


    b = []
    if norm_length:
        # init output matrix by normalizing to longest contact
        durations = np.zeros(len(trains),dtype='int64')
        for ii,train in enumerate(trains):
            duration = train.t_stop - train.t_start
            duration.units = pq.ms
            durations[ii] = int(duration)
        max_duration = np.max(durations)+1
        b = np.zeros([max_duration,len(trains)],dtype='bool')

    # loop over each train and convert to a boolean vector
    for ii,train in enumerate(trains):
        # if there are no spikes, return a vector of all zeros. This is required because binarize errors on empty spiketrains
        if len(train) == 0:
            duration = train.t_stop-train.t_start
            duration.units=pq.ms
            duration = int(duration)
            if norm_length:
                b[:duration,ii] = np.zeros(duration,dtype='bool')
            else:
                b.append(np.zeros(duration,dtype='bool'))
        # calculate the binary spike train
        else:
            if norm_length:
                b_temp = elephant.conversion.binarize(train, sampling_rate=pq.kHz)
                b[:len(b_temp),ii]=b_temp
            else:
                b.append(elephant.conversion.binarize(train,sampling_rate=pq.kHz))


    if norm_length:
        # return the durations if the length is kept consistent across all
        return b,durations
    else:
        return b


def get_CV_LV(ISI):
    '''
    Given a list of ISIs, get the Coefficient of variation
    and the local variation of each spike train. Generally a list of ISIs during contact epochs
    :param ISI:         List of ISIs
    :return CV_array:   Array of CV values for each contact interval
    :return LV_array:   Array of LV values for each contact interval
    '''
    CV_array = np.array([])
    LV_array = np.array([])
    for interval in ISI:
        if np.all(np.isfinite(interval)):
            CV_array = np.concatenate([CV_array, [elephant.statistics.cv(interval)]])
            LV_array = np.concatenate([LV_array, [elephant.statistics.lv(interval)]])

    return CV_array,LV_array


def get_PSTH(blk,unit):
    '''
    plots a PSTH aligned to contact for a given unit

    :param blk: a neo block
    :param unit: a neo unit
    :return: ax
    '''
    if True:
        raise Exception('TODO: this function is probably depriacted. It does not allow for pre-contact times to be considered')

    FR, ISI, contact_trains = get_contact_sliced_trains(blk)
    b, durations = get_binary_trains(contact_trains[unit.name])
    b_times = np.where(b)[1] * pq.ms  # interval.units
    PSTH, t_edges = np.histogram(b_times, bins=np.arange(0, np.max(durations), float(binsize)))
    ax = plt.bar(t_edges[:-1],
            PSTH.astype('f8') / len(durations) / binsize * 1000,
            width=float(binsize),
            align='edge',
            alpha=0.8
            )
    return ax


def get_raster(blk,unit):
    '''
    Plot a raster time locked to the onset of contact

    :param blk: a neo block
    :param unit: a neo unit
    :return: None
    '''
    if True:
        raise Exception('TODO: This is probably deprecated. it does not allow for time pre contact')

    count = 0
    pad = 0.3
    f=plt.figure()
    ax = plt.gca()
    for train,seg in zip(unit.spiketrains,blk.segments):
        epoch = seg.epochs[0]
        for start,duration in zip(epoch,epoch.durations):
            stop = start+duration
            t_sub = train.time_slice(start,stop).as_quantity()-start
            ax.vlines(t_sub,count-pad,count+pad)
            count+=1
    ax.set_ylim(0,count)

def get_STC(signal,train,window):
    '''
    Compute the Spike Triggered Covariance for a set of signals.
    TODO: this function was never finished, and I am coding up the math, so it needs to be very well tested
    :param signal:
    :param train:
    :param window:
    :return:
    '''
    if True:
        raise Exception('TODO: this function was never finished, and I am coding up the math, so it needs to be very well tested')

    X = np.empty([len(train),int(window.magnitude)*2,signal.shape[-1]])
    for ii,spike in enumerate(train):
        X[ii,:,:]=(signal.time_slice(spike-window,spike+window))

    STA = np.nanmean(X, axis=0)
    X_centered = X-STA
    STC = np.empty([X.shape[1],X.shape[2]])
    STC[:] = np.nan
    for ii in xrange(signal.shape[-1]):
        var_X = X_centered[:,:,ii]
        STC[:,ii] = np.nanmean(np.dot(var_X.T,var_X),axis=0)

    C = (float(1)/(X.shape[-1]-1))*np.dot(signal.as_array().T,signal.as_array())

    return STA

def binary_to_neo_train(y):
    return(neo.SpikeTrain(np.where(y)[0],t_stop=len(y),units=pq.ms))


def get_onset_contacts(blk,unit_num=0,num_spikes=1,varname='M'):
    '''
    Finds contacts which ellicited the desired number of spikes. Useful if trying to look at RA onset.
    :param blk:
    :param unit_num:
    :param num_spikes:
    :param varname:
    :return: var_sliced, c_idx (an index of which contacts have the desired numer of spikes)
    '''
    use_flag = neoUtils.concatenate_epochs(blk,-1)
    unit = blk.channel_indexes[-1].units[unit_num]
    trains = get_contact_sliced_trains(blk,unit)[-1]
    c_idx=[]
    for ii,train in enumerate(trains):
        if len(train)==num_spikes:
            c_idx.append(ii)
    var = neoUtils.get_var(blk,varname)
    var_sliced = neoUtils.get_analog_contact_slices(var, use_flag)
    return(var_sliced[:,c_idx,:],c_idx)

>>>>>>> 3b11c5f4af99f36a81ddccb8ca00e3e426f3bf60





