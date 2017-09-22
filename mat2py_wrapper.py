import numpy as np
from scipy.io.matlab import loadmat, savemat
from neo.core import SpikeTrain
from quantities import ms, s
import neo
import quantities as pq
import elephant
import sys
from neo.io import NeoMatlabIO as NIO
import glob
import os
import re


def convertC(C):
    ''' converts a boolean contact vector into an Nx2 array of contact onsets and offsets.
    '''
    C = np.squeeze(C)
    if C.ndim > 1:
        raise ValueError('C should be a 1-D vector')

    C[np.isnan(C)] = 0
    C[C < 0.95] = 0
    C = C.astype('int32')

    # stack a zero on either end in case the first or last frame is contact
    d = np.diff(np.hstack(([0], C, [0])), n=1, axis=0)

    cstarts = np.where(d == 1)[0]
    cends = np.where(d == -1)[0]
    # return the Nx2 Matrix of onsets (inclusive) and offsets(exclusive)
    return np.vstack((cstarts, cends)).T


def createSeg(fname):
    # load the data
    dat = loadmat(fname)

    # access the analog data
    vars = dat['vars'][0, 0]
    filtvars = dat['filtvars'][0, 0]
    rawvars = dat['rawvars'][0, 0]
    PT = dat['PT'][0, 0]
    C = dat['C']
    cc = convertC(C)
    num_contacts = cc.shape[0]

    # access the neural data
    sp = dat['sp'][0]
    spikes = dat['spikes'][0]

    # initialize the segment
    seg = neo.core.Segment(file_origin=fname)

    seg.annotate(
        ratnum=PT['ratnum'][0],
        whisker=PT['whisker'][0],
        trial=PT['trial'][0],
        id=PT['id'][0],
        frames=PT['Frames'][0],
        TAG=PT['TAG'][0],
        s=PT['s'][0],
        rbase=PT['E3D_rbase'][0],
        rtip=PT['E3D_rtip'][0]
    )

    for varname in filtvars.dtype.names:
        # get the metadata for the signal
        sig = filtvars[varname]
        if varname == 'M':
            U = 'N*m'
            name = 'Moment'
        elif varname == 'F':
            U = 'N'
            name = 'Force'
        elif varname == 'Rcp':
            U = 'm'
            name = varname
        else:
            U = 'deg'
            name = varname

        # append the signal to the segment
        seg.analogsignals.append(
            neo.core.AnalogSignal(
                sig, units=U, sampling_rate=pq.kHz, name=name
            )
        )
    # add the spike trains to the segment



    for times, waveshapes in zip(sp, spikes):
        waveshapes = np.expand_dims(waveshapes.T, 1)
        idx = times * ms <= seg.analogsignals[0].t_stop
        times = times[idx]
        waveshapes = waveshapes[idx.ravel(), :, :]

        train = SpikeTrain(times * ms, t_stop=seg.analogsignals[0].t_stop, units=ms, waveforms=waveshapes * pq.mV)
        seg.spiketrains.append(train)

    seg.epochs.append(
        neo.core.Epoch(
            times=cc[:, 0] * ms,
            durations=np.diff(cc, axis=1) * ms,
            labels=None,
            name='contacts')
    )

    return seg


def batch_convert(d_list, p):
    d_list = list(d_list)
    for root in d_list:
        try:
            root_full = os.path.join(p, root)
            fname_NEO = root_full + '_NEO.mat'
            fid = NIO(fname_NEO)
            files = glob.glob(root_full + '*1K.mat')
            blk = neo.core.Block()
            for filename in files:
                print(filename)
                seg = createSeg(filename)
                if os.path.isfile(fname_NEO):
                    blk = fid.read_block()
                blk.segments.append(seg)
                fid.write_block(blk)
        except:
            print('problem with {}'.format(root))


def get_list(p, fname_spec):
    d_list = glob.glob(os.path.join(p, fname_spec))
    d_list = [os.path.split(f)[1] for f in d_list]
    d_list = [re.search('^rat\d{4}_\d{2}_[A-Z]{3}\d\d_VG_[A-Z]\d', f).group() for f in d_list]
    d_list = set(d_list)
    return (d_list)


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


if __name__ == '__main__':
    p = r'C:\Users\nbush257\Box Sync\__VG3D\_E3D_1K\deflection_trials'
    fname_spec = '*NEO.mat'
    d_list = get_list(p, fname_spec)
    batch_convert(d_list, p)


