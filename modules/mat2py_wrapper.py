import numpy as np
from scipy.io.matlab import loadmat, savemat
from neo.core import SpikeTrain
from quantities import ms, s
import neo
import quantities as pq
import elephant
import sys
import neo.io
from neo_utils import add_channel_indexes
import glob
import os
import re


def convertC(C):
    ''' converts a boolean contact vector into an Nx2 array of contact onsets and offsets.
    '''
    if type(C) == neo.core.epoch.Epoch:
        C_out = np.array(len)
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
    trial_idx = int(PT['trial'][0][1:])
    # access the neural data
    sp = dat['sp'][0]
    spikes = dat['spikes'][0]

    # initialize the segment
    seg = neo.core.Segment(str(PT['id'][0]),file_origin=fname,index=trial_idx)

    seg.annotate(
        nix_name    =   str(PT['id'][0]),
        ratnum      =   str(PT['ratnum'][0]),
        whisker     =   str(PT['whisker'][0]),
        trial       =   str(PT['trial'][0]),
        id          =   str(PT['id'][0]),
        frames      =   str(PT['Frames'][0]),
        TAG         =   str(PT['TAG'][0]),
        s           =   str(PT['s'][0]),
        rbase       =   str(PT['E3D_rbase'][0]),
        rtip        =   str(PT['E3D_rtip'][0]),
        trial_type  =   'deflection'
    )

    for varname in filtvars.dtype.names:
        # get the metadata for the signal
        sig = filtvars[varname]
        if varname == 'M':
            U = pq.N*pq.m
            name = 'Moment'
        elif varname == 'F':
            U = pq.N
            name = 'Force'
        elif varname == 'Rcp':
            U = pq.m
            name = varname
        else:
            U = pq.rad
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
            fname_M = root_full + '_NEO.mat'
            fname_N = root_full + '_NEO.h5'

            fid_M = neo.io.NeoMatlabIO(fname_M)

            files = glob.glob(root_full + '*1K.mat')

            for filename in files:
                fid_N = neo.io.NixIO(fname_N)
                print(filename)
                seg = createSeg(filename)
                if len(fid_N.read_all_blocks())!=0:
                    blk = fid_N.read_block()
                else:
                    blk = neo.core.Block(name=seg.annotations['id'][:-3])
                    blk.annotate(
                        ratnum=seg.annotations['ratnum'],
                        whisker=seg.annotations['whisker'],
                        id=seg.annotations['id'],
                        s=seg.annotations['s'],
                        rbase=seg.annotations['rbase'],
                        rtip=seg.annotations['rtip'],
                        trial_type=seg.annotations['trial_type'],
                        date=re.search('(?<=_)[A-Z]{3}\d\d(?=_)',seg.annotations['TAG']).group()
                    )

                blk.segments.append(seg)

                fid_N.write_block(blk)

                fid_N.close()
                # create channel indexes

            # create chx
            fid_N = neo.io.NixIO(fname_N)
            blk = fid_N.read_block()
            add_channel_indexes(blk)

            # write NIX
            fid_N.write_block(blk)
            fid_N.close()

            #write matlab
            fid_M.write_block(blk)
        except:
            print('problem with {}'.format(root))



def get_list(p, fname_spec):
    d_list = glob.glob(os.path.join(p, fname_spec))
    d_list = [os.path.split(f)[1] for f in d_list]
    d_list = [re.search('^rat\d{4}_\d{2}_[A-Z]{3}\d\d_VG_[A-Z]\d', f).group() for f in d_list]
    d_list = set(d_list)
    return (d_list)



if __name__ == '__main__':
    p = r'K:\VG3D\_E3D_PROC\_vibration_trials'
    fname_spec = '*1k.mat'
    d_list = get_list(p, fname_spec)
    batch_convert(d_list, p)



