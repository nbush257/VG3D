import numpy as np
from scipy.io.matlab import loadmat,savemat
from neo.core import SpikeTrain
from quantities import ms,s
import neo
import quantities as pq
import elephant
import sys
from neo.io.nixio import NixIO



def convertC(C):
    ''' converts a boolean contact vector into an Nx2 array of contact onsets and offsets.
    '''
    C = np.squeeze(C)
    if C.ndim>1:
        raise ValueError('C should be a 1-D vector')

    C[np.isnan(C)] = 0
    C[C<0.95]=0
    C = C.astype('int32')

    # stack a zero on either end in case the first or last frame is contact
    d = np.diff(np.hstack(([0],C,[0])),n=1,axis=0)

    cstarts = np.where(d==1)[0]
    cends = np.where(d==-1)[0]
    # return the Nx2 Matrix of onsets (inclusive) and offsets(exclusive)
    return np.vstack((cstarts,cends)).T


def createSeg(fname,blk=None):
    # load the data
    dat = loadmat(fname)

    # access the analog data
    vars = dat['vars'][0,0]
    filtvars = dat['filtvars'][0,0]
    rawvars = dat['rawvars'][0,0]
    C = dat['C']
    cc = convertC(C)
    num_contacts = cc.shape[0]

    # access the neural data
    sp = dat['sp'][0]

    #initialize the segment
    seg = neo.core.Segment(file_origin=fname)

    for varname in filtvars.dtype.names:
        # get the metadata for the signal
        sig = filtvars[varname]
        if varname=='M':
            U = 'N*m'
            name = 'Moment'
        elif varname=='F':
            U = 'N'
            name='Force'
        else:
            U = 'rad'
            name = varname

        # append the signal to the segment
        seg.analogsignals.append(
            neo.core.AnalogSignal(
                sig,units=U,sampling_rate=pq.kHz,name=name
            )
        )
    # add the spike trains to the segment
    for cell in sp:
        train = SpikeTrain(cell*ms,t_stop=seg.analogsignals[0].t_stop,units=s)
        seg.spiketrains.append(train)

    spbool = elephant.conversion.binarize(seg.spiketrains[0],sampling_rate=pq.kHz)
    seg.analogsignals.append(neo.core.AnalogSignal(spbool,ms,t_stop=seg.analogsignals[0].t_stop,sampling_rate=pq.kHz),name='Binarized Spikes')

    seg.epochs.append(
        neo.core.Epoch(
            times=cc[:, 0] * ms,
            durations=np.diff(cc, axis=1) * ms,
            labels=None,
            name='contacts')
    )

    if blk!=None:
        blk = neo.core.Block()

    blk.segments.append(seg)
    return blk


def main():
    blk = None
    fname = sys.argv[1]
    if length(sys.argv) == 3:
        f_blk = nixio(sys.argv[2],'r')
    blk = createSeg(fname, blk)

    blk = createSeg(fname)
if __name__=='__main__':
    main()


