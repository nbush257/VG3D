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




