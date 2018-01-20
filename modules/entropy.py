import pyentropy as pye
import neoUtils
import spikeAnalysis
import worldGeometry
import varTuning
import quantities as pq
import numpy as np
import elephant


def calc_ent(var,sp,use_flags,nbins=100):

    b = elephant.conversion.binarize(sp,sampling_rate=pq.kHz)[:-1]
    idx = np.logical_and(use_flags.ravel(), np.isfinite(var).ravel())
    X = pye.quantise(var[idx],nbins,uniform='bins')[0].ravel()
    Y = b[idx].astype('int').ravel()

    S = pye.DiscreteSystem(X,(1,nbins),Y,(1,2))
    S.calculate_entropies(calc=['HX','HY','HXY','SiHXi','HiX','HshX','HiXY','HshXY','ChiX','ChiXY1','HXY1'])
    return(S.H['HX'],S.H['HXY'],S.I())


def shuf_sp(spbool):
    sp_out = np.zeros_like(spbool)
    num_spikes = np.sum(spbool)
    T = len(spbool)
    idx = np.random.choice(np.arange(0, T), num_spikes, replace=False)
    sp_out[idx] = True
    return(sp_out)

#
# HX=[]
# HXY=[]
# I=[]
# for ii in np.logspace(1,4,20):
#     print(ii)
#     var =F[:,0].magnitude
#     ENT = calc_ent(var,sp,use_flags,nbins=int(ii))
#     HX.append(ENT[0])
#     HXY.append(ENT[1])
#     I.append(ENT[2])
# plt.plot(np.logspace(1,4,20),I)