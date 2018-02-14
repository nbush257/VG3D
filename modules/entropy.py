import pyentropy as pye
import neoUtils
import spikeAnalysis
import worldGeometry
import varTuning
import quantities as pq
import numpy as np
import elephant


def calc_ent(var,sp,use_flags,nbins=100,sigma=5*pq.ms):

    b = elephant.conversion.binarize(sp,sampling_rate=pq.kHz)[:-1]
    K = elephant.kernels.GaussianKernel(sigma=sigma)
    r = elephant.statistics.instantaneous_rate(sp,sampling_period=pq.ms,kernel=K).magnitude
    idx = np.logical_and(use_flags.ravel(), np.isfinite(var).ravel())
    X = pye.quantise(var[idx],nbins,uniform='bins')[0].ravel()
    # Y = b[idx].astype('int').ravel()
    Y = r[idx].astype('int').ravel()

    S = pye.DiscreteSystem(X,(1,nbins),Y,(1,np.max(Y)+1))
    S.calculate_entropies(calc=['HX','HY','HXY','SiHXi','HiX','HshX','HiXY','HshXY','ChiX'])
    return(S.H['HX'],S.H['HXY'],S.I(),S)


HX=[]
HXY=[]
I=[]
for ii in np.logspace(1,4,20):
    print(ii)
    var =F[:,0].magnitude
    ENT = calc_ent(var,sp,use_flags,nbins=int(ii))
    HX.append(ENT[0])
    HXY.append(ENT[1])
    I.append(ENT[2])
plt.plot(np.logspace(1,4,20),I)

HX=[]
HXY=[]
I=[]
HY =[]
for ii in np.linspace(2,100,20):
    print(ii)
    var =F[:,1].magnitude
    ENT = calc_ent(var,sp,use_flags,nbins=1000,sigma=ii*pq.ms)
    HX.append(ENT[0])
    HXY.append(ENT[1])
    I.append(ENT[2])
    HY.append(ENT[3].H['HY'])
plt.plot(np.linspace(2,100,20),I,'.-')