import pyentropy as pye
import neoUtils
import spikeAnalysis
import worldGeometry
import varTuning


def hold(var,sp,use_flags):
    if True:
        raise Exception
    rate,edges = varTuning.stim_response_hist(var,sp,use_flags,200)
    edges = edges[:-1]
    idx = np.isfinite(rate)
    rate = rate[idx]*1000
    edges = edges[idx]


    max_rate = np.max(np.ceil(rate)).astype('int')

    rate = pye.quantise(rate,max_rate,uniform='bins')[0]
    edges = pye.quantise(edges,len(edges),uniform='bins')[0]
    S = pyentropy.DiscreteSystem(rate,(1,max_rate),edges,(1,len(edges)))
    S.calculate_entropies(calc=['HX', 'HY', 'HXY', 'SiHXi', 'HiX', 'HshX', 'HiXY', 'HshXY', 'ChiX'])