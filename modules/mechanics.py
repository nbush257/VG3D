import numpy as np
from scipy.io.matlab import loadmat,savemat
import quantities as pq
import seaborn as sns
import sklearn
import matplotlib.ticker as ticker
from sklearn.preprocessing import scale
from scipy.signal import savgol_filter
from neo_utils import *

''' this module needs cleanup'''
def get_analog_contact(var, cc):
    ''' this gets the mean and min-max of a given analog signal in each contact interval'''
    print('Minmax only works for zero-centered data')
    # MINMAX ONLY WORKS FOR ZERO CENTERED DATA, AND IT IS PRONE TO POINT NOISE!!!
    if type(var)==neo.core.analogsignal.AnalogSignal:
        var = np.array(var)
    mean_var = np.empty([cc.shape[0], var.shape[1]])
    minmax_var = np.empty([cc.shape[0], var.shape[1]])

    for ii, contact in enumerate(cc):
        var_slice = var[contact[0]:contact[1], :]
        mean_var[ii, :] = np.nanmean(var_slice, 0)
        first = var_slice[np.all(np.isfinite(var_slice),1)][0,:]
        minmax_idx = np.nanargmax(np.abs(var_slice-first), 0)
        minmax_var[ii, :] = var_slice[minmax_idx, np.arange(len(minmax_idx))]

    return mean_var,minmax_var


def iterate_filtvar(filtvars,cc):
    ''' this loops through every variable in a matlab structure (generally filtvars)
    and returns a dict of mean and minmax for each variable in each contact'''
    mean_filtvars = {}
    minmax_filtvars = {}

    for attr in filtvars._fieldnames:
        mean_filtvars[attr], minmax_filtvars[attr] = get_analog_contact(getattr(filtvars,attr),cc)
    return mean_filtvars,minmax_filtvars


def create_heatmap(var1,var2,bins,C,r):
    '''Need to figure out axes'''

    cmap = sns.cubehelix_palette(as_cmap=True)
    C = C.astype('bool').ravel()
    var1 = var1[C]
    var2 = var2[C]
    if type(r)!=np.ndarray:
        r = r.as_array()
    r = r.ravel()[C]
    H_prior,x_edges,y_edges = np.histogram2d(var1, var2, bins=bins)
    H_post = np.histogram2d(var1, var2, bins=bins, weights=r)[0]


    fig = sns.heatmap(H_post/H_prior,
                      vmin=0,
                      cmap=cmap,
                      robust=True,
                      cbar=True,
                      square=True,
                      xticklabels=20,
                      yticklabels=20)

    plt.colorbar()

def get_deriv(var,smooth=False):
    ''' returns the temporal derivative of a numpy array with time along the 0th axis'''
    if var.ndim==1:
        var = var[:,np.newaxis]
    if var.shape[1]>var.shape[0]:
        raise Warning('Matrix was wider than it is tall, are variables in the columns?')
    # assumes a matrix or vector where the columns are variables
    if smooth:
        var = savgol_filter(var,window_length=21)

    zero_pad = np.zeros([1, var.shape[1]], dtype='f8')
    var = np.concatenate([zero_pad, var], axis=0)

    return np.diff(var,axis=0)


def epoch_to_cc(epoch):
    ''' take a NEO epoch representing contacts and turn it into an Nx2 matrix which
    has contact onset in the first column and contact offset in the second.'''
    cc = np.empty([len(epoch),2])
    cc[:,0] = np.array(epoch.times).T
    cc[:,1] = cc[:,0]+np.array(epoch.durations).T

    print('cc is in {}'.format(epoch.units))
    return cc.astype('int64')


def categorize_deflections(blk):
    plot_tgl=1
    X = []
    last_contact = 0
    for seg in blk.segments:
        cc = epoch_to_cc(seg.epochs[0])
        dur = np.diff(cc,axis=1)

        TH_contact = get_analog_contact(seg.analogsignals[4],cc)[0]
        PHIE_contact = get_analog_contact(seg.analogsignals[3],cc)[0]
        Rcp_contact = get_analog_contact(seg.analogsignals[5],cc)[0]
        X.append(np.hstack((dur,TH_contact,PHIE_contact,Rcp_contact,cc[:,0].reshape([-1,1])+last_contact)))

        last_contact += cc[-1,1]

    X = np.concatenate(X,axis=0)
    X= sklearn.preprocessing.scale(X)
    clf = sklearn.cluster.SpectralClustering(n_clusters=48)
    labels = clf.fit_predict(X)

    if plot_tgl:
        ax=Axes3D(plt.figure())
        ax.scatter(X[:,4], X[:,2], X[:,3],c = labels,cmap='hsv')


def get_MB_MD(M):
    dat = M.magnitude
    MD = np.arctan2(dat[:, 2], dat[:, 1])
    MB = np.sqrt(dat[:, 1] ** 2 + dat[:, 2] ** 2)
    if type(M)==neo.core.analogsignal.AnalogSignal:
        MD = neo.core.AnalogSignal(MD, units=pq.radians, sampling_rate=pq.kHz)
        MB = neo.core.AnalogSignal(MB, units=pq.N*pq.m, sampling_rate=pq.kHz)


    return (MB, MD)


