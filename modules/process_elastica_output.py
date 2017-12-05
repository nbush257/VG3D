import numpy as np
import scipy.signal
from sklearn import preprocessing,neighbors,ensemble,covariance
from scipy import signal,interpolate

import pandas as pd

import matplotlib.pyplot as plt

# TODO: employ different types of outlier detection to either (1) remove entire contacts or (2) smart remove points in the contact
# TODO: make sure the in-place operations are working correctly
# TODO: Port filtering from MATLAB code to python??

def cbool_to_cc(cbool):
    pad_cbool = np.concatenate([[[0]],cbool],axis=0)
    d = np.diff(pad_cbool,axis=0)
    starts = np.where(d==1)[0]
    stops = np.where(d==-1)[0]

    return np.array([[x,y] for x,y in zip(starts,stops)])


def median_by_contact(var,cc):
    y = []
    for start,stop in cc:
        slice = var[start:stop,:]
        y.append(np.nanmedian(slice,axis=0))
    return np.array(y)


def subtract_median(var,cc):
    var_out = var.copy()
    for start, stop in cc:
        slice = var_out[start:stop, :]
        slice -= np.nanmedian(slice, axis=0)
    return(var_out)


def get_d(var):
    if var.ndim>1:
        n_cols = var.shape[-1]
    else:
        var = var[:,np.newaxis]
        n_cols = 1
    zero_pad = np.zeros([1,n_cols])

    d = np.diff(var,axis=0)

    return np.concatenate([zero_pad,d],axis=0)

def E_operator(x):
    if x.shape[1]>1:
        raise Exception('Energy operator cannot accept more than one column')
    E = np.zeros_like(x)
    t1 = x ** 2
    t2 = np.concatenate([x[1:], [[0]]]) * np.concatenate([[[0]], x[:-1]])
    E[1:-2] = t1[1:-2]-t2[1:-2]
    return E

def energy_by_contact(var,cc):
    E_out = []
    for start,stop in cc:
        slice = var[start:stop,:]
        E = E_operator(slice)
        E_out.append(np.sum(E**2))
    return(np.array(E_out))

def impute_snippet(snippet):
    x = np.arange(len(snippet))
    y = snippet.ravel()
    f = interpolate.interp1d(x[np.isfinite(y)],y[np.isfinite(y)],kind='cubic')
    y[np.isnan(y)] = f(x[np.isnan(y)])


def fill_nan_gaps(var, cbool, thresh=10):
    use_flag = cbool.copy()
    var_out = var.copy()

    if var_out.shape[-1]>1:
        raise Exception('var must be single column')
    cc = cbool_to_cc(use_flag)

    for start,stop in cc:
        snippet = var_out[start:stop]
        if np.all(np.isnan(snippet)):
            use_flag[start:stop]=0
            continue

        gaps = cbool_to_cc(np.isnan(snippet))
        if np.any(np.diff(gaps)>thresh):
            use_flag[start:stop]=0
        else:
            impute_snippet(snippet)
    return var_out, use_flag


def remove_bad_contacts(var_in,cbool):
    var = var_in.copy()

    cc = cbool_to_cc(cbool)
    # var = scale_by_contact(var, cc)

    d = get_d(var)
    d[np.isnan(d)]=0
    E = energy_by_contact(var,cc)
    E2 = []
    # bad_idx = E>np.nanmedian(E)*50
    for start,stop in cc:
        E2.append(np.sum(d[start:stop] ** 2))

    E2 = np.asarray(E2)

    bad_idx = E2>np.median(E2)*100
    for ii,(start,stop) in enumerate(cc):
        if bad_idx[ii]:
            cbool[start:stop]=0


def get_feature_matrix(var, cc, window_size=7):

    d_abs = get_d(var)
    var_contact_centered = subtract_median(var,cc)
    d_var_contact_centered = get_d(var_contact_centered)


    var_nan_zero = var.copy()
    var_nan_zero[np.isnan(var_nan_zero)]=0
    running_median =scipy.signal.medfilt(var_nan_zero.T,window_size)


def scale_by_contact(var,cc):
    var_out = var.copy()
    for start,stop in cc:
        slice = var_out[start:stop,:]

        scaler = preprocessing.RobustScaler()
        idx = np.isfinite(slice).ravel()
        if np.any(idx):
            slice[idx] = scaler.fit_transform(slice[idx])
    return var_out



if __name__=='__main__':
    fid = np.load(r'C:\Users\guru\Desktop\temp\M_and_C_test.npz')
    M = fid['M']
    C = fid['C']

    M2,use_flags = fill_nan_gaps(M,C) # either impute small nan gaps or remove the entire contact from consideration. Allows us to make all non-flaggd M= 0
    M2[use_flags == 0] = 0
    M_filt = signal.medfilt(M2,kernel_size=[3,1])
    remove_bad_contacts(M_filt,use_flags)

    cc = cbool_to_cc(use_flags)
    M_scale = scale_by_contact(M2,cc)
    Md = get_d(M2)
    Md_scale = scale_by_contact(Md,cc)

    X = np.concatenate([M_scale,Md_scale],axis=1)

    y = np.zeros(X.shape[0],dtype='int')
    # clf = neighbors.LocalOutlierFactor(contamination=0.05)
    clf = covariance.EllipticEnvelope(contamination=.001)
    # clf = ensemble.IsolationForest()
    idx = np.squeeze(use_flags==1)
    clf.fit(X[idx,:])
    y[idx] = clf.predict(X[idx,:])

    y[y==1]=0
    y[y==-1]=1

    use_flags[y==1]=0
    M_out = M.copy()
    M_out[use_flags==0] = np.nan

    cc_use = cbool_to_cc(use_flags)
    for start,stop in cc_use:
        if stop-start>10:
            impute_snippet(M_out[start:stop])

    M_out[C==0]=0