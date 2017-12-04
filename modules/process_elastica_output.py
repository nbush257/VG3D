import numpy as np
import scipy.io.matlab as sio
import neoUtils
import scipy.signal

# TODO: employ different types of outlier detection to either (1) remove entire contacts or (2) smart remove points in the contact
# TODO: make sure the in-place operations are working correctly
# TODO: Port filtering from MATLAB code to python??

def cbool_to_cc(cbool):
    pad_cbool = np.concatenate([[[0]],cbool],axis=0)
    d = np.diff(pad_cbool,axis=0)
    starts = np.where(d==1)[0]
    stops = np.where(d==-1)[0]

    return np.array([[x,y] for x,y in zip(starts,stops)])


def median_contact(var,cc):
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


def get_contact_energy(var,cc):
    E = []
    for start,stop in cc:
        slice = var[start:stop,:]
        t1 = slice**2
        t2 = np.concatenate([slice[1:],[[0]]])*np.concatenate([[[0]],slice[:-1]])
        E.append(np.sum((t1[1:-2]-t2[1:-2])**2))
    return(np.array(E))


def get_feature_matrix(var, cc, window_size=7):

    d_abs = get_d(var)
    var_contact_centered = subtract_median(var,cc)
    d_var_contact_centered = get_d(var_contact_centered)


    var_nan_zero = var.copy()
    var_nan_zero[np.isnan(var_nan_zero)]=0
    running_median =scipy.signal.medfilt(var_nan_zero.T,window_size)

