import numpy as np
import scipy.signal
from sklearn import preprocessing,neighbors,ensemble,covariance
from scipy import signal,interpolate

import pandas as pd

import matplotlib.pyplot as plt


# TODO: make sure the in-place operations are working correctly
# TODO: Port filtering from MATLAB code to python??

def cbool_to_cc(cbool):
    pad_cbool = np.concatenate([[[0]],cbool],axis=0)
    d = np.diff(pad_cbool,axis=0)
    starts = np.where(d==1)[0]
    stops = np.where(d==-1)[0]

    return np.array([[x,y] for x,y in zip(starts,stops)])


def median_by_contact(var,cc):
    '''
    get the median value of each contact segment 
    --Currently not being used
    
    :param var:     variable to get the median of 
    :param cc:      contact boundaries
    :return y:      an array of the median values of each contact segment 
    '''
    y = []
    for start,stop in cc:
        snippet = var[start:stop,:]
        y.append(np.nanmedian(snippet,axis=0))

    return np.array(y)



def get_d(var):
    '''
    Perform diff on a matrix along the rows. That is, assume a matrix is [n_obs x n_features]
    Maintains the input shape by padding the diff output with a zero row
    
    :param var: Matrix of values to diff along the rows 
    :return d:  The diffed matrix of the same size as var 
    '''

    if var.ndim>1:
        n_cols = var.shape[-1]
    else:
        var = var[:,np.newaxis]
        n_cols = 1

    zero_pad = np.zeros([1,n_cols])
    d = np.diff(var,axis=0)

    return np.concatenate([zero_pad,d],axis=0)

def E_operator(x):
    '''
    Performs the discrete energy calculation on a signal. Not used in currently, but may be useful at some point.
    
    :param x:   a column vector of the signal to calculate energy on. This should probably be expanded 
    :return E:  The energy vector    
    '''
    if True:
        raise Exception('This function ought to be expanded to accept more general inputs. Leaving as a reference. NEB 20171205')


    if x.shape[1]>1:
        raise Exception('Energy operator cannot accept more than one column')

    E = np.zeros_like(x)
    t1 = x ** 2
    t2 = np.concatenate([x[1:], [[0]]]) * np.concatenate([[[0]], x[:-1]])
    E[1:-2] = t1[1:-2]-t2[1:-2]
    return E

def energy_by_contact(var,cc):
    '''
    Applies the energy operator to single contact episodes. NOT CURRENTLY USED
    
    :param var:     Variable to apply energy operator to 
    :param cc:      An [N x 2] array of contact starts and stops (in samples)
    
    :return E_out:  An array (length = # of contacts) of the sum(E**2) for all the samples in each contact episode

    '''
    E_out = []
    for start,stop in cc:
        slice = var[start:stop,:]
        E = E_operator(slice)
        E_out.append(np.sum(E**2))
    return(np.array(E_out))

def impute_snippet(snippet,kind = 'cubic'):
    '''
    Helper function to impute over nan gaps in a given snippet. Performs operation in place 
    Be careful--this will perform on all inputs, so be sure you have excluded segments you don't want to impute over by this point 
    
    :param snippet: A row or column vector of the signal to be imputed. 
    :param kind:    Passes to scipy.interpolate.interp1d kind
    
    :return None:   Performs in-place 
    '''

    if snippet.ndim>1:
        if snippet.shape[1]>1:
            raise ValueError('Snippet must be a row vector or a column vector')

    x = np.arange(len(snippet))
    y = snippet.ravel()

    f = interpolate.interp1d(x[np.isfinite(y)],y[np.isfinite(y)],kind='cubic')
    y[np.isnan(y)] = f(x[np.isnan(y)])



def fill_nan_gaps(var, cbool, thresh=10):
    '''
    Important function that fills small NaN gaps in a signal. 
    Discards the entire contact interval if the number of consecutive nans is too many (>thresh).
    Allows you to set all NaN values to zero later when you are looking for outliers. That is, you avoid setting
     values to zero in the middle of contact episodes. If the gap is so large, we must omit the contact.
    
    :param var:     Signal over which to impute or discard entire intervals. Must be a column vector or row vector 
    :param cbool:   Contact boolean
    :param thresh:  Maximum number of consecutive NaNs to impute over. If a NaN gap is found larger than thresh, the contact episode is flagged for non-use
    
    :return var_out:    The imputed version of the variable passed in
    :return use_flag:   Indicates whether a frame is to be used (1) or not (0). Inherits from cbool
     
    '''
    # copy inputs to avoid overwriting
    use_flag = cbool.copy()
    var_out = var.copy()
    if var_out.ndim>1:
        if var_out.shape[1]>1:
            raise ValueError('var must be single column')

    cc = cbool_to_cc(use_flag)


    for start,stop in cc:
        snippet = var_out[start:stop]

        # Remove a contact episode if it is all NaNs
        if np.all(np.isnan(snippet)):
            use_flag[start:stop]=0
            continue

        # calculate the length of the NaN gaps in this segment
        # TODO: Boundary conditions on if the NaN gap is underthreshold but abutts the border of contact?
        gaps = cbool_to_cc(np.isnan(snippet))
        if np.any(np.diff(gaps)>thresh):
            use_flag[start:stop]=0
        else:
            impute_snippet(snippet)
    return var_out, use_flag


def remove_bad_contacts(var_in,cbool,thresh=100):
    """
    Uses the derivative of a signal to determine if a given contact has a lot of erroneous tracking. 
    If it does, we remove to contact episode by setting cbool for that segment equal to 0 
    
    :param var_in:  The signal to evaluate for large changes -- this signal is not altered  
    :param cbool:   Vector indicating contact. Operated on inplace to omit contact segments where large errors occur
    :param thresh:  Sets the sensitivity of the removal of entire contacts. Larger allows more conatacts to be untouched. 
                        (This is what we multiply the median by to get a cutoff.)
    
    :return None: Performs operation on cbool in place 
    """
    var = var_in.copy()

    cc = cbool_to_cc(cbool)

    d = get_d(var)
    d[np.isnan(d)]=0

    # get an estimate of energy. Seems to work better thatn just normal energy, but that could be changed.
    E = []
    for start,stop in cc:
        E.append(np.sum(d[start:stop] ** 2))
    E = np.asarray(E)

    # find and remove bad contact episodes where the sum(d**2) exceeds a threshold
    bad_idx = E>np.median(E)*thresh
    for ii,(start,stop) in enumerate(cc):
        if bad_idx[ii]:
            cbool[start:stop]=0


def scale_by_contact(var,cc):
    # TODO: Test this with multi-dimensions of signal variable?
    '''
    Scale each contact episode individually. This is used to be able to perform outlier detection on the whole dataset
    
    :param var:         Signal on which to use outlier detection. Currently restricted to a single column vector. Probably adequate  
    :param cc:          Contact boundaries [num_contacts x 2] matrix of starts and stops
    
    :return var_out:    The signal quantity scaled withing each contact segments 
    '''
    # Copy to keep the original untouched
    var_out = var.copy()

    for start,stop in cc:
        snippet = var_out[start:stop,:]
        scaler = preprocessing.RobustScaler()
        idx = np.isfinite(snippet).ravel() # omit nans
        if np.any(idx):
            snippet[idx] = scaler.fit_transform(snippet[idx])
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