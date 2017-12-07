import numpy as np
from sklearn import preprocessing,covariance
from scipy import signal,interpolate
import matplotlib.pyplot as plt
from optparse import OptionParser
import os
import scipy.io.matlab as sio

# TODO: Port filtering from MATLAB code to python??
# TODO: Port neural alignment from MATLAB code to python?? Probably a different module.
# TODO: Build input output code from the elastica data format.
# TODO: Make the contamination parameter easily editable after fitting the outlier detection (i.e., change the boundary of the decision function without recomputing')

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


def impute_snippet(snippet,kind = 'linear'):
    '''
    Helper function to impute over nan gaps in a given snippet.  
    Be careful--this will perform on all inputs, so be sure you have excluded segments you don't want to impute over by this point 
    
    :param snippet: A row or column vector of the signal to be imputed. 
    :param kind:    Passes to scipy.interpolate.interp1d kind
    
    :return:   The interpolated snippet 
    '''
    if snippet.ndim<1:
        raise ValueError('snippet must be a 2D array')
    # map snippet and pad with zeros on either side
    zeropad_length=3
    x = np.arange(-zeropad_length,len(snippet)+zeropad_length)
    y = np.concatenate([np.zeros([zeropad_length,snippet.shape[-1]]),snippet,np.zeros([zeropad_length,snippet.shape[-1]])])
    for ii in xrange(y.shape[1]):
        temp_y = y[:,ii].ravel()
        f = interpolate.interp1d(x[np.isfinite(temp_y)],temp_y[np.isfinite(temp_y)],kind=kind)
        y[np.isnan(temp_y),ii] = f(x[np.isnan(temp_y)])

    snippet_out = y[zeropad_length:len(snippet)+zeropad_length,:]

    return snippet_out


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
        gaps = cbool_to_cc(np.isnan(snippet))
        if np.any(np.diff(gaps)>thresh):
            use_flag[start:stop]=0
        else:
            snippet_out = impute_snippet(snippet)
            var_out[start:stop] = snippet_out
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
    var = scale_by_contact(var, cc)

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
        scaler = preprocessing.RobustScaler(with_centering=False)
        idx = np.isfinite(snippet).ravel() # omit nans
        if np.any(idx):
            snippet[idx] = scaler.fit_transform(snippet[idx])
    return var_out


def cleanup(var,cbool,outlier_thresh=0.05):
    '''
    Runs the primary algorithm to find outliers and remove bad contact segments
    
    :param var:     The variable to use as a signal for when the tracking is bad 
    :param cbool:   The contact boolean
    
    :return use_flags:  A boolean mask inherited from cbool which is only 1 during good contacts (cbool includes all contacts)
                            These should be set to zero
    :return outliers:   A boolean mask indicating which frames are bad tracking outliers.
                            These should be NaN'd and interpolated over
    '''

    # either impute small nan gaps or remove contacts with more than 10 consecutive NaNs.
    # Allows us to make all non-flagged var=0
    var_imputed, use_flags = fill_nan_gaps(var, cbool, thresh=10)
    var_imputed[use_flags == 0] = 0

    # Perform a small medfilt to filter over single point outliers
    var_filt = signal.medfilt(var_imputed, kernel_size=[3, 1])


    # remove contacts where there are many bad points
    remove_bad_contacts(var_filt, use_flags,thresh=10)

    # Get CC for all the contact segments to be kept
    cc = cbool_to_cc(use_flags)

    # =========================== #
    # =========================== #
    # Find point outliers once the bad contact segments have been deleted
    var_scaled = scale_by_contact(var_imputed, cc)
    var_d = get_d(var_imputed)
    var_d_scaled = scale_by_contact(var_d, cc)

    X = np.concatenate([var_scaled, var_d_scaled], axis=1)
    y = np.zeros(X.shape[0], dtype='int')

    # Fit outlier detection
    clf = covariance.EllipticEnvelope(contamination=outlier_thresh)
    idx = np.squeeze(use_flags == 1)
    clf.fit(X[idx, :])

    # Find outliers
    y[idx] = clf.predict(X[idx, :])
    y[y == 1] = 0
    y[y == -1] = 1

    # =========================== #
    # =========================== #

    # set outputs [use_flags, outliers]
    outliers = y == 1

    # var_out is used mostly to evaluate the quality of the outlier detection.
    # We should evelntually use 'use_flags' and 'outliers' to alter all mechanics data uniformly.

    var_out = var.copy()
    var_out[use_flags == 0] = np.nan
    var_out[outliers] = np.nan

    # impute over the variable.
    # for start, stop in cc:
    #     if stop - start > 10:
    #
    #         var_out[start:stop] = impute_snippet(var_out[start:stop])
    var_out,use_flags = fill_nan_gaps(var_out,use_flags)
    var_out[use_flags == 0] = 0
    var_out_filt = signal.medfilt(var_out, kernel_size=[3, 1])
    var_out_filt = signal.savgol_filter(var_out_filt.ravel(),7,3)


    plt.plot(var)
    plt.plot(var_out_filt)
    plt.show()
    return(use_flags,outliers)


def main(fname,use_var='M',outlier_thresh=0.1):
    print('Using variable: {}\t Using outlier_thresh={}'.format(use_var,outlier_thresh))
    print('Loading {} ...'.format(os.path.basename(fname)))
    fid = sio.loadmat(fname)
    print('Loaded!')
    var = fid[use_var]
    cbool = fid['C']
    use_flags,outliers = cleanup(var,cbool,outlier_thresh=outlier_thresh)

    fname_out = os.path.splitext(fname)[0]+'_outliers'
    print('Saving to {}...'.format(fname_out))
    sio.savemat(fname_out,{'use_flags':use_flags,'outliers':outliers})
    print('Saved')

    return 0

if __name__=='__main__':
    usage = "usage: %prog filename [options]"
    parser = OptionParser(usage)
    parser.add_option('-t', '--thresh',
                      dest='outlier_thresh',
                      default=0.001,
                      type=float,
                      help='Sensitivity to outliers to remove')
    parser.add_option('-v', '--var',
                      dest='use_var',
                      default='M',
                      type=str,
                      help='Variable to be used as a signal for outlier detection. Must be a valid variable in the matlab data')

    (options, args) = parser.parse_args()
    if len(args)<1:
        parser.error('Need to pass a filename fist')

    main(args[0],options.use_var,options.outlier_thresh)

