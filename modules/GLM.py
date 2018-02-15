import numpy as np
import statsmodels.api as sm
import numpy.matlib as matlib
from pygam import GAM
from pygam.utils import generate_X_grid
import neo

import elephant
from scipy import corrcoef
import quantities as pq
import progressbar
import neoUtils

try:
    import cmt
    from cmt.models import STM,Bernoulli
    from cmt.nonlinear import LogisticFunction
except:
    pass

import sys
from numpy.random import binomial
try:
    from keras.models import Sequential
    from keras.layers import Dense,Convolution1D,Dropout,MaxPooling1D,AtrousConv1D,Flatten,AveragePooling1D,UpSampling1D,Activation,ELU
    from keras.utils.np_utils import to_categorical
    from keras.regularizers import l2,l1
    from keras.constraints import Constraint
    import keras.backend as K
    keras_tgl=True
except ImportError:
    keras_tgl=False


def make_bases(num_bases,endpoints,b=1):
    '''
    Imported from the neuroGLM toolbox. Makes non-linear raised cosine basis functions a la Pillow 2008
     
    :param num_bases:   int -- Number of basis functions 
    :param endpoints:   list or array -- first and last temporal peaks to be covered in the basis 
    :param b:           float -- nonlinear scaling parameter
    
    :returns (bases, centers, time_domain):
                        bases -- actual functions to be used in projections
                        centers -- indices of peaks in basis functinos
                        time_domain -- array of times which correspond to the indices in the basis functions.
     
    '''
    binsize=1
    endpoints = np.asarray(endpoints)
    nlin = lambda x: np.log(x+1e-20)
    invnl = lambda x: np.exp(x)-1e-20

    yrange = nlin(endpoints+b)
    db = np.diff(yrange)/(num_bases-1)
    ctrs = np.arange(yrange[0],yrange[1]+.1,db)
    mxt = invnl(yrange[-1]+2*db)-b
    iht = np.arange(0,mxt,binsize)
    bases = []

    def f_t(x,c,dc):
        a1 = (x-c)*np.pi/dc/2
        a1[a1>np.pi]=np.pi
        a1[a1<-np.pi]=-np.pi
        return (np.cos(a1)+1)/2

    xx = matlib.repmat(nlin(iht + b)[:, np.newaxis], 1, num_bases)
    cc = matlib.repmat(ctrs[np.newaxis,:], len(iht), 1)

    return(f_t(xx,cc,db),ctrs,iht)


def apply_bases(X, bases, lag=0):
    '''
    Project a matrix X into a lower dimensional basis given by bases.
    
    :param X:       A matrix [time x dimensions] of features to project into bases 
    :param bases:   basis functions as returned by make_bases
    :param lag:     number of samples by which to lag the input [for example, if we don't want to use the current 
                        observation to predict the current output, we can force a lag here] 
    
    :return X_out:  A matrix of [time x (dimensions * num_bases)] of the input mapped into the basis 
                        
    '''
    if np.any(np.isnan(X)):
        raise Warning('input contains NaNs. some timepoints will lose data')

    X_out = np.zeros([X.shape[0],X.shape[1]*bases.shape[1]])

    for ii in xrange(X.shape[1]):
        for jj in xrange(bases.shape[1]):
            temp = np.convolve(X[:,ii],bases[:,jj],mode='full')
            zero_pad = np.zeros(lag)
            temp = np.concatenate([zero_pad, temp[:-lag-1]])
            X_out[:,ii*bases.shape[1]+jj] = temp[:X.shape[0]]
    return(X_out)


def map_bases(weights,bases):
    '''
    takes the fitted weights from the GLM and maps them back into time space
        columns of ww are the basis, rows are the inputs
    
    :param weights: weights as returned by statsmodels GLM 
    :param bases:   basis functions as returned by make_bases 
    
    :return(filters, ww):   filters -- The weights represented in the time_domain, rather than the basis domain.
                            ww      -- the weights reshaped into a matrix that can be multiplied by the basis functions.
    '''

    ww = weights.reshape([-1,bases[0].shape[1]])
    filters = np.dot(bases[0],ww.T)

    return filters,ww


def make_tensor(timeseries, window_size=10,lag=0):
    '''
    Create a tensor which takes a window prior to the current time and sets that as the entries along the third dimension
    :param timeseries: variable on which to window
    :param window_size: number of samples for the window to look into the past
    :param lag: number of samples to lag the entries of the Q dimension
    :return: X -- a tensor of N x Q x M tensor where N is the number of timesteps in the original timeseries, Q is the window size, and M is the number of variable dimensions,
                    the 0th slice of the Q dimension is the current-lag, the -1th slice is the most latent time.
    '''
    if type(timeseries) == neo.core.analogsignal.AnalogSignal:
        timeseries = timeseries.magnitude

    X = np.empty((timeseries.shape[0],window_size,timeseries.shape[-1]))
    for ii in xrange(window_size+lag,timeseries.shape[0]-window_size):
        X[ii,:,:] = np.flipud(timeseries[ii-window_size+1-lag:ii+1-lag,:])
    return X


def make_binned_tensor(timeseries, binned_train, window_size=10, lag=0):
    '''
    Create a tensor which takes a window prior to the current bin and sets that to the third dimension.
    
    :param timeseries:      variable on which to window
    :param binned_train:    an elephant binned spike train--gives us the indices for the bins
    :param window_size:     The number of samples to look into the past
    :param lag:             The number of samples prior to the current time to include in the tensor. Defaults to zero: the 0th element of the window is the current time
    
    :return X:              a tensor of N x Q x M tensor where N is the number of bins, Q is the window size, and M is the number of variable dimensions,
                            the 0th slice of the Q dimension is  [bin_start_time-lag], the -1th slice is the most latent time[bin_start_time-lag-window_size].        
    '''

    # Init the output tensor
    X = np.empty((binned_train.num_bins, window_size, timeseries.shape[-1]))
    X[:] = np.nan

    # convert signal to array if needed
    if type(timeseries)==neo.core.analogsignal.AnalogSignal:
        timeseries =timeseries.magnitude

    # get indices of bin start time
    starts = binned_train.bin_edges.magnitude.astype('int')

    # slice the input signal and put it in the tensor
    for ii,start in enumerate(starts[:-1]):
        if (start-window_size)<0:
            continue
        X[ii,:,:]= np.flipud(timeseries[start - window_size+1-lag:start+1-lag, :])

    return X


def reshape_tensor(X):
    '''
    takes a 3D tensor and flattens it to a 2D array such that each observation in a window is now a column.
    Orders columns such that the adjacent columns correspond to the same variable at different points in the window
    (0th column is oth variable at current time, 1st column is zeroth variable 1 sample into the past...)

    :param X:   input tensor
    :return X2: 
    '''

    if X.ndim!=3:
        raise ValueError('Input tensor needs to have three dimensions')
    new_ndims = X.shape[1]*X.shape[2]
    pos = np.arange(new_ndims).reshape([X.shape[2],X.shape[1]])
    pos = pos.T.ravel()
    X2 = X.reshape([-1, new_ndims])
    X2 = X2[:,np.argsort(pos)]
    return X2


def add_spike_history(X, y, B=None):
    '''
    Underdeveloped function which concatenates spike history to a design matrix X. The spike history is mapped to a set of basis functions.
    :param X: A design matrix of [observations x dimensions]
    :param y: A vector of spikes. If boolean dtype, map to int
    :param B: [Optional] basis functions as returned by make_bases
    
    :return XX: A concatenated design matrix with the spike history as the last columns 
    '''
    if B is None:
        B = make_bases(2, [1, 4])[0]
        yy = apply_bases(y, B, lag=0)
    elif B is -1:
        yy = np.concatenate([[[0]],y[:-1]],axis=0)


    XX = np.concatenate([X,yy],axis=1)
    return XX


def split_pos_neg(var):
    '''
    Takes a matrix and doubles the number of dimensions by splitting the positive and negative values into separate dimensions
    :param var:         A matrix [num_obs x num_dims] of input feature values.
    :return var_out:    A matrix [num_obs x num_dims*2] with positive and negative values split.
    
    '''
    var_out = np.zeros([var.shape[0],(var.shape[1])*2])
    for ii in range(var.shape[1]):
        idx_pos = var[: ,ii] > 0.
        idx_neg = var[:, ii] < 0.
        var_out[idx_pos, ii] = var[idx_pos,ii]
        var_out[idx_neg, (ii+var.shape[1])] = var[idx_neg, ii]
    return var_out


def run_GLM(X,y,family=None,link=None):
    ''' Runs a Generalized linear model on the design matrix X given the target y.
    This function adds its own constant term to the design matrix'''

    # assumes Binomial distribution
    if link==None:
        link = sm.genmod.families.links.logit
    if family==None:
        family=sm.families.Binomial(link=link)
    else:
        family = family(link=link)

    # make y a column vector
    if y.ndim==1:
        y = y[:,np.newaxis]

    # make y a float
    if y.dtype=='bool':
        y = y.astype('f8')

    # init yhat
    yhat = np.empty_like(y).ravel()
    yhat[:] = np.nan

    # get nans so we dont predict on nans
    idx = np.all(np.isfinite(X), axis=1)

    # add a constant term to the design matrix
    constant = np.ones([X.shape[0],1])
    X = np.concatenate([constant, X], axis=1)

    # fit and predict
    glm_binom = sm.GLM(y,X,family=family,missing='drop')
    glm_result = glm_binom.fit()
    # history_names = glm_binom.exog_names[-2:]
    # res_c = glm_binom.fit_constrained(['{}<=0'.format(history_names[0]),'{}<=0'.format(history_names[1])])
    yhat[idx] = glm_result.predict(X[idx,:])

    return yhat,glm_result


def run_GAM(X,y,n_splines=15,distr='binomial',link='logit'):
    ''' Run a Generalized additive model on the inputs.
    This function does NOT add a constant, as I think pygam takes care of that.'''

    # make y a column vector
    if y.ndim==1:
        y = y[:,np.newaxis]

    # init yhat
    yhat = np.empty_like(y).ravel()
    yhat[:]=np.nan

    # get idx of nans so we dont try to predict those
    idx = np.all(np.isfinite(X), axis=1)

    # init, fit, and predict othe GAM
    gam = GAM(distribution=distr, link=link, n_splines=n_splines)
    gam.gridsearch(X[idx, :], y[idx])
    yhat[idx] = gam.predict(X[idx,:])

    return yhat,gam


def run_STM(X,y,num_components=3,num_features=20):
    if X.shape[0]>X.shape[1]:
        X = X.T

    if y.ndim==1:
        y = y[:,np.newaxis]

    if y.shape[0]>y.shape[1]:
        y = y.T

    model = STM(X.shape[0], 0, num_components, num_features, LogisticFunction, Bernoulli)

    model.train(X,y, parameters={
        'verbosity':1,
        'threshold':1e-7
            }
                )
    yhat = model.predict(X).ravel()
    return yhat, model


def evaluate_correlation(yhat,sp,Cbool=None,kernel_mode='box',sigma_vals=np.arange(2, 100, 2)):
    '''
    Takes a predict spike rate and smooths the
    observed spike rate at different values to find the optimal smoothing.

    -- if a Cbool is passed, the calculation of the correlation only occurs during contact
    -- kernel input is a string ('box','gaussian','exp','alpha','epan')

    '''
    def get_kernel(mode='box',sigma=5.):
        ''' Get the kernel for a given mode and sigma'''
        if mode=='box':
            box_sigma = sigma / 2 / np.sqrt(3)
            kernel = elephant.kernels.RectangularKernel(sigma=box_sigma)
        elif mode=='gaussian':
            kernel = elephant.kernels.GaussianKernel(sigma=sigma)
        elif mode=='exp':
            kernel = elephant.kernels.ExponentialKernel(sigma=sigma)
        elif mode=='alpha':
            kernel = elephant.kernels.AlphaKernel(sigma=sigma)
        elif mode=='epan':
            kernel = elephant.kernels.EpanechnikovLikeKernel(sigma=sigma)
        else:
            raise ValueError('Kernel mode is not a valid kernel')

        return kernel

    if Cbool is None:
        Cbool=np.ones_like(yhat)


    # only calculate correlation on non nans and contact(if desired)
    idx = np.logical_and(np.isfinite(yhat),Cbool)

    # calculate Pearson correlation for all smoothings
    rr = []
    for sigma in sigma_vals:
        kernel =get_kernel(mode=kernel_mode,sigma=sigma)
        # get rate, need to convert from a neo analog signal to a numpy float,
        r = elephant.statistics.instantaneous_rate(sp, kernel=kernel, sampling_period=pq.ms).magnitude.squeeze()

        rr.append(corrcoef(r[idx], yhat[idx])[1, 0])
    return rr


def create_design_matrix(blk,varlist,window=1,binsize=1,deriv_tgl=False,bases=None):
    '''
    Takes a list of variables and turns it into a matrix.
    Sets the non-contact mechanics to zero, but keeps all the kinematics as NaN
    You can append the derivative or apply the pillow bases, or both.
    Scales, but does not center the output
    '''
    X = []
    if type(window)==pq.quantity.Quantity:
        window = int(window)

    if type(binsize)==pq.quantity.Quantity:
        binsize = int(binsize)
    Cbool = neoUtils.get_Cbool(blk,-1)
    use_flags = neoUtils.concatenate_epochs(blk)

    # ================================ #
    # GET THE CONCATENATED DESIGN MATRIX OF REQUESTED VARS
    # ================================ #

    for varname in varlist:
        if varname in ['MB','FB']:
            var = neoUtils.get_var(blk,varname[0],keep_neo=False)[0]
            var = neoUtils.get_MB_MD(var)[0]
            var[np.invert(Cbool)]=0
        elif varname in ['MD','FD']:
            var = neoUtils.get_var(blk,varname[0],keep_neo=False)[0]
            var = neoUtils.get_MB_MD(var)[1]
            var[np.invert(Cbool)]=0
        elif varname in ['ROT','ROTD']:
            TH = neoUtils.get_var(blk,'TH',keep_neo=False)[0]
            PH = neoUtils.get_var(blk,'PHIE',keep_neo=False)[0]
            TH = neoUtils.center_var(TH,use_flags=use_flags)
            PH = neoUtils.center_var(PH,use_flags=use_flags)
            TH[np.invert(Cbool)] = 0
            PH[np.invert(Cbool)] = 0
            if varname=='ROT':
                var = np.sqrt(TH**2+PH**2)
            else:
                var = np.arctan2(PH,TH)
        else:
            var = neoUtils.get_var(blk,varname, keep_neo=False)[0]

        if varname in ['M','F']:
            var[np.invert(Cbool),:]=0
        if varname in ['TH','PHIE']:
            var = neoUtils.center_var(var,use_flags)
            var[np.invert(Cbool),:]=0

        var = neoUtils.replace_NaNs(var,'pchip')
        var = neoUtils.replace_NaNs(var,'interp')

        X.append(var)
    X = np.concatenate(X, axis=1)

    return X

def bin_design_matrix(X,binsize):
    idx = np.arange(0,X.shape[0],binsize)
    return(X[idx,:])

def get_deriv(blk,blk_smooth,varlist,smoothing=range(10)):
    """

    :param blk:
    :param blk_smooth:
    :param varlist:
    :param smoothing: A list of indices of which smoothing parameter to use. Default is all 10
    :return:
    """
    use_flags = neoUtils.concatenate_epochs(blk)
    Cbool = neoUtils.get_Cbool(blk)
    X =[]
    for varname in varlist:
        var = neoUtils.get_var(blk_smooth, varname+'_smoothed', keep_neo=False)[0]

        if varname in ['M', 'F']:
            var[np.invert(Cbool), :, :] = 0
        if varname in ['TH', 'PHIE']:
            for ii in smoothing:
                var[:, :, ii] = neoUtils.center_var(var[:,:,ii], use_flags)
            var[np.invert(Cbool), :, :] = 0
        var = var[:, :, smoothing]
        # var = neoUtils.replace_NaNs(var, 'pchip')
        # var = neoUtils.replace_NaNs(var, 'interp')

        X.append(var)
    X = np.concatenate(X, axis=1)
    zero_pad = np.zeros([1,X.shape[1],X.shape[2]])
    Xdot = np.diff(np.concatenate([zero_pad,X],axis=0),axis=0)
    Xdot = np.reshape(Xdot,[Xdot.shape[0],Xdot.shape[1]*Xdot.shape[2]])
    return(Xdot)

if keras_tgl:
    def conv_model(X,y,num_filters,winsize,l2_penalty=1e-8,is_bool=True):
        # set y
        if y.ndim==1:
            y = y[:, np.newaxis, np.newaxis]
        elif y.ndim==2:
            y = y[:, np.newaxis]

        yhat = np.empty_like(y).ravel()
        yhat[:]=np.nan
        if X.ndim!=3:
            X = make_tensor(X,winsize)

        idx = np.all(np.all(np.isfinite(X), axis=1), axis=1)

        input_shape = X.shape[1:3]

        model = Sequential()
        model.add(Convolution1D(num_filters,
                                winsize,
                                input_shape=input_shape,
                                kernel_regularizer=l2(l2_penalty)
                                )
                  )
        model.add(Activation('relu'))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        model.fit(X[idx, :, :], y[idx, :, :], epochs=5, batch_size=32, validation_split=0.)

        yhat[idx] = model.predict(X[idx,:,:]).squeeze()

        return yhat,model

    def sim_conv(model,X,num_sims = 5):
        X[:,:,-1] = 0

        hist = np.zeros([X.shape[1],num_sims])
        yhat = np.empty([X.shape[0],num_sims])
        yhat[:]=np.nan
        is_spike = np.zeros(num_sims)
        for timestep in xrange(X.shape[1],X.shape[0]):
            if timestep%1000==0:
                print('Timestep = {}'.format(timestep))
            tempX = X[timestep,:,:]
            tempX = np.tile(tempX,[num_sims,1,1])
            tempX[:,:,-1] = hist.T
            activation = model.predict(tempX).squeeze()
            for sim in xrange(num_sims):
                is_spike[sim] = binomial(1,activation[sim],1)
            hist=np.append(is_spike[np.newaxis,:],hist[:-1,:],axis=0)
            yhat[timestep,:]=is_spike
        return yhat

def sim(yhat, y,num_sims=100,lim=500):
    ISI = np.diff(np.where(y)[0])
    prob,time=np.histogram(ISI[ISI<lim],lim,density=True)
    cum_prob = np.cumsum(prob)
    cum_prob[cum_prob>1]=1
    h = np.ones(num_sims,dtype='int')*lim-1
    sim_out=[]
    bar = progressbar.ProgressBar(max_value=len(yhat))
    for ii,p in enumerate(yhat):
        if ii%10000==0:
            bar.update(ii)
        p = np.repeat(p,num_sims)*cum_prob[h]
        sim_temp = np.random.binomial(1,p)
        h+=1
        h[sim_temp.astype('bool')] = 0
        h[h>=lim]=lim-1
        sim_out.append(sim_temp)
    sim_out = np.array(sim_out)
    return sim_out
