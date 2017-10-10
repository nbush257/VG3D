import numpy as np
from pygam import GAM
from pygam.utils import generate_X_grid
import statsmodels.api as sm
import elephant
from scipy import corrcoef
import quantities as pq
from neo.io import PickleIO as PIO
import sys
from numpy.random import binomial
try:
    from keras.models import Sequential
    from keras.constraints import max_norm
    from keras.layers import Dense,Convolution1D,Dropout,MaxPooling1D,AtrousConv1D,Flatten,AveragePooling1D,UpSampling1D,Activation
    from keras.regularizers import l2,l1
    from keras.constraints import Constraint
    import keras.backend as K
    keras_tgl=True
except ImportError:
    keras_tgl=False



def make_tensor(timeseries, window_size=16):
    X = np.empty((timeseries.shape[0],window_size,timeseries.shape[-1]))
    for ii in xrange(window_size,timeseries.shape[0]-window_size):
        X[ii,:,:] = timeseries[ii-window_size:ii,:]
    return X


def reshape_tensor(X):
    if X.ndim!=3:
        raise ValueError('Input tensor needs to have three dimensions')
    new_ndims = X.shape[1]*X.shape[2]
    pos = np.arange(new_ndims).reshape([X.shape[2],X.shape[1]])
    pos = pos.T.ravel()
    X2 = X.reshape([-1, new_ndims])
    X2 = X2[:,np.argsort(pos)]
    return X2



def run_GLM(X,y,family=None,link=None):
    ''' Runs a Generalized linear model on the design matrix X given the target y.
    This function adds its own constant term to the design matrix'''

    # assumes Binomial distribution
    if link==None:
        link = sm.genmod.families.links.logit
    if family==None:
        family=sm.families.Binomial(link=link)

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
            kernel = elephant.kernels.RectangularKernel(sigma=sigma * pq.ms)
        elif mode=='gaussian':
            kernel = elephant.kernels.GaussianKernel(sigma=sigma * pq.ms)
        elif mode=='exp':
            kernel = elephant.kernels.ExponentialKernel(sigma=sigma * pq.ms)
        elif mode=='alpha':
            kernel = elephant.kernels.AlphaKernel(sigma=sigma * pq.ms)
        elif mode=='epan':
            kernel = elephant.kernels.EpanechnikovLikeKernel(sigma=sigma * pq.ms)
        else:
            raise ValueError('Kernel mode is not a valid kernel')
        return kernel

    if Cbool==None:
        Cbool=np.ones_like(yhat)


    # only calculate correlation on non nans and contact(if desired)
    idx = np.logical_and(np.isfinite(yhat),Cbool)
    if type(sp)==dict:
        raise ValueError('Need to choose a cell from the spiketrain dict')

    # calculate Pearson correlation for all smoothings
    rr = []
    for sigma in sigma_vals:
        kernel =get_kernel(mode=kernel_mode,sigma=sigma)
        # get rate, need to convert from a neo analog signal to a numpy float,
        r = elephant.statistics.instantaneous_rate(sp, sampling_period=pq.ms, kernel=kernel).as_array().astype('f8')/1000

        rr.append(corrcoef(r[idx].ravel(), yhat[idx])[1, 0])
    return rr


def split_pos_neg(var):
    var_out = np.zeros([var.shape[0],(var.shape[1])*2])
    for ii in range(var.shape[1]):
        idx_pos = var[: ,ii] > 0.
        idx_neg = var[:, ii] < 0.
        var_out[idx_pos, ii] = var[idx_pos,ii]
        var_out[idx_neg, (ii+var.shape[1])] = var[idx_neg, ii]
    return var_out

if keras_tgl:

    class NonPosLast(Constraint):

        def __call__(self, w):
            last_row = w[:,-1, :] * K.cast(K.less_equal(w[:,-1, :], 0.), K.floatx())
            last_row = K.expand_dims(last_row, axis=1)
            full_w = K.concatenate([w[:,:-1, :], last_row], axis=1)
            return full_w

    def conv_model(X,y,winsize):
        # set y
        if y.ndim==1:
            y = y[:,np.newaxis]

        yhat = np.empty_like(y).ravel()
        yhat[:]=np.nan

        idx = np.all(np.all(np.isfinite(X), axis=1), axis=1)

        input_shape = X.shape[1:3]

        model = Sequential()
        model.add(Convolution1D(2,
                                winsize,
                                input_shape=input_shape,
                                kernel_constraint=NonPosLast()
                                )
                  )
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model.fit(X[idx,:,:], y[idx,:,:], epochs=2, batch_size=32, validation_split=0.33)
        yhat[idx] = model.predict(X[idx,:,:])
        return model,yhat

    def sim_conv(model,X,num_sims = 5):
        X[:,:,-1] = 0

        hist = np.zeros([X.shape[1],num_sims])
        yhat = np.empty([X.shape[0],num_sims])
        yhat[:]=np.nan
        is_spike = np.zeros(num_sims)
        for timestep in xrange(X.shape[1],X.shape[0]):
            if timestep%1000==0:
                print 'Timestep = {}'.format(timestep)
            tempX = X[timestep,:,:]
            tempX = np.tile(tempX,[num_sims,1,1])
            tempX[:,:,-1] = hist.T
            activation = model.predict(tempX).squeeze()
            for sim in xrange(num_sims):
                is_spike[sim] = binomial(1,activation[sim],1)
            hist=np.append(is_spike[np.newaxis,:],hist[:-1,:],axis=0)
            yhat[timestep,:]=is_spike
        return yhat

def make_bases(num_bases,endpoints,b=1):
    ''' ported from the neuroGLM toolbox. 
    returns:
        Bases,centers,and time vector    
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

    xx = np.matlib.repmat(nlin(iht + b)[:, np.newaxis], 1, num_bases)
    cc = np.matlib.repmat(ctrs[np.newaxis,:], len(iht), 1)

    return(f_t(xx,cc,db),ctrs,iht)


def apply_bases(X,bases):
    if np.any(np.isnan(X)):
        raise Warning('input contains NaNs. some timepoints will lose data')

    X_out = np.zeros([X.shape[0],X.shape[1]*bases.shape[1]])

    for ii in xrange(X.shape[1]):
        for jj in xrange(bases.shape[1]):
            temp = np.convolve(X[:,ii],bases[:,jj],mode='full')
            X_out[:,ii*bases.shape[1]+jj] = temp[:X.shape[0]]
    return(X_out)

