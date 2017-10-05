import numpy as np
from pygam import GAM
from pygam.utils import generate_X_grid
import statsmodels.api as sm
import elephant
from scipy import corrcoef
import quantities as pq
# from keras.models import Sequential
# from keras.layers import Dense,Convolution1D,Dropout,MaxPooling1D,AtrousConv1D,Flatten,AveragePooling1D,UpSampling1D,Activation
# from keras.regularizers import l2,l1

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


def get_design_matrix(X,y,Cbool):
    if y.ndim==1:
        y = y[:,np.newaxis]
    return X,y


def run_GLM(X,y,Cbool):
    if y.ndim==1:
        y = y[:,np.newaxis]

    if y.dtype=='bool':
        y = y.astype('f8')

    yhat = np.empty_like(y).ravel()
    yhat[:] = np.nan

    # set non contact to zero
    X[np.invert(Cbool),:]=0
    idx = np.all(np.isfinite(X), axis=1)

    link = sm.genmod.families.links.logit
    glm_binom = sm.GLM(y,X,family=sm.families.Binomial(link=link),missing='drop')
    glm_result = glm_binom.fit()
    yhat[idx] = -np.log(glm_result.predict(X[idx,:]))

    return yhat,glm_result


def run_GAM(X,y,Cbool,n_splines=5):
    if y.ndim==1:
        y = y[:,np.newaxis]

    yhat = np.empty_like(y).ravel()
    yhat[:]=np.nan

    X[np.invert(Cbool), :] = 0
    idx = np.all(np.isfinite(X), axis=1)

    gam = GAM(distribution='binomial', link='logit', n_splines=n_splines)
    gam.gridsearch(X[idx, :], y[idx])
    yhat[idx] = gam.predict(X[idx,:])
    return yhat,gam


def evaluate_correlation(yhat,sp,Cbool,sigma_vals=np.arange(2, 100, 2)):
    idx = np.logical_and(np.isfinite(yhat),Cbool)
    if type(sp)==dict:
        raise ValueError('Need to choose a cell from the spiketrain dict')

    rr = []
    for sigma in sigma_vals:
        kernel = elephant.kernels.RectangularKernel(sigma=sigma * pq.ms)
        r = elephant.statistics.instantaneous_rate(sp, sampling_period=pq.ms, kernel=kernel)
        r_ = r.as_array().astype('f8')/1000
        rr.append(corrcoef(r_[idx].ravel(), yhat[idx])[1, 0])
    return rr


def split_pos_neg(var):
    var_out = np.zeros([var.shape[0],(var.shape[1])*2])
    for ii in range(var.shape[1]):
        idx_pos = var[: ,ii] > 0.
        idx_neg = var[:, ii] < 0.
        var_out[idx_pos, ii] = var[idx_pos,ii]
        var_out[idx_neg, (ii+var.shape[1])] = var[idx_neg, ii]
    return var_out


# def conv_model(X,y,cbool):
#     # set y
#     if y.ndim==1:
#         y = y[:,np.newaxis]
#
#     yhat = np.empty_like(y).ravel()
#     yhat[:]=np.nan
#
#     # set non contact to zero
#     X[np.invert(Cbool),:,:]=0
#     idx = np.all(np.isfinite(X), axis=1)
#
#
#     input_shape = X.shape[1:3]
#
#     model = Sequential()
#     model.add(Convolution1D(1, 20, input_shape=input_shape))
#     model.add(Dense(1))
#     model.add(Activation('sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#     model.fit(X[idx,:], y[idx], epochs=5, batch_size=32, validation_split=0.33)


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
