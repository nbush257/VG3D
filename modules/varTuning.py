import numpy as np
import neo
import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import neoUtils
sns.set()


def direction_test(rate, theta_k):
    '''This will eventually be used to calcualte whether direction tuning is signifcant for a given neuron'''
    if True:
        raise Exception('This code has not been verified to work.')

    x = rate * np.cos(theta_k[:-1])
    y = rate * np.sin(theta_k[:-1])
    X = np.sum(x) / len(x)
    Y = np.sum(y) / len(x)

    obs = np.concatenate([x[:, np.newaxis], y[:, np.newaxis]], axis=1)
    preferred = np.array([X, Y])[:, np.newaxis] / np.sqrt(X ** 2 + Y ** 2)

    projection = np.dot(obs, preferred)
    t = scipy.stats.ttest_1samp(projection, 0.05)
    return t.pvalue


def get_PD_from_hist(theta_k,rate):
    '''Calculate the vector sum direction and tuning strength from a histogram of responses in polar space
    e.g.: we have a histogram on -pi:pi of spike rates for a given Bending direction.
    This should generalize to any polar histogram
    INPUTS: theta_k -- sampled bin locations from a polar histogram.
                        * Assumes number of bins is the same as the number of observed rates (i.e. if you use bin edges you will probably have to truncate the input to fit)
                        * Bin centers is a better usage
            rate -- observed rate at each bin location
    OUTPUTS:    theta -- the vector mean direction of the input bin locations and centers
                L_dir -- the strength of the tuning as defined by Mazurek FiNC 2014. Equivalient to 1- Circular Variance
            '''
    # Calculate the direction tuning strength
    L_dir = np.abs(
        np.sum(
            rate * np.exp(1j * theta_k)) / np.sum(rate)
    )

    # calculate vector mean
    x = rate * np.cos(theta_k)
    y = rate * np.sin(theta_k)

    X = np.sum(x) / len(x)
    Y = np.sum(y) / len(x)

    theta = np.arctan2(Y, X)

    return theta,L_dir


def angular_response_hist(angular_var, sp, use_flags, nbins=100,min_obs=5):
    '''Given an angular variable (like MD that varies on -pi:pi,
    returns the probability of observing a spike (or gives a spike rate) normalized by
    the number of observations of that angular variable.
    INPUTS: angular var -- either a numpy array or a neo analog signal. Should be 1-D
            sp -- type: neo.core.SpikeTrain, numpy array. Sp can either be single spikes or a rate

    OUTPUTS:    rate -- the stimulus evoked rate at each observed theta bin
                theta_k -- the observed theta bins
                theta -- the preferred direction as determined by vector mean
                L_dir -- The preferred direction tuning strength (1-CircVar)
    '''

    if type(angular_var)==neo.core.analogsignal.AnalogSignal:
        angular_var = angular_var.magnitude
    if angular_var.ndim==2:
        if angular_var.shape[1] == 1:
            angular_var = angular_var.ravel()
        else:
            raise Exception('Angular var must be able to be unambiguously converted into a vector')
    if type(nbins)==int:
        bins = np.linspace(-np.pi,np.pi,nbins+1,endpoint=True)
    # not nan is a list of finite sample indices, rather than a boolean mask. This is used in computing the posterior
    not_nan = np.where(np.logical_and(np.isfinite(angular_var),use_flags))[0]
    prior,prior_edges = np.histogram(angular_var[not_nan], bins=bins)
    prior[prior < min_obs] = 0
    # allows the function to take a spike train or a continuous rate to get the posterior
    if type(sp)==neo.core.spiketrain.SpikeTrain:
        spt = sp.times.magnitude.astype('int')
        idx = [x for x in spt if x in not_nan]
        posterior, theta_k = np.histogram(angular_var[idx], bins=bins)
    else:
        posterior, theta_k = np.histogram(angular_var[not_nan], weights=sp[not_nan], bins=bins)

    #
    rate = np.divide(posterior,prior,dtype='float32')
    theta,L_dir = get_PD_from_hist(theta_k[:-1],rate)

    return rate,theta_k,theta,L_dir


def stim_response_hist(var, sp, use_flags, nbins=100, min_obs=5):
    ''' Return the histograms for a single variable normalized by the number of observations of that variable
    INPUTS: var -- either a numpy array or a neo analog signal. Should be unambiguously converted to a vector
            sp -- either a neo spike train, or a numpy array. The numpy array can be a continuous rate estimate
            nbins -- number of bins, or boundaries of bins to make the histograms
            min_obs -- minimum number of observations of the prior to count as an instance. If less than min obs, returns nan for that bin

    OUTPUTS:    response -- probability (or rate) of spiking for the associated bin index
                stim_edges -- edges of the bins of the stimulus histogram
    '''

    # Handle input variable:
    if type(var)==neo.core.analogsignal.AnalogSignal:
        var = var.magnitude
    if var.ndim==2:
        if var.shape[1] == 1:
            var = var.ravel()
        else:
            raise Exception('var must be able to be unambiguously converted into a vector')
    if type(use_flags) is neo.core.epoch.Epoch:
        raise ValueError('use_flags has to be a boolean vector')

    # grab indicies of finite variable observations
    not_nan = np.where(np.logical_and(np.isfinite(var).ravel(),use_flags))[0]

    # compute prior
    prior, stim_edges = np.histogram(var[not_nan], bins=nbins)

    # compute posterior
    if type(sp)==neo.core.spiketrain.SpikeTrain:
        spt = sp.times.magnitude.astype('int')
        idx = [x for x in spt if x in not_nan]
        post, stim_edges = np.histogram(var[idx], bins=nbins)
    else:
        post, stim_edges = np.histogram(var[not_nan], weights=sp[not_nan], bins=nbins)

    # remove too few observations
    prior[prior < min_obs] = 0

    # get normalized response
    response = np.divide(post, prior, dtype='float32')

    return response,stim_edges

def joint_response_hist(var1, var2, sp, cbool, bins=None, min_obs=5):
    ''' Returns the noramlized response histogram of two variables
    INPUTS:     var1,var2 -- the two variables on which to plot the joint histogram. Must be either 1D numpy or column vector
                sp -- either a neo spike train, or a numpy array. The numpy array can be a continuous rate estimate
                nbins -- number of bins, or boundaries of bins to make the histograms
                min_obs -- minimum number of observations of the prior to count as an instance. If less than min obs, returns nan for that bin
    OUTPUS:     bayesm -- a masked joint histogram hiehgts
                var1_edges = bin edges on the first variable
                var2_edges = bin edges on the second variable
    '''
    # handle bins -- could probably be cleaned up NEB
    if type(bins)==int:
        bins = [bins,bins]
    elif type(bins)==list:
        pass
    else:
        bins = [50,50]

    if type(var1)==neo.core.analogsignal.AnalogSignal:
        var1 = var1.magnitude.ravel()
    if type(var2)==neo.core.analogsignal.AnalogSignal:
        var2 = var2.magnitude.ravel()
    if var1.ndim==2:
        if var1.shape[1] == 1:
            var1 = var1.ravel()
        else:
            raise Exception('var1 must be able to be unambiguously converted into a vector')
    if var2.ndim==2:
        if var2.shape[1] == 1:
            var2 = var2.ravel()
        else:
            raise Exception('var2 must be able to be unambiguously converted into a vector')

    # use only observations where both vars are finite
    not_nan_mask = np.logical_and(np.isfinite(var1), np.isfinite(var2))
    not_nan_mask = np.logical_and(not_nan_mask,cbool)
    not_nan = np.where(not_nan_mask)[0]

    # handle bins -- NEB may want to make this more flexible/clean.
    # if bins == None:
    #     bins = []
    #     max_var1 = np.nanmax(var1)
    #     min_var1 = np.nanmin(var1)
    #     step = round(max_var1 / nbins, abs(np.floor(math.log10(max_var1)).astype('int64')) + 2)
    #     bins.append(np.arange(min_var1, max_var1, step))
    #     # bins.append(np.arange(min_var1,max_var1,bin_size))
    #
    #     max_var2 = np.nanmax(var2)
    #     min_var2 = np.nanmin(var2)
    #
    #     step = round(max_var2 / nbins, abs(np.floor(math.log10(max_var2)).astype('int64')) + 2)
    #     bins.append(np.arange(min_var2, max_var2, step))
    #     # bins.append(np.arange(min_var2, max_var2, bin_size))


    prior,var1_edges,var2_edges= np.histogram2d(var1[not_nan_mask],var2[not_nan_mask],bins=bins)

    if type(sp)==neo.core.spiketrain.SpikeTrain:
        spt = sp.times.magnitude.astype('int')
        idx = [x for x in spt if x in not_nan]
        post = np.histogram2d(var1[idx], var2[idx], bins=[var1_edges,var2_edges],)[0]
    else:
        post = np.histogram2d(var1[not_nan_mask], var2[not_nan_mask], bins=[var1_edges,var2_edges], weights = sp[not_nan_mask])[0]

    bayes = np.divide(post,prior,dtype='float32')
    bayes = bayes.T
    idx_mask = np.logical_or(np.isnan(bayes),prior.T<min_obs)
    bayesm = np.ma.masked_where(idx_mask,bayes)
    return bayesm,var1_edges,var2_edges


def plot_joint_response(bayes,x_edges,y_edges,contour=False,ax=None):
    '''previously used code to plot the joints. It is moved out of the calculation of the joint histograms.'''
    if ax==None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    levels = MaxNLocator(nbins=30).tick_values(bayes.min(), bayes.max())
    if contour:
        handle = ax.contourf(x_edges[:-1], y_edges[:-1], bayes, levels=levels, cmap='OrRd')
        for c in handle.collections:
            c.set_edgecolor("face")
    else:
        handle = ax.pcolormesh(x_edges[:-1],y_edges[:-1],bayes,cmap='OrRd',edgecolors='None')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(handle,cax=cax)
    ax.grid('off')
    return ax

