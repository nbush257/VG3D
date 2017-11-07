from neo_utils import *
from spikeAnalysis import *
from mechanics import *
import numpy as np
from scipy.io.matlab import loadmat, savemat
from neo.core import SpikeTrain
from quantities import ms, s
import neo
import quantities as pq
import elephant
import sys
from neo.io import PickleIO as PIO
import math
import glob
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from elephant.statistics import *
import elephant
from matplotlib.ticker import MaxNLocator
from sklearn.neighbors import KernelDensity as KD
from statsmodels.nonparametric.smoothers_lowess import lowess
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import vonmises
import spm1d
sns.set()



def get_MD_tuning_curve(MD,b,nbins=100,smooth_tgl=False,barplot=False):

    fig = plt.figure()
    if smooth_tgl==False:
        smooth=None

    bins = np.arange(-np.pi, np.pi, 2*np.pi/nbins)
    MD_prior,edges_prior = np.histogram(MD[np.isfinite(MD)],bins=bins)
    MD_post,edges_post = np.histogram(MD[np.isfinite(MD)],bins=bins,weights=b[np.isfinite(MD)])
    MD_bayes = MD_post/MD_prior
    PD = edges_prior[np.argmax(MD_post/MD_prior)]
    if smooth_tgl:
        smooth = lowess(MD_bayes,edges_post[:-1],frac=0.1)
    else:
        smooth = None
    ax = plt.subplot(111, polar=True)
    if barplot:
        width = (2 * np.pi) / nbins
        ax.bar(edges_post[:-1],MD_post/MD_prior,width=width,edgecolor='k')
    else:
        ax.plot(edges_post[:-1],MD_post/MD_prior,'o')
        if smooth_tgl:
            ax.plot(smooth[:,0],smooth[:,1],linewidth=5,alpha=0.6)

    plt.tight_layout()
    return fig,MD_bayes,edges_prior,smooth


# MB Bayes
def get_MB_tuning_curve(MB,b,nbins=100,min_obs=1):
    fig = plt.figure()
    max_MB = np.nanmax(MB)
    idx_MB = np.isfinite(MB)
    # nbins = 20
    # step = round(max_MB/nbins,abs(np.floor(math.log10(max_MB))+2)
    # bins=np.arange(0,np.max(MB[idx_MB]),nbins)
    MB_prior,edges_prior = np.histogram(MB[idx_MB],bins=nbins)
    if b.dtype=='bool':
        MB_post, edges_post = np.histogram(MB[np.logical_and(idx_MB,b)], bins=nbins)
    else:
        MB_post,edges_post = np.histogram(MB[idx_MB],bins=nbins,weights=b[idx_MB])
    MB_prior[MB_prior<min_obs]=0
    MB_bayes = np.divide(MB_post,MB_prior,dtype='f8')
    plt.plot(edges_post[:-1],MB_bayes,'o')
    ax = plt.gca()
    ax.set_xlabel('MB (N-m)')
    ax.set_ylabel('Spike Probability')
    ax.set_title('Probability of a spike given Bending Moment')
    return ax,MB_bayes,edges_post

def get_single_var_tuning(var,b,nbins=100,min_obs=5):
    idx_var = np.isfinite(var)
    prior, edges = np.histogram(var[idx_var], bins=nbins)
    if b.dtype == 'bool':
        post, edges_post = np.histogram(var[np.logical_and(idx_var, b)], bins=nbins)
    else:
        post, edges_post = np.histogram(var[idx_var], bins=nbins, weights=b[idx_var])
    prior[prior < min_obs] = 0
    bayes = np.divide(post, prior, dtype='f8')
    return(bayes,edges)

def bayes_plots(var1,var2,b,bins=None,ax=None,contour=False,min_obs=5):
    # bin_size = 5e-9
    if type(bins)==int:
        nbins = bins
        bins=None
    elif type(bins)==list:
        pass
    else:
        nbins = 50


    idx = np.isfinite(var1) & np.isfinite(var2)
    if bins == None:
        bins = []

        max_var1 = np.nanmax(var1)
        min_var1 = np.nanmin(var1)
        step = round(max_var1 / nbins, abs(np.floor(math.log10(max_var1)).astype('int64')) + 2)
        bins.append(np.arange(min_var1, max_var1, step))
        # bins.append(np.arange(min_var1,max_var1,bin_size))

        max_var2 = np.nanmax(var2)
        min_var2 = np.nanmin(var2)

        step = round(max_var2 / nbins, abs(np.floor(math.log10(max_var2)).astype('int64')) + 2)
        bins.append(np.arange(min_var2, max_var2, step))
        # bins.append(np.arange(min_var2, max_var2, bin_size))

    H_prior,x_edges,y_edges= np.histogram2d(var1[idx],var2[idx],bins=bins)
    H_post = np.histogram2d(var1[idx],var2[idx],bins=bins,weights = b[idx])[0]
    H_bayes = np.divide(H_post,H_prior,dtype='f8')
    H_bayes = H_bayes.T
    idx_mask = np.logical_or(np.isnan(H_bayes),H_prior.T<min_obs)
    H_bayesm = np.ma.masked_where(idx_mask,H_bayes)
    X,Y = np.meshgrid(x_edges,y_edges)
    if ax==None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    levels = MaxNLocator(nbins=30).tick_values(H_bayesm.min(), H_bayesm.max())
    if contour:
        handle = ax.contourf(x_edges[:-1], y_edges[:-1], H_bayesm, levels=levels, cmap='OrRd')
        for c in handle.collections:
            c.set_edgecolor("face")
    else:
        handle = ax.pcolormesh(x_edges[:-1],y_edges[:-1],H_bayesm,cmap='OrRd',edgecolors='None')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(handle,cax=cax)
    ax.grid('off')
    return ax


def plot_summary(blk,cell_no,p_save):
    plotMD=True
    plotMB=True
    plotbayes=True
    root = get_root(blk,cell_no)
    cell_str = 'cell_{}'.format(cell_no)

    M = get_var(blk,'M')[0]
    MB,MD = get_MB_MD(M)
    sp = concatenate_sp(blk)
    st = sp[cell_str]
    kernel = elephant.kernels.GaussianKernel(5*pq.ms)
    b = binarize(st,sampling_rate=pq.kHz)
    r = np.array(instantaneous_rate(st,sampling_period=pq.ms,kernel =kernel)).ravel()
    trains = get_contact_sliced_trains(blk)


    if plotMD:
        get_MD_tuning_curve(MD,r,nbins=100)
        plt.gca().set_title('Spike rate by MD {}'.format(root))
        plt.savefig(os.path.join(p_save,root+'_MD.png'),dpi=300)
        plt.close()
    if plotMB:
        get_MB_tuning_curve(MB,r,nbins=100)
        plt.gca().set_title(root)
        plt.savefig(os.path.join(p_save,root + '_MB.png'), dpi=300)
        plt.close()

    if plotbayes:
        bayes_plots(M[:,1]*10e6,M[:,2]*10e6,r,50)
        ax = plt.gca()
        ax.set_title(root)
        ax.set_xlabel('M$_y$ ($\mu$N-m)')
        ax.set_ylabel('M$_z$ ($\mu$N-m)')
        plt.tight_layout()
        ax.grid('off')
        ax.set_facecolor([0.3,0.3,0.3])
        plt.savefig(os.path.join(p_save, root + '_heatmap.png'), dpi=300)
        plt.close()
def get_PD_from_hist(theta_k,rate):

    L_dir = np.abs(
        np.sum(
            rate * np.exp(1j * theta_k[:-1])) / np.sum(rate)
    )
    # L_dir = 1-CircVar
    x = rate * np.cos(theta_k[:-1])
    y = rate * np.sin(theta_k[:-1])

    X = np.sum(x) / len(x)
    Y = np.sum(y) / len(x)

    theta = np.arctan2(Y, X)
    return(theta,L_dir)




def PD_fitting(MD,sp):
    '''outputs histogram edges and heights, along with a vector sum and weight of the bins'''
    if type(MD)==neo.core.analogsignal.AnalogSignal:
        MD = MD.magnitude

    MD = MD.ravel()

    not_nan = np.where(np.isfinite(MD))[0]
    prior,prior_edges = np.histogram(MD[np.isfinite(MD)],bins=100)
    if type(sp)==neo.core.spiketrain.SpikeTrain:
        spt = sp.times.magnitude.astype('int')
        idx = [x for x in spt if x in not_nan]
        posterior, theta_k = np.histogram(MD[idx], bins=100)
    else:
        posterior, theta_k = np.histogram(MD[not_nan], weights=sp[not_nan],bins=100)

    rate = np.divide(posterior,prior,dtype='float32')


    theta,L_dir = get_PD_from_hist(theta_k,rate)

    return rate,theta_k,L_dir,theta

def direction_test(rate,theta_k):
    x = rate*np.cos(theta_k[:-1])
    y = rate*np.sin(theta_k[:-1])
    X = np.sum(x) / len(x)
    Y = np.sum(y) / len(x)

    obs = np.concatenate([x[:, np.newaxis], y[:, np.newaxis]], axis=1)
    preferred = np.array([X,Y])[:,np.newaxis]/np.sqrt(X**2+Y**2)

    projection=np.dot(obs,preferred)
    t = scipy.stats.ttest_1samp(projection,0.05)
    return t.pvalue

if __name__=='__main__':
    p = r'C:\Users\guru\Box Sync\__VG3D\deflection_trials\data'
    p_save = r'C:\Users\guru\Box Sync\__VG3D\deflection_trials\figs'
    for file in glob.glob(p+'\*.pkl'):
        print(file)
        fid = PIO(os.path.join(p, file))
        blk = fid.read_block()
        for cell_no,cell in enumerate(blk.channel_indexes[-1].units):
            plot_summary(blk,cell_no,p_save)


