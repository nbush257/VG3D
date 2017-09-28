from neo_utils import *
from spikeAnalysis import *
import numpy as np
from scipy.io.matlab import loadmat, savemat
from neo.core import SpikeTrain
from quantities import ms, s
import neo
import quantities as pq
import elephant
import sys
from neo.io import PickleIO as NIO
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
sns.set()



def get_MD_tuning_curve(MD,b,nbins=100):

    fig = plt.figure()
    barplot=False

    bins = np.arange(-np.pi, np.pi, 2*np.pi/nbins)
    MD_prior,edges_prior = np.histogram(MD[np.isfinite(MD)],bins=bins)
    MD_post,edges_post = np.histogram(MD[np.isfinite(MD)],bins=bins,weights=b[np.isfinite(MD)])
    MD_bayes = MD_post/MD_prior
    PD = edges_prior[np.argmax(MD_post/MD_prior)]
    smooth = lowess(MD_bayes,edges_post[:-1],frac=0.1)
    ax = plt.subplot(111, polar=True)
    if barplot:
        width = (2 * np.pi) / nbins
        ax.bar(edges_post[:-1],MD_post/MD_prior,width=width,edgecolor='k')
    else:

        ax.plot(edges_post[:-1],MD_post/MD_prior,'o')
        ax.plot(smooth[:,0],smooth[:,1],linewidth=5,alpha=0.6)

    plt.tight_layout()
    return fig


# MB Bayes
def get_MB_tuning_curve(MB,b,nbins=100):
    fig = plt.figure()
    max_MB = np.nanmax(MB)
    idx_MB = np.isfinite(MB)
    # nbins = 20
    # step = round(max_MB/nbins,abs(np.floor(math.log10(max_MB))+2)
    # bins=np.arange(0,np.max(MB[idx_MB]),nbins)
    MB_prior,edges_prior = np.histogram(MB[idx_MB],bins=nbins)
    MB_post,edges_post = np.histogram(MB[idx_MB],bins=nbins,weights=b[idx_MB])
    MB_prior[MB_prior<1]=0
    plt.plot(edges_post[:-1],MB_post/MB_prior,'o')
    ax = plt.gca()
    ax.set_xlabel('MB (N-m)')
    ax.set_ylabel('Spike Probability')
    ax.set_title('Probability of a spike given Bending Moment')
    return ax

def bayes_plots(var1,var2,b,bins=None):
    # bin_size = 5e-9
    if type(bins)==int:
        nbins = bins
        bins=None
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
    H_bayes = H_post/H_prior
    H_bayes = H_bayes.T
    idx_mask = np.logical_or(np.isnan(H_bayes),H_prior.T<1)
    H_bayesm = np.ma.masked_where(idx_mask,H_bayes)
    X,Y = np.meshgrid(x_edges,y_edges)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    levels = MaxNLocator(nbins=30).tick_values(H_bayesm.min(), H_bayesm.max())
    cf = ax.contourf(x_edges[:-1], y_edges[:-1], H_bayesm, levels=levels, cmap='OrRd')
    # pmesh = ax.pcolormesh(x_edges,y_edges,H_bayesm,cmap='OrRd')
    fig.colorbar(cf)
    ax.set_aspect('equal')

def plot_summary(blk,cell_no):
    plotMD=True
    plotMB=True
    plotbayes=True
    p_save = r'C:\Users\guru\Desktop\test'
    root = blk.annotations['ratnum']+blk.annotations['whisker']+'c{:01d}'.format(cell_no)
    cell_str = 'cell_{}'.format(cell_no)
    M = get_var(blk,'M')[0]
    MD = np.arctan2(M[:,2],M[:,1])
    MB = np.sqrt(M[:,1]**2+M[:,2]**2)

    sp = concatenate_sp(blk)
    st = sp[cell_str]
    kernel = elephant.kernels.GaussianKernel(5*pq.ms)
    b = binarize(st,sampling_rate=pq.kHz)
    r = np.array(instantaneous_rate(st,sampling_period=pq.ms,kernel =kernel)).ravel()
    trains = get_contact_sliced_trains(blk)
    trains = trains[2][cell_str]

    b = b.astype('float')[:-1]

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


if __name__=='__main__':
    p = r'C:\Users\guru\Box Sync\__VG3D\_E3D_1K\deflection_trials'
    for file in glob.glob(p+'\*.pkl'):
        print(file)
        fid = NIO(os.path.join(p, file))
        blk = fid.read_block()
        for cell_no,cell in enumerate(blk.channel_indexes[-1].units):
            plot_summary(blk,cell_no)

