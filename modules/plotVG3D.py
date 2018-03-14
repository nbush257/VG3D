import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import neoUtils
import spikeAnalysis
import worldGeometry
import numpy as np
import matplotlib.gridspec as gridspec
import quantities as pq
import os
sns.set()

def plot3():
    f = plt.figure()
    return(Axes3D(f))
def polar():
    f = plt.figure()
    return(f.add_subplot(111,projection='polar'))

def savefig(p_save,name,dpi=600):
    plt.savefig(os.path.join(p_save,name),dpi=dpi)

def polar_histogram(vals, nbins=20, kind='bar'):
    R,theta = np.histogram(vals,nbins)
    plt.polar()
    if kind== 'bar':
        plt.bar(theta[:-1],R,width=np.mean(np.diff(theta)),bottom=0,align='edge',color='k',alpha=0.4)
    elif kind== 'line':
        theta =theta+np.mean(np.diff(theta))/2
        plt.plot(theta[:-1], R,'ko--')
def arclength_group_colors():
    return(sns.color_palette('Blues',4)[1:])
def set_fig_style():
    '''
    standard set of values to use when making plots
    :return:
    '''
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['figure.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['font.sans-serif'] = 'Arial'
    sns.set()
    sns.set_style('ticks')
    dpi_res = 600
    fig_width = 6.9 # in
    fig_height = 9 # in
    figsize=(fig_width,fig_height)
    ext = 'png'

    return(dpi_res,figsize,ext)
def rotate_3D_plot(X):
    ax = plot3()
    ax.plot(X[:,0],X[:,1],X[:,2],'.')
