import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import neoUtils
import spikeAnalysis
import worldGeometry
import numpy as np
import matplotlib.gridspec as gridspec
import quantities as pq
sns.set()

def plot3():
    f = plt.figure()
    return(Axes3D(f))
def polar():
    f = plt.figure()
    return(f.add_subplot(111,projection='polar'))



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