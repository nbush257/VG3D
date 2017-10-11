
import numpy as np
from scipy.io.matlab import loadmat, savemat
from neo.core import SpikeTrain
from quantities import ms, s
import neo
import quantities as pq
import elephant
import sys
from neo.io import NeoMatlabIO as NIO
import glob
import os
import re
from neo_utils import *
import matplotlib.pyplot as plt
import seaborn as sns
from elephant.statistics import *
import elephant
from spikeAnalysis import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation


p = r'C:\Users\guru\Box Sync\__VG3D\_E3D_1K\deflection_trials'
f = r'rat2017_08_FEB15_VG_D1_NEO.mat'
cell_no = 0
cell_str = 'cell_{}'.format(cell_no)
fid=NIO(os.path.join(p,f))
blk=fid.read_block()
add_channel_indexes(blk)

theta = get_var(blk,'THcp')[0].ravel()
phi = get_var(blk,'PHIcp')[0].ravel()
r = get_var(blk,'Rcp')[0].ravel()
PHIE = get_var(blk,'PHIE')[0].ravel()
TH = get_var(blk,'TH')[0].ravel()



sp = concatenate_sp(blk)
st = sp[cell_str]
kernel = elephant.kernels.GaussianKernel(5*pq.ms)
b = binarize(st,sampling_rate=pq.kHz)
b = b[:-1]
r = np.array(instantaneous_rate(st,sampling_period=pq.ms,kernel =kernel)).ravel()

trains = get_contact_sliced_trains(blk)
trains = trains[2][cell_str]


from numpy import cos,sin
x = r*sin(phi)*cos(theta)
y = r*sin(phi)*sin(theta)
z = r*cos(phi)



ax = Axes3D(plt.figure())

ax.plot(x,y,z,'o',alpha=0.01)
x_spike = np.empty_like(x);x_spike[:]=np.nan
y_spike = np.empty_like(x);y_spike[:]=np.nan
z_spike = np.empty_like(x);z_spike[:]=np.nan

x_spike[b] = x[b]
y_spike[b] = y[b]
z_spike[b] = z[b]
ax.plot(x_spike,y_spike,z_spike,'o',alpha=0.01)

f = plt.figure()
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

canvas = FigureCanvas(f)
ax = Axes3D(f)
p = r'C:\Users\guru\Desktop\test'
count=0
for ii,frame in enumerate(x):
    if ii%1000==0:
        print('Frame {}'.format(ii))
    history = 10
    if np.isnan(frame):
        continue
    plt.cla()

    ax.scatter(x[ii-history:ii],y[ii-history:ii],z[ii-history:ii],'ko',c=np.linspace(0,0.4,history))
    ax.plot(x[ii-history:ii],y[ii-history:ii],z[ii-history:ii],'k',alpha=0.4)


    ax.plot(x_spike[:ii],y_spike[:ii],z_spike[:ii],'ro',alpha=0.1)

    ax.set_xlim([np.nanmin(x), np.nanmax(x)])
    ax.set_ylim([np.nanmin(y), np.nanmax(y)])
    ax.set_zlim([np.nanmin(z), np.nanmax(z)])
    plt.title('Frame: {}'.format(ii))

    frame_name = os.path.join(p,'{:06d}.tiff'.format(count))
    count+=1
    plt.savefig(frame_name)
