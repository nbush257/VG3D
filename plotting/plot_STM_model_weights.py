import quantities as pq
import neo
import varTuning
import elephant
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style('ticks')
# ===================== #
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['font.sans-serif'] = 'Arial'
dpi_res = 600
fig_width = 6.9 # in
sns.set_style('ticks')
fig_height = 9 # in
ext = 'png'
kernel=16
try:
    p_save = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')
except KeyError:
    p_save = '/projects/p30144/_VG3D/deflections/_NEO/results/'

fname =os.path.join(p_save,r'201702C3c0_STM_continuous.npz')
dat = np.load(fname)
root = os.path.basename(fname)[:10]
yhat = dat['yhat'].item()['full']
y = dat['y']
cbool = dat['cbool']
X = dat['X']
# model  = dat['models'].item()['full']

def make_gridpace(X,col1,col2,nbins=100):
    var1= X[:,col1]
    var2= X[:,col2]
    X1 = np.linspace(np.nanmin(var1),np.nanmax(var1),nbins)[:,np.newaxis]
    X2 = np.linspace(np.nanmin(var2),np.nanmax(var2),nbins)[:,np.newaxis]
    return(np.concatenate([X1,X2],axis=1))
def weight_investigate(model):
    u = model.features
    v = model.linear_predictor
    g = model.nonlinearity
    w = model.predictors
    b = model.weights
    a = model.biases
    wnl = np.dot(b,u.T**2)
    return(wnl,w)

def model_id_to_name():
    'assumes the 32 length full model'
    var_names = ['Mx','My','Mz','Fx','Fy','Fz','TH','PHI']
    vard_names = [[y+'_d{}'.format(x) for y in var_names] for x in [0,1,2]]
    vard_names = [item for sublist in vard_names for item in sublist]
    return(var_names+vard_names)


spt = neo.SpikeTrain(np.where(y)[0]*pq.ms,sampling_rate=pq.kHz,t_stop=y.shape[0]*pq.ms)
if kernel is not None:
    kernel = elephant.kernels.GaussianKernel(kernel*pq.ms)
    rate = elephant.statistics.instantaneous_rate(spt,sampling_period=pq.ms,kernel=kernel).magnitude.ravel()
    rate = rate/1000.
else:
    rate = y

pred_response,edges1,edges2 = varTuning.joint_response_hist(X[:,6],X[:,7],yhat,cbool)
obs_response,edges1,edges2 = varTuning.joint_response_hist(X[:,6],X[:,7],rate,cbool)
max_r = np.max([np.max(pred_response),np.max(obs_response)])
# ==================
plt.figure()
plt.subplot(121)
h=plt.pcolormesh(edges1[:-1],edges2[:-1],pred_response,vmin=0,vmax=max_r)
plt.ylabel('$\\Delta\\phi$')
plt.xlabel('$\\Delta\\theta$')
plt.axvline(color='k')
plt.axhline(color='k')
plt.axis('square')
plt.title('Predicted')
plt.subplot(122)
plt.pcolormesh(edges1[:-1],edges2[:-1],obs_response,vmin=0,vmax=max_r)
plt.axis('square')
plt.axvline(color='k')
plt.axhline(color='k')
plt.title('Observed')
plt.xlabel('$\\Delta\\theta$')
plt.suptitle('{}'.format(root))
fig = plt.gcf()
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.9,0.25,0.02,0.5])
fig.colorbar(h,cax=cbar_ax)
plt.show()

