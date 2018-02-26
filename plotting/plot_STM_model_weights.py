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
try:
    p_save = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\results')
except KeyError:
    p_save = '/projects/p30144/_VG3D/deflections/_NEO/results/'

fname =os.path.join(p_save,r'201708D1c0_STM_continuous.npz')
dat = np.load(fname)
yhat = dat['yhat'].item()
y = dat['y']
X = dat['X']
model  = dat['models'].item()['full']

def make_gridpace(X,col1,col2,nbins=100):
    var1= X[:,col1]
    var2= X[:,col2]
    X1 = np.linspace(np.nanmin(var1),np.nanmax(var1),nbins)[:,np.newaxis]
    X2 = np.linspace(np.nanmin(var2),np.nanmax(var2),nbins)[:,np.newaxis]
    return(np.concatenate([X1,X2],axis=1))
def apply_models(X,model,col1,col2)
    u = model.features
    v = model.linear_predictor
    g = model.nonlinearity
    w = model.predictors
    b = model.weights
    a = model.biases
    Xout = make_gridspace(X,col1,col2)

