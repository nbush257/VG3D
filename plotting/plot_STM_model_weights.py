import numpy as np
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
p_save = r'C:\Users\guru\Box Sync\__VG3D\_deflection_trials\_NEO\results'

fname =r'C:\Users\guru\Box Sync\__VG3D\_deflection_trials\_NEO\results\201708C3c0_STM_continuous.npz'
dat = np.load(fname)
yhat = dat['yhat'].item()
y = dat['y']


