from neo.io import PickleIO as PIO
import os
from neo_utils import *
from mechanics import *
from GLM import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import cPickle as pickle
import statsmodels.api as sm
import elephant
import pygam
import glob
window_size=1
p=r'C:\Users\nbush257\Box Sync\__VG3D\deflection_trials\data\NEO'
p_save=r'C:\Users\nbush257\Box Sync\__VG3D\deflection_trials\figs'
all_corrs = {}
all_corrs['GAM'] = []
all_corrs['GLM'] = []
all_corrs['id'] = []
for file in glob.glob(os.path.join(p,'*.pkl')):
    print(os.path.basename(file))
    fid = PIO(os.path.join(p,file))
    blk = fid.read_block()

    M = get_var(blk,'M')[0]
    F = get_var(blk,'F')[0]
    Cbool = get_Cbool(blk)
    X = np.concatenate([M,F],axis=1)


    for unit in blk.channel_indexes[-1].units:
        id = blk.annotations['ratnum'] + blk.annotations['whisker'] + 'c{}'.format(unit.name[-1])
        sp = concatenate_sp(blk)[unit.name]
        b = binarize(sp,sampling_rate=pq.kHz)[:-1]
        y = b[:,np.newaxis].astype('f8')

        yhat_glm,glm = run_GLM(X,y,Cbool)
        yhat_gam,gam = run_GAM(X,y,Cbool)

        sigma_vals = np.arange(2,100,2)
        corrs_glm = evaluate_correlation(yhat_glm,sp,Cbool,sigma_vals)
        corrs_gam = evaluate_correlation(yhat_gam,sp,Cbool,sigma_vals)
        plt.plot(sigma_vals,corrs_glm)
        plt.plot(sigma_vals,corrs_gam)
        ax = plt.gca()
        ax.legend(['GLM','GAM'])
        ax.set_xlabel('Gaussian Rate Kernel Sigma')
        ax.set_ylabel('Pearson Correlation')
        ax.set_title(id)
        plt.savefig(os.path.join(p_save,'model_performance_{}.png'.format(id)), dpi=300)

        all_corrs['GAM'].append(corrs_gam)
        all_corrs['GLM'].append(corrs_gam)
        all_corrs['id'].append(id)

corr_save_name =os.path.join(p,'model_performance.pkl')
with open(corr_save_name,'w') as data_fid:
    pickle.dump(all_corrs,data_fid)
