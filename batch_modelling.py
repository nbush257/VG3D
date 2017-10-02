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
window_size=5
p=r'/media/nbush257/5446399F46398332/Users/nbush257/Box Sync/__VG3D/deflection_trials/data/NEO'
p_save=r'/media/nbush257/5446399F46398332/Users/nbush257/Box Sync/__VG3D/deflection_trials/figs'
sigma_vals = np.arange(2,100,2)
all_corrs = {}
all_corrs['GAM'] = []
all_corrs['GLM'] = []
all_corrs['GLM_deriv'] = []
all_corrs['GLM_window'] = []
all_corrs['id'] = []
all_corrs['sigmas']=sigma_vals
all_corrs['window_size']=window_size

for file in glob.glob(os.path.join(p,'*.pkl')):

    print(os.path.basename(file))
    fid = PIO(os.path.join(p,file))
    blk = fid.read_block()

    M = get_var(blk,'M')[0]
    F = get_var(blk,'F')[0]
    Cbool = get_Cbool(blk)
    X = np.concatenate([M,F],axis=1)
    zeropad = np.zeros([1,6])
    Xdot = np.concatenate([zeropad,np.diff(X,axis=0)],axis=0)
    X_window = make_tensor(X,window_size)
    X_window = reshape_tensor(X_window)


    for unit in blk.channel_indexes[-1].units:
        try:
            id = blk.annotations['ratnum'] + blk.annotations['whisker'] + 'c{}'.format(unit.name[-1])
            sp = concatenate_sp(blk)[unit.name]
            b = binarize(sp,sampling_rate=pq.kHz)[:-1]
            y = b[:,np.newaxis].astype('f8')

            yhat_glm,glm = run_GLM(X,y,Cbool)
            yhat_glm_deriv,glm_deriv = run_GLM(np.concatenate([X,Xdot],axis=1),y,Cbool)
            yhat_glm_window,glm_window = run_GLM(X_window,y,Cbool)
            yhat_gam,gam = run_GAM(X,y,Cbool)


            corrs_glm = evaluate_correlation(yhat_glm,sp,Cbool,sigma_vals)
            corrs_glm_deriv = evaluate_correlation(yhat_glm_deriv,sp,Cbool,sigma_vals)
            corrs_glm_window = evaluate_correlation(yhat_glm_window,sp,Cbool,sigma_vals)
            corrs_gam = evaluate_correlation(yhat_gam,sp,Cbool,sigma_vals)

            plt.plot(sigma_vals,corrs_glm)
            plt.plot(sigma_vals,corrs_glm_deriv)
            plt.plot(sigma_vals,corrs_glm_window)
            plt.plot(sigma_vals,corrs_gam,'--')

            ax = plt.gca()
            ax.set_ylim(-0.1,1)
            ax.legend(['GLM','GLM with derivative','GLM 5ms window','GAM'])
            ax.set_xlabel('Gaussian Rate Kernel Sigma')
            ax.set_ylabel('Pearson Correlation')
            ax.set_title(id)
            plt.savefig(os.path.join(p_save,'model_performance_{}.png'.format(id)), dpi=300)
            plt.close('all')
            all_corrs['GAM'].append(corrs_gam)
            all_corrs['GLM'].append(corrs_glm)
            all_corrs['GLM_deriv'].append(corrs_glm_deriv)
            all_corrs['GLM_window'].append(corrs_glm_window)

            all_corrs['id'].append(id)
        except:
            plt.close('all')
            all_corrs['GAM'].append([])
            all_corrs['GLM'].append([])
            all_corrs['GLM_deriv'].append([])
            all_corrs['GLM_window'].append([])

            all_corrs['id'].append(id)

corr_save_name =os.path.join(p,'model_performance.pkl')
with open(corr_save_name,'w') as data_fid:
    pickle.dump(all_corrs,data_fid)
