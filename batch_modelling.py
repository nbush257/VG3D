from neo.io import PickleIO as PIO
import os
from neo_utils import *
from mechanics import *
from GLM import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle as pickle
import statsmodels.api as sm
import elephant
import pygam
import glob
from sklearn.preprocessing import RobustScaler,StandardScaler

sns.set()

window_size=5
scale_tgl = False
p=r'C:\Users\nbush257\Box Sync\__VG3D\deflection_trials\data'
p_save=r'C:\Users\nbush257\Box Sync\__VG3D\deflection_trials\figs'
sigma_vals = np.arange(2,100,2)
B = make_bases(5,[0,15],b=2)
all_corrs = {}
all_corrs['GAM'] = []
all_corrs['GLM'] = []
all_corrs['GAM_deriv'] = []
all_corrs['id'] = []
all_corrs['sigmas']=sigma_vals
all_corrs['B'] = B

all_models = {}
all_models['GLM']=[]
all_models['GAM']=[]
all_models['GAM_deriv']=[]
for file in glob.glob(os.path.join(p,'*.pkl')):

    print(os.path.basename(file))
    fid = PIO(os.path.join(p,file))
    blk = fid.read_block()

    M = get_var(blk,'M')[0]
    F = get_var(blk,'F')[0]

    Cbool = get_Cbool(blk)

    X = np.concatenate([M,F],axis=1)
    X[np.invert(Cbool),:]=0
    replace_NaNs(X, 'pchip')
    replace_NaNs(X, 'interp')

    Xdot = get_deriv(X)


    X_pillow = apply_bases(X,B[0])

    scaler = StandardScaler(with_mean=False)
    X_pillow = scaler.fit_transform(X_pillow)
    if scale_tgl:
        scaler = RobustScaler()
        idx = np.all(np.isfinite(X),axis=1)
        Xs = X[idx,:]
        Xs =scaler.fit_transform(Xs)
        X[idx,:]=Xs[:,:]

    for unit in blk.channel_indexes[-1].units:
        # try:
        id = blk.annotations['ratnum'] + blk.annotations['whisker'] + 'c{}'.format(unit.name[-1])
        sp = concatenate_sp(blk)[unit.name]
        b = binarize(sp,sampling_rate=pq.kHz)[:-1]
        y = b[:,np.newaxis].astype('f8')

        # yhat_glm,glm = run_GLM(X_pillow,y,Cbool)
        yhat_gam,gam = run_GAM(X,y,Cbool)
        yhat_gam_deriv,gam_deriv = run_GAM(np.concatenate([X,Xdot],axis=1),y,Cbool)

        # corrs_glm = evaluate_correlation(yhat_glm,sp,Cbool,sigma_vals)
        corrs_gam = evaluate_correlation(yhat_gam,sp,Cbool,sigma_vals)
        corrs_gam_deriv = evaluate_correlation(yhat_gam_deriv,sp,Cbool,sigma_vals)

        # plt.plot(sigma_vals,corrs_glm)
        plt.plot(sigma_vals,corrs_gam,'--')
        plt.plot(sigma_vals,corrs_gam_deriv,'--')

        ax = plt.gca()
        ax.set_ylim(-0.1,1)
        ax.legend(['GAM','GAM w deriv'])
        ax.set_xlabel('Gaussian Rate Kernel Sigma')
        ax.set_ylabel('Pearson Correlation')
        ax.set_title(id)
        plt.savefig(os.path.join(p_save,'model_performance_{}.png'.format(id)), dpi=300)
        plt.close('all')

        all_corrs['GAM'].append(corrs_gam)
        # all_corrs['GLM'].append(corrs_glm)
        all_corrs['GAM_deriv'].append(corrs_gam_deriv)

        # all_models['GLM'].append(glm)
        all_models['GAM'].append(gam)
        all_models['GAM_deriv'].append(gam_deriv)


        all_corrs['id'].append(id)
        # except:
        #     plt.close('all')
        #     all_corrs['GAM'].append([])
        #     all_corrs['GLM'].append([])
        #     all_corrs['GAM_deriv'].append([])
        #     all_models['GLM'].append([])
        #     all_models['GAM'].append([])
        #     all_models['GAM_deriv'].append([])
        #     all_corrs['id'].append(id)

corr_save_name =os.path.join(p,'model_performance.pkl')
with open(corr_save_name,'w') as data_fid:
    pickle.dump(all_corrs,data_fid)
    pickle.dump(all_models,data_fid)
