from neo.io import PickleIO as PIO
import os
from neo_utils import *
from mechanics import *
from GLM import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
if sys.version_info.major==3:
    import pickle
else:
    import cPickle as pickle
import statsmodels.api as sm
import elephant
import pygam
import glob

from sklearn.preprocessing import RobustScaler,StandardScaler
sns.set()


def main(argv=None):
    if argv==None:
        argv=sys.argv
    fname = argv[1]
    if len(argv)==3:
        plot_tgl=argv[2]
    else:
        plot_tgl=False
    p_save = os.path.split(fname)[0]


    sigma_vals = np.arange(2, 200, 4)
    B = make_bases(5, [0, 15], b=2)
    winsize=int(B[0].shape[0])

    print(os.path.basename(fname))
    fid = PIO(fname)
    blk = fid.read_block()

    M = get_var(blk,'M')[0]
    F = get_var(blk,'F')[0]

    Cbool = get_Cbool(blk)

    X = np.concatenate([M,F],axis=1)
    X[np.invert(Cbool),:]=0
    X = replace_NaNs(X, 'pchip')
    X = replace_NaNs(X, 'interp')

    # Xdot = get_deriv(X)

    X_pillow = apply_bases(X,B[0])

    scaler = StandardScaler(with_mean=False)
    X_pillow = scaler.fit_transform(X_pillow)
    X = scaler.fit_transform(X)

    for unit in blk.channel_indexes[-1].units:
        yhat={}
        mdl={}
        corrs={}

        id =get_root(blk,int(unit.name[-1]))
        f_save = os.path.join(p_save, 'model_results_{}.npz'.format(id))
        if os.path.isfile(f_save):
            continue

        sp = concatenate_sp(blk)[unit.name]
        b = binarize(sp,sampling_rate=pq.kHz)[:-1]
        y = b[:,np.newaxis].astype('f8')

        yhat['glm'],mdl['glm'] = run_GLM(X_pillow,y)
        yhat['gam'],mdl['gam'] = run_GAM(X,y)
        for num_filters in xrange(1,5):
            yhat['conv_{}_node'.format(num_filters)],mdl['conv_{}_node'.format(num_filters)]=conv_model(X,y,num_filters=num_filters,winsize=winsize)


        for model in yhat.iterkeys():
            corrs[model] = evaluate_correlation(yhat[model],sp,Cbool=Cbool,sigma_vals=sigma_vals)
        if plot_tgl:
            for model in yhat.iterkeys():
                plt.plot(sigma_vals,corrs[model])

            ax = plt.gca()
            ax.set_ylim(-0.1,1)
            ax.legend(corrs.get_keys())
            ax.set_xlabel('Gaussian Rate Kernel Sigma')
            ax.set_ylabel('Pearson Correlation')
            ax.set_title(id)
            plt.savefig(os.path.join(p_save,'model_performance_{}.png'.format(id)), dpi=300)
            plt.close('all')

    np.savez(f_save,
             corrs=corrs,
             yhat=yhat,
             sigma_vals=sigma_vals,
             mdl=mdl,
             y=y,
             X=X,
             X_pillow=X_pillow,
             B=B)

if __name__=='__main__':
    sys.exit(main())
