import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import elephant
import quantities as pq
import keras
import scipy
from GLM import *
import pygam
import statsmodels
sns.set()

colors = [[0.4, 0.5, 1], [1, 0.5, 0.4], [0.8, 0.8, 0.8], [0.6, 0.6, 0.6], [0.4, 0.4, 0.4], [0.2, 0.2, 0.2]]


def analyse_model(p,f):
    res = np.load((os.path.join(p,f)))
    model_names = ['glm', 'gam', 'conv_1_node', 'conv_2_node', 'conv_3_node', 'conv_4_node']
    # get sigmas
    sigma_vals = res['sigma_vals']
    # get bases:
    B = res['B']
    # ============================= #
    # GET ACCURACIES
    # ============================= #
    corrs = res['corrs'].item()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ii,model in enumerate(model_names):
        plt.plot(sigma_vals,corrs[model],color=colors[ii])
    ax.legend(model_names)
    ax.set_title('Performance of {}'.format(f[-15:-4]))
    ax.set_ylim(-0.1,1)
    ax.set_ylabel('Pearson correlation')
    ax.set_xlabel('Box width (ms)')
    plt.savefig(os.path.join(p,'performance_{}.png'.format(f[-15:-4])),dpi=300)

    rr = np.empty([len(sigma_vals),0])
    for model in model_names:
        r = np.array(corrs[model])[:,np.newaxis]
        rr = np.append(rr,r,axis=1)


    models = res['mdl'].item()
    weights = {}
    for name,model in models.iteritems():
        if type(model)==keras.models.Sequential:
            fig = plt.figure()
            w = model.get_weights()[0]
            weights[name] = w
            # for ii in xrange(weights.shape[-1]):
                # ax=fig.add_subplot(2,2,ii+1)
                # ax.plot(w[:,:,ii])
        elif type(model)==pygam.pygam.GAM:
            pass
        elif type(model)==statsmodels.genmod.generalized_linear_model.GLMResultsWrapper:
            w = model.params[1:]
            w = np.reshape(w,[-1,6])
            weights[name] = np.dot(B[0],w)
        else:
            print('model not recognized, skipping')

    yhat = np.empty([res['y'].shape[0],0])
    for model in model_names:
        yhat = np.append(yhat,res['yhat'].item()[model][:,np.newaxis],axis=1)

    return rr,weights,yhat,res['y']

def concatenate_data():
    p = r'K:\VG3D\_model_results'
    f_out = 'glm-gam-conv1-4model_results.npz'
    spec = 'model*.npz'
    all_rr = np.empty([50, 6, 0])
    all_weights = []
    all_yhat = []
    all_y = []
    id = []

    for f in glob.glob(os.path.join(p, spec)):
        print(f)
        rr, weights, yhat, y = analyse_model(p, f)
        all_rr = np.concatenate([all_rr, rr[:, :, np.newaxis]], axis=-1)
        all_weights.append(weights)
        all_yhat.append(yhat)
        all_y.append(y)
        id.append(f[-4:-15])
    np.savez(os.path.join(p, f_out),
             rr=all_rr,
             weights=all_weights,
             yhat=all_yhat,
             y=all_y,
             id=id)

def all_plots():
    p =r'K:\VG3D\_model_results'
    f = r'glm-gam-conv1-4model_results.npz'
    sigma_vals = np.arange(2,200,4)
    model_names = ['glm', 'gam', 'conv_1_node', 'conv_2_node', 'conv_3_node', 'conv_4_node']
    res = np.load(os.path.join(p,f))
    weights = res['weights']
    rr = res['rr']
    rr_idx=3
    sub_rr = rr[rr_idx,:,:].squeeze()
    best_smoothing_idx = np.argmax(rr,axis=0)
    best_smoothing_vals = sigma_vals[best_smoothing_idx]
    max_rr = np.nanmax(rr, axis=0)
    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    sns.violinplot(data=max_rr.T,palette=colors,bw=.2)
    swarm_colors = np.array([[0,0,0.1,0.1,0.7,0.7]]).T*np.array([[1,1,1]])
    sns.swarmplot(data=max_rr.T,size=4,palette=swarm_colors)
    ax1.set_xticklabels(model_names)
    ax1.set_title('Performance at optimal smoothing for all cell/models')
    ax1.set_ylabel('Pearson Correlation')

    ax2 = fig.add_subplot(212)
    sns.swarmplot(data=best_smoothing_vals.T,palette=colors,orient='h')
    ax2.set_yticklabels(model_names)
    ax2.set_xlabel('Box kernel width (ms)')
    ax2.set_title('Best smoothing kernel width')
    plt.tight_layout()
    for ii in xrange(len(model_names)):
        sns.distplot(best_smoothing_vals[ii,:],20,kde=False)

    # performance and smoothing parameter
    for ii in xrange(6):
        sns.jointplot(best_smoothing_vals[ii, :], max_rr[ii, :],edgecolor='w',marginal_kws=dict(bins=25), size=5, ratio=4, color=colors[ii]).plot_joint(sns.kdeplot,n_levels=3)


if __name__=='__main__':
    concatenate_data()
