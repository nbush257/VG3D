import numpy as np
import glob
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import elephant
import quantities as pq
import keras
import scipy
from GLM import *
import pygam
import sklearn
import statsmodels
import cmt
from mpl_toolkits.mplot3d import Axes3D
sns.set()
# colors = [[0.4, 0.5, 1], [1, 0.5, 0.4], [0.8, 0.8, 0.8], [0.6, 0.6, 0.6], [0.4, 0.4, 0.4], [0.2, 0.2, 0.2]]

def get_colors(model_names):
    # I don't have time to make this good, but this will eventually want to be nice depending on model types and numbe
    colors = [[0.4, 0.5, 1], [1, 0.5, 0.4], [0.8, 0.8, 0.8], [0.6, 0.6, 0.6], [0.4, 0.4, 0.4], [0.2, 0.2, 0.2]]
    return colors


def get_model_names(p,fspec):
    ''' use the options to get a list of model names. 
    This is not going to be a very flexible function, 
    so will need to be updated with new model introductions'''

    first_file = np.load(glob.glob(os.path.join(p, fspec))[0])
    opts = first_file['options'].item()
    model_names = []
    if opts.glm_tgl:
        model_names.append('glm')

    if opts.gam_tgl:
        model_names.append('gam')

    if opts.stm_tgl:
        model_names.append('stm')

    if opts.conv_tgl:
        for ii in range(opts.max_num_conv):
            model_names.append('conv_{}_node'.format(ii+1))

    return model_names


def get_weights(models):
    '''take a npz file of data and 
    returns a dict of model weights. 
    '''

    weights = {}
    for name,model in models.iteritems():
        if type(model)==keras.models.Sequential:
            w = model.get_weights()[0]
            weights[name] = w
        elif type(model)==pygam.pygam.GAM:
            pass
        elif type(model)==statsmodels.genmod.generalized_linear_model.GLMResultsWrapper:
            weights[name] = model.params[1:]
        elif type(model)==cmt.models.STM:
            weights[name] = {}
            weights[name]['features'] = model.features
            weights[name]['weights'] = model.weights
            weights[name]['biases'] = model.biases
        else:
            print('model not recognized, skipping')

    return(weights)


def get_all_correlations(corrs,model_names,sigma_vals):
    '''returns a numpy array of correlation values between predicted spike rate and smoothed observed spike rate. 
    each row is a smoothing value, each column is a model
    '''
    rr = np.empty([len(sigma_vals), 0])
    for model in model_names:
        r = np.array(corrs[model])[:,np.newaxis]
        rr = np.append(rr,r,axis=1)
    return rr


def get_yhat(fid,model_names):
    '''returns a numpy array of predicted spike probability in a bin,
    each row is a time point, each column is a model's prediction
    '''
    yhat = np.empty([fid['y'].shape[0], 0])
    for model in model_names:
        yhat = np.append(yhat, fid['yhat'].item()[model][:, np.newaxis], axis=1)
    return yhat


def analyse_model(p,f,model_names,plot_tgl=False):
    ''' runs all the analyses we generally want from a particular cell
    INPUTS: 
        p:              path to where the model file is
        f:              filename of the model
        model_names:    list of model names we are evaluating, in the order they are to appear
        plot_tgl:       Boolean whether we want to plot this data now. Default False
    OUTPUTS:
        rr:             numpy aray of correlations for all sigma values ordered by model_names
        weights:        dict of model weights
        yhat:           predicted firing rate as a numpy array. Columns correspond to model names
        y:              Observed spiking
        B:              Basis functions used
    '''
    fid = np.load((os.path.join(p,f)))
    # get sigmas
    sigma_vals = fid['sigma_vals']

    # get bases:
    B = fid['B']
    # ============================= #
    # GET ACCURACIES
    # ============================= #
    corrs = fid['corrs'].item()
    rr = get_all_correlations(corrs, model_names, sigma_vals)

    if plot_tgl:
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

    # get weights
    models = fid['mdl'].item()
    weights = get_weights(models)

    yhat = get_yhat(fid,model_names)

    return rr,weights,yhat,fid['y'],B,fid['Cbool'],fid['options']

def concatenate_data(p,fspec,f_out,model_names):
    ''' creates a numpy file that carries all the cells modeled data
    INPUTS:
        p:              path to where the models live
        fspec:          wildcard expression to grab models with a particular filename type
        f_out:          name to put all the concatenated data in
        model_names:    list of model names to care about in order 
    OUTPUTS:
        none--saves a numpy array to '<fspec>'     
    '''
    
    
    all_rr = np.empty([50, len(model_names), 0])
    all_weights = []
    all_yhat = []
    all_y = []
    id = []
    all_Cbool = []
    all_opts = []
    B = None
    for f in glob.glob(os.path.join(p, fspec)):
        print(os.path.basename(f))
        rr, weights, yhat, y, B, Cbool, opts = analyse_model(p, f,model_names=model_names,plot_tgl=False)

        all_rr = np.concatenate([all_rr, rr[:, :, np.newaxis]], axis=-1)
        all_weights.append(weights)
        all_yhat.append(yhat)
        all_y.append(y)
        all_Cbool.append(Cbool)
        all_opts.append(opts)
        id.append(f[-15:-4])
    np.savez(os.path.join(p, f_out),
             rr=all_rr,
             weights=all_weights,
             yhat=all_yhat,
             y=all_y,
             bases=B,
             id=id,
             Cbool=all_Cbool,
             opts = all_opts,
             model_names=model_names
             )


def plot_summary_performance(f,colors=None):

    ''' Takes the summary file (which contains the data from all cells) and creates violin/swarm plots of the model accuracies and best smoothinig parameters.
        INPUTS:
            f:  full filename of the model summary file
        OUTPUTS:
            none--creates plots
    '''
    colors = get_colors(None)
    sigma_vals = np.arange(2,200,4)
    res = np.load(f)

    model_names = res['model_names']
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
    ax2.set_xlabel('Gaussian kernel sigma (ms)')
    ax2.set_title('Best smoothing kernel width')
    plt.tight_layout()

    fig3 = plt.figure()

    for ii in xrange(len(model_names)):
        sns.distplot(best_smoothing_vals[ii,:],20,kde=False)

    # performance and smoothing parameter
    for ii in xrange(len(model_names)):
        sns.jointplot(best_smoothing_vals[ii, :], max_rr[ii, :],edgecolor='w',marginal_kws=dict(bins=25), size=5, ratio=4, color=colors[ii]).plot_joint(sns.kdeplot,n_levels=3)
    plt.show()


def glm_PCA(f):
    '''Takes the summary file (which contains data from all cells) and plots the PCA space of the GLM weights
        INPUTS: 
            f:  full filename of the model summary file
        OUTPUTS:
            none--creates plots    
    '''
    
    res = np.load(f)
    colors = get_colors(None)

    B = res['bases']
    weights = res['weights']
    ww = np.empty([0,len(weights[0]['glm'])])
    for ii,cell in enumerate(weights):
        if ii==43:
            continue

        ww = np.concatenate([ww,cell['glm'][np.newaxis,:]],axis=0)
    PCA = sklearn.decomposition.PCA()
    yy = PCA.fit_transform(ww)
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(PCA.explained_variance_, 'k')
    ax.set_xlabel('n_component')
    ax.set_ylabel('Explained Variance')
    ax.set_facecolor([0., 0., 0., 0.2])

    ax3D = fig.add_subplot(122, projection='3d')
    plt.plot(yy[:, 0], yy[:, 1], yy[:, 2], 'k.')
    ax3D.set_xlabel('PC1')
    ax3D.set_ylabel('PC2')
    ax3D.set_zlabel('PC3')

    ax3D.w_xaxis.set_pane_color((0., 0., 0., 0.2))
    ax3D.w_yaxis.set_pane_color((0., 0., 0., 0.2))
    ax3D.w_zaxis.set_pane_color((0., 0., 0., 0.2))
    ax3D.patch.set_facecolor('w')

    fig.suptitle('GLM filter PC space')
    nComp=3
    What = np.dot(yy[:, :nComp], PCA.components_[:nComp, :])
    filters=[]
    for cell in enumerate(weights):
        if ii == 43:
            continue
        filters.append(map_bases(What[ii,:],B)[0])
    # first three filter components
    comps = PCA.components_[:nComp, :]
    fig = plt.figure()
    for ii in xrange(nComp):
        principle_filter=map_bases(comps[ii,:],B)[0]
        ax = fig.add_subplot(3,1,ii+1)
        plt.plot(np.arange(-principle_filter.shape[0],0),np.flipud(principle_filter))
        ax.set_title('Component {}'.format(ii+1))

    ax.legend(['M_x','M_y','M_z','F_x','F_y','F_z'],frameon=True,loc='center left',bbox_to_anchor=(0, .5),framealpha=0.8,facecolor='w')
    plt.tight_layout()


def plot_weights(weights,model_names,B,rr,id,sigma_vals,f_out=None):
    '''
    This is a more complicated script which plots the weights and model performance for a cell in one big figure. 
    This will likely not be run in a general sense yet, and I do not have time now to make it generalized.

    '''
    colors = get_colors(None)


    fig = plt.figure()
    col=0
    row=0
    for model in model_names:
        if model=='gam':
            continue
        w_mat = weights[model]
        if model=='glm':
            w_mat = map_bases(w_mat,B)[0]
            ax = plt.subplot2grid((4, 5), (0, 0))
            xx = np.arange(-w_mat.shape[0],0)
            plt.plot(xx,np.flipud(w_mat))
            ax.set_title(model)
            col+=1
        else:
            for ii in xrange(w_mat.shape[-1]):
                ax = plt.subplot2grid((4, 5), (ii, col))
                xx = np.arange(-w_mat.shape[0], 0)
                ax.plot(xx,np.flipud(w_mat[:,:,ii]))
                if ii==0:
                    ax.set_title(model)
            col+=1

    ax = plt.subplot2grid((4,5),(2,0),rowspan=2,colspan=2)
    for ii in xrange(rr.shape[1]):
        plt.plot(sigma_vals, rr[:,ii], color=colors[ii])
    ax.legend(model_names)
    ax.set_title('Performance of {}'.format(id))
    ax.set_ylim(-0.1, 1)
    ax.set_ylabel('Pearson correlation')
    ax.set_xlabel('Box width (ms)')

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.pause(0.01)
    if f_out is not None:
        plt.savefig(f_out,dpi=300)
    plt.close('all')


def batch_weight_plots(f):
    '''Calls plot weights for each cell in the summary file
    '''
    
    fid = np.load(f)
    bname = os.path.splitext(os.path.basename(f))[0]
    model_names = fid['model_names']
    p_save = os.path.split(f_in)[0]
    sigma_vals = np.arange(2,200,4)
    for ii in xrange(len(fid['id'])):
        weights = fid['weights'][ii]
        B = make_bases(5,[0,15],2)
        rr = fid['rr'][:,:,ii]
        id = fid['id'][ii]
        f_out = os.path.join(p_save,'{}_weights_{}.png'.format(bname,id))
        plot_weights(weights,model_names,B,rr,id,sigma_vals,f_out)


if __name__=='__main__':    
    f_in = sys.argv[1]
    plot_summary_performance(f_in)
    batch_weight_plots(f_in)
    glm_PCA(f_in)