from neo.io import PickleIO
from neo.io import NixIO
import neoUtils
import quantities as pq
import os
import sys
VG3D_modules = os.path.join(os.path.abspath(os.path.join(os.getcwd(),os.pardir)),'modules')
sys.path.append(VG3D_modules)
import neoUtils
import GLM
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

if sys.version_info.major==3:
    import pickle
else:
    import cPickle as pickle
import elephant
import glob
from optparse import OptionParser
from sklearn.preprocessing import RobustScaler,StandardScaler
sns.set()

def init_model_params():
    sigma_vals = np.arange(2, 200, 4)*pq.ms
    B = GLM.make_bases(5, [0, 15], b=2)
    winsize = int(B[0].shape[0])
    return sigma_vals,B,winsize


def optarg_list(option,opt,value,parser):
    ''' Parses a comma seperated list of variables to include in the model'''
    setattr(parser.values,option.dest,value.split(','))
def bin_model(X,y,cbool,binsize):
    """
    This model will attempt to predict the number of spikes in a given bin that is larger than 1ms
    :param X: The full design matrix, binning is done here.
                - Assumes you have all covariates in X
                - Assumes you have NOT included spike history in X
    :param y: A neo spiketrain. Eventually will want to extend to allow for boolean
    :return:
    """
    # time zero will inform us about the proceeding spikes between time 0 and 1, which is why there is an extra index
    # in Xbin

    Xbin = GLM.bin_design_matrix(X,binsize=binsize)[:-1]
    scaler = StandardScaler(with_mean=False)
    Xbin = scaler.fit_transform(Xbin)
    cbool= GLM.bin_design_matrix(cbool[:,np.newaxis],binsize=binsize)[:-1].ravel()
    ybin = elephant.conversion.BinnedSpikeTrain(y,binsize=binsize*pq.ms).to_array().T.astype('f8')

    # add history
    Xbin = GLM.add_spike_history(Xbin,ybin,-1) # no basis
    # prep model
    yhat = np.zeros(ybin.shape[0])
    # TODO run stm
    yhat_out,mdl = GLM.run_GLM(Xbin[cbool,:],ybin[cbool,:],family=sm.families.Poisson,link=sm.genmod.families.links.log)
    yhat[cbool] = yhat_out.ravel()

def continuous_model(X,y,cbool):


def main():
    # ================================ #
    # SET UP OPTION PARSER
    # ================================ #
    usage = "usage: %prog filename [options]"
    parser = OptionParser(usage)
    parser.add_option('-p','--prefix',
                      dest='prefix',
                      default='model_results',
                      type=str,
                      help='prefix to append to the results filename')
    parser.add_option('-v','--varlist',
                      dest='varlist',
                      default='M',
                      type=str,
                      help='list of strings which indicate which variables to include in the model')
    parser.add_option('-b','--binsize',
                      dest='binsize',
                      default=1,
                      type=int,
                      help='number of milliseconds to bin the spikes.')
    parser.add_option('-D','--deriv_tgl',
                      action='store_true',
                      dest='deriv_tgl',
                      default=False,
                      help='Derivative toggle, set to true to include the derivative in the model')
    parser.add_option('-P','--pillow_tgl',
                      action='store_true',
                      dest='pillow_tgl',
                      default=False,
                      help='Basis toggle, set to true to map the inputs to a pillow basis \nfor use in the GLM only at this time')
    parser.add_option('--GLM',
                      action='store_true',
                      dest='glm_tgl',
                      default=False,
                      help='Toggles a GLM model. \nIf pillow toggle is false, takes an input where each point in the windowsize is its own dimension ')
    parser.add_option('--GAM',
                      action='store_true',
                      dest='gam_tgl',
                      default=False,
                      help='GAM toggle, call the flag to run a GAM')
    parser.add_option('-C','--conv_tgl',
                      action='store_true',
                      dest='conv_tgl',
                      default=False,
                      help='Convolutional network toggle. Call the flag to run a convolutional network')
    parser.add_option('--plot_tgl',
                      action='store_true',
                      dest='plot_tgl',
                      default=False,
                      help='Plot toggle, call to plot the results during the run. This should never be called on quest.')
    parser.add_option('-w','--window',
                      dest='window',
                      default=1,
                      type=int,
                      help='Window into the past to set the convolutional window to look in ms')
    parser.add_option('-n','--num_conv',
                      dest='max_num_conv',
                      default=4,
                      type=int,
                      help='Max number of convolutional nodes to use')
    parser.add_option('--l2',
                      dest='l2_penalty',
                      default=1e-6,
                      type=float,
                      help='l2 penalty')
    parser.add_option('-k','--kernel',
                      dest='kernel_mode',
                      default='gaussian',
                      type=str,
                      help='Kernel Mode (\'box\',\'gaussian\',\'exp\',\'alpha\',\'epan\')')
    parser.add_option('--STM','--STM_tgl',
                      action='store_true',
                      dest='stm_tgl',
                      default=False,
                      help='STM toggle. Call the flag to run a STM network (Thies 2013)')
    parser.add_option('--num_stm_components',
                      action='store',
                      dest='num_stm_components',
                      default=3,
                      type=int,
                      help='Number of components to use in the STM model')
    parser.add_option('--num_stm_features',
                      action='store',
                      dest='num_stm_features',
                      default=20,
                      type=int,
                      help='Number of features to use in the STM model')
    parser.add_option('--silence_noncontact',
                      action='store_true',
                      dest='silence_noncontact',
                      default=False,
                      help='If called, sets all spiking that occurs during non_contact to zero')

    (options,args)=parser.parse_args()
    if len(args)<1:
        parser.error('Need to pass a filename first')

    # map options
    plot_tgl = options.plot_tgl
    pillow_tgl = options.pillow_tgl
    varlist = options.varlist.split(',')
    conv_tgl = options.conv_tgl
    gam_tgl = options.gam_tgl
    binsize = options.binsize
    deriv_tgl = options.deriv_tgl
    prefix = options.prefix
    max_num_conv = options.max_num_conv
    l2_penalty = options.l2_penalty
    kernel_mode = options.kernel_mode




    # Get desired filenames
    fname = args[0]
    p_save = os.path.join(os.path.split(fname)[0],'results')
    print(os.path.basename(fname))

    # read data in
    fid = neo.io.NixIO(fname)
    blk = fid.read_block()

    # set binsize to a quantity
    binsize = binsize*pq.ms

    # initialize parameters
    sigma_vals, B, winsize = init_model_params()

    # calculate the design matrices based on input toggles
    X = create_design_matrix(blk, varlist, window=options.window, binsize=options.binsize,deriv_tgl=deriv_tgl, bases=None)


    # calculate pillow bases if desired.
    if pillow_tgl:
        B = GLM.make_bases(5,[0,15],2)
        bases=B[0]
        X_pillow = create_design_matrix(blk, varlist, deriv_tgl=options.deriv_tgl, bases=bases)
    else:
        B=None
        bases = None
        X_pillow = X


    for unit in blk.channel_indexes[-1].units:
        # ===================================== #
        # INIT OUTPUTS
        # ===================================== #
        yhat={}
        mdl={}
        corrs={}
        weights={}

        id =get_root(blk,int(unit.name[-1]))
        f_save = os.path.join(p_save, '{}_{}.npz'.format(prefix,id))
        if os.path.isfile(f_save):
            raise Warning('Output file found. Skipping {}'.format(id))
            continue

        # ===================================== #
        # GET SPIKE TIMES
        # CONVERT TO BINNED SPIKE TRAIN
        # ===================================== #
        sp = concatenate_sp(blk)[unit.name]
        b = elephant.conversion.BinnedSpikeTrain(sp,binsize=binsize)
        Cbool=get_Cbool(blk,-1)

        spike_isbool=binsize==pq.ms
        if spike_isbool:
            y = b.to_bool_array().ravel().astype('float32')
        else:
            y = b.to_array().ravel().astype('float32')

        if options.silence_noncontact:
            y[np.invert(Cbool)] = 0
        # ===================================== #
        # MAKE TENSOR FOR CONV NETS
        # ===================================== #
        Xt = create_design_matrix(blk, varlist, window=1, deriv_tgl=deriv_tgl,
                                 bases=None)
        Xt = make_binned_tensor(Xt, b, window_size=options.window)

        # ===================================== #
        # RUN ALL THE MODELS REQUESTED
        # ===================================== #
        if options.glm_tgl:
            if pillow_tgl:
                yhat['glm'],mdl['glm'] = run_GLM(X_pillow, y)
                weights['glm'] = mdl['glm'].params
            else:
                yhat['glm'], mdl['glm'] = run_GLM(X, y)
                weights['glm'] = mdl['glm'].params
        if gam_tgl:
            yhat['gam'],mdl['gam'] = run_GAM(X, y)

        if conv_tgl:
            for num_filters in range(1,max_num_conv+1):
                mdl_name = 'conv_{}_node'.format(num_filters)
                yhat[mdl_name],mdl[mdl_name]=conv_model(Xt, y[:, np.newaxis, np.newaxis],
                                                        num_filters=num_filters,
                                                        winsize=options.window,
                                                        is_bool=spike_isbool,
                                                        l2_penalty=l2_penalty
                                                        )
                weights[mdl_name] = mdl[mdl_name].get_weights()[0]

        if options.stm_tgl:
            yhat['stm'], mdl['stm'] = run_STM(X, y,
                                          num_components=options.num_stm_components,
                                          num_features=options.num_stm_features)


        # ===================================== #
        # EVALUATE ALL THE MODELS -- THIS MAY NEED TO BE ALTERED
        # ===================================== #

        for model in yhat.iterkeys():
            corrs[model] = evaluate_correlation(yhat[model],sp,kernel_mode=kernel_mode,Cbool=Cbool,sigma_vals=sigma_vals)
        # ===================================== #
        # PLOT IF REQUESTED
        # ===================================== #
        if plot_tgl:
            for model in yhat.iterkeys():
                plt.plot(sigma_vals,corrs[model])

            ax = plt.gca()
            ax.set_ylim(-0.1,1)
            ax.legend(corrs.keys())
            ax.set_xlabel('{} Rate Kernel Sigma'.format(options.kernel_mode))
            ax.set_ylabel('Pearson Correlation')
            ax.set_title(id)
            plt.savefig(os.path.join(p_save,'performance_{}_{}.svg'.format(options.prefix,id)))
            plt.close('all')

        # ===================================== #
        # SAVE THE MODEL OUTPUTS
        # ===================================== #

        np.savez(f_save,
                 corrs=corrs,
                 yhat=yhat,
                 sigma_vals=sigma_vals,
                 mdl=mdl,
                 y=y,
                 X=X,
                 Cbool=Cbool,
                 options=options,
                 B=B)

if __name__=='__main__':
    main()
