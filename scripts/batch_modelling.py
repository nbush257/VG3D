from neo.io import PickleIO as PIO
import os
import sys
VG3D_modules = os.path.join(os.path.abspath(os.path.join(os.getcwd(),os.pardir)),'modules')
sys.path.append(VG3D_modules)
from neo_utils import *
from mechanics import *
from GLM import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if sys.version_info.major==3:
    import pickle
else:
    import cPickle as pickle
import statsmodels.api as sm
import elephant
import pygam
import glob
from optparse import OptionParser
from sklearn.preprocessing import RobustScaler,StandardScaler
sns.set()

def init_model_params():
    sigma_vals = np.arange(2, 200, 4)
    B = make_bases(5, [0, 15], b=2)
    winsize = int(B[0].shape[0])
    return sigma_vals,B,winsize

def create_design_matrix(blk,varlist,deriv_tgl=False,bases=None):
    ''' 
    Takes a list of variables and turns it into a matrix.
    Sets the non-contact mechanics to zero, but keeps all the kinematics as NaN
    You can append the derivative or apply the pillow bases, or both.
    Scales, but does not center the output
    '''
    X = []


    Cbool = get_Cbool(blk)

    # ================================ #
    # GET THE CONCATENATED DESIGN MATRIX OF REQUESTED VARS
    # ================================ #
    for varname in varlist:
        var = get_var(blk,varname, keep_neo=False)[0]
        if varname in ['M','F']:
            var[np.invert(Cbool),:]=0
            var = replace_NaNs(var,'pchip')
            var = replace_NaNs(var,'interp')

        X.append(var)
    X = np.concatenate(X,axis=1)

    # ================================ #
    # CALCULATE DERIVATIVE
    # ================================ #
    if deriv_tgl:
         Xdot = get_deriv(X)
         X = np.append(X,Xdot,axis=1)

     # ================================ #
     # APPLY BASES FUNCTIONS
     # ================================ #
    if bases is not None:
        X = apply_bases(X,bases)

    # ================================ #
    # SCALE
    # ================================ #
    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(X)
    return X

def optarg_list(option,opt,value,parser):
    ''' Parses a comma seperated list of variables to include in the model'''
    setattr(parser.values,option.dest,value.split(','))


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
                      dest='deriv_tgl',
                      default=False,
                      help='Derivative toggle, set to true to include the derivative in the model')
    parser.add_option('-P','--pillow_tgl',
                      action='store_true',
                      dest='pillow_tgl',
                      default=False,
                      help='Basis toggle, set to true to map the inputs to a pillow basis')
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
    parser.add_option('-w','--conv_window',
                      dest='conv_window',
                      default=10,
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
    conv_window = options.conv_window
    max_num_conv = options.max_num_conv
    l2_penalty = options.l2_penalty

    # Get desired filenames
    fname = args[0]
    p_save = os.path.split(fname)[0]
    print(os.path.basename(fname))

    # read data in
    fid = PIO(fname)
    blk = fid.read_block()

    # set binsize to a quantity
    binsize = binsize*pq.ms

    # initialize parameters
    sigma_vals, B, winsize = init_model_params()

    # calculate pillow bases if desired.
    if pillow_tgl:
        B = make_bases(5,[0,15],2)
        bases=B[0]
    else:
        B=None
        bases = None

    # calculate the design matrices based on input toggles
    X = create_design_matrix(blk,varlist,deriv_tgl=deriv_tgl,bases=bases)


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
            continue

        # ===================================== #
        # GET SPIKE TIMES
        # CONVERT TO BINNED SPIKE TRAIN
        # ===================================== #
        sp = concatenate_sp(blk)[unit.name]
        b = elephant.conversion.BinnedSpikeTrain(sp,binsize=binsize)
        Cbool=get_Cbool(blk)

        spike_isbool=binsize==pq.ms
        if spike_isbool:
            y = b.to_bool_array().ravel().astype('float32')
        else:
            y = b.to_array().ravel().astype('float32')

        # ===================================== #
        # MAKE TENSOR FOR CONV NETS
        # ===================================== #
        Xt = make_binned_tensor(X, b, window_size=conv_window)

        # ===================================== #
        # RUN ALL THE MODELS REQUESTED
        # ===================================== #
        if pillow_tgl:
            yhat['glm'],mdl['glm'] = run_GLM(X,y)
            weights['glm'] = mdl.params

        if gam_tgl:
            yhat['gam'],mdl['gam'] = run_GAM(X,y)

        if conv_tgl:
            for num_filters in range(1,max_num_conv+1):
                mdl_name = 'conv_{}_node'.format(num_filters)
                yhat[mdl_name],mdl[mdl_name]=conv_model(Xt,y[:,np.newaxis,np.newaxis],num_filters=num_filters,winsize=conv_window,is_bool=spike_isbool,l2_penalty=l2_penalty)
                weights[mdl_name] = mdl[mdl_name].get_weights()[0]

        # ===================================== #
        # EVALUATE ALL THE MODELS -- THIS MAY NEED TO BE ALTERED
        # ===================================== #

        for model in yhat.iterkeys():
            corrs[model] = evaluate_correlation(yhat[model],sp,Cbool=Cbool,sigma_vals=sigma_vals)
        # ===================================== #
        # PLOT IF REQUESTED
        # ===================================== #
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
                 B=B)

if __name__=='__main__':
    main()
