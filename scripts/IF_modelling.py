import sys
import spikeAnalysis
import quantities as pq
import neo
import elephant
import spikeAnalysis
import neoUtils
import os
import glob
import GLM
import numpy as np
import scipy
import sklearn
import matplotlib.pyplot as plt

"""
This module is based on Dong,..Bensmaia et al. 2013
"""

def get_X_y(fname,p_smooth,unit_num=0):
    varlist = ['M', 'F', 'TH', 'PHIE']
    blk = neoUtils.get_blk(fname)
    blk_smooth = GLM.get_blk_smooth(fname,p_smooth)

    cbool = neoUtils.get_Cbool(blk)
    X = GLM.create_design_matrix(blk,varlist)
    Xdot,Xsmooth = GLM.get_deriv(blk,blk_smooth,varlist,[9])

    X = np.concatenate([X,Xdot],axis=1)
    X = neoUtils.replace_NaNs(X,'pchip')
    X = neoUtils.replace_NaNs(X,'interp')
    scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
    X = scaler.fit_transform(X)

    y = neoUtils.get_rate_b(blk,unit_num)[1][:,np.newaxis]
    y[np.invert(cbool)]=0
    return(X,y,cbool)


def calc_IInj(X,K):
    if K.ndim==1:
        return(np.dot(X,K))
    else:
        aa = np.dot(K,X.T)
        aa[aa<0]=0
        return(np.sum(aa,0))


def saturation(I,I0):
    """ Caclulate the saturation of the injected current"""
    return(np.divide(
        np.multiply(I,I0),(np.abs(I)+I0)
    ))


def cost_function(y,yhat,tau=5*pq.ms):
    # DO we want to compute the average cost per deflection?
    if type(y) is not neo.core.spiketrain.SpikeTrain:
        y = spikeAnalysis.binary_to_neo_train(y)
    if type(yhat) is not neo.core.spiketrain.SpikeTrain:
        yhat = spikeAnalysis.binary_to_neo_train(yhat)


    D = elephant.spike_train_dissimilarity.van_rossum_dist([y,yhat],tau)[0,1]
    return(D)


def init_free_params(X,nfilts):
    params = {'K':np.random.uniform(-1.,1.,[nfilts,X.shape[1]]),#free
              'tau':5.,#free (ms) #as tau incresaes, the neuron integrates more over time
              'A0':-1e4,#free (current nA)
              'A1':-1e4,#free (current nA?)
              'a':1e-3,#free
              'I0':1., #free
              }
    return(params)


def init_constrained_params():
    params = {
        'b':10e-3,# 1/s
        'C':150.,#pF
        'Vrest':-70.,#mV
        'THETA_inf':-30., # mV
    }

    return(params)

def run_IF(I_inj,free_params,const_params):
    """

    :param I_inj: The injected current due to the stimulus and the weighting function
    :param free_params: a dict of the free paramters
    :param const_params: a dict of the constrained parameters
    :return:
    """
    T = len(I_inj)
    I_ind = np.zeros_like(I_inj)
    i0 = np.zeros_like(I_inj)
    i1 = np.zeros_like(I_inj)
    tau_0 = 5.
    tau_1 = 50.

    V = np.ones_like(I_inj)*const_params['Vrest']
    THETA = np.ones_like(I_inj)*const_params['THETA_inf']
    spikes=[]
    aa=[]
    for t in range(1,T):

        #calculate Iind
        di0 = -i0[t-1]/tau_0
        di1 = -i1[t-1]/tau_1

        # update spike induced currents
        i0[t] = i0[t-1]+di0
        i1[t] = i1[t-1]+di1
        I_ind[t] = i0[t]+i1[t]

        # V'(t) = -1/tau[V(t)-V_rest]+(I(t)+I_ind(t))/C)
        dV = (-1./free_params['tau'])*(V[t-1]-const_params['Vrest']) + (I_inj[t-1]+I_ind[t-1])/const_params['C']
        # THETA'(t) = a[V(t)-V_rest]-b[THETA(t)-THETA_inf]
        dTHETA = free_params['a']*(V[t-1]-const_params['Vrest']) - const_params['b']*(THETA[t-1] - const_params['THETA_inf'])
        aa.append(dTHETA)

        # update voltge
        V[t] = V[t-1]+dV
        # update Threshold?
        THETA[t] = THETA[t-1] + dTHETA

        # If exceed threshold
        if V[t]>THETA[t]:
            i0[t] = i0[t]+free_params['A0']
            i1[t] = i1[t]+free_params['A1']
            V[t] = const_params['Vrest']
            THETA[t] = np.max([THETA[t],const_params['THETA_inf']])
            spikes.append(t)


    spikes = neo.core.spiketrain.SpikeTrain(spikes,t_stop=T*pq.ms,units=pq.ms)
    return(V,spikes,THETA,I_ind)


def optim_func(free_params,X,y,nfilts,const_params,plot_tgl=False):
    """
    This function is passed into the optimization
    in order to find the optimal values of the free parameters of the model

    Runs an IF simulation on the observed data to simulatie a spike train, and
    uses the Van Rossum distance as the cost function

    :param free_params: a list of all the free parameters of the model.
    :param X: an array of time x dimensions
    :param y: a target binary spike train

    :return: the value of the objective function using the current value of the parameters
    """

    # map free params to dict
    free_params_d = convert_free_params(free_params,X,nfilts)

    # calculate saturated injected current
    Iinj = calc_IInj(X,free_params_d['K'])
    Isat = saturation(Iinj,free_params_d['I0'])*10000

    # choose a contact to simulate
    onsets = np.where(np.diff(cbool.astype('int'))==1)[0]+1
    offsets = np.where(np.diff(cbool.astype('int'))==-1)[0]
    idx = np.random.randint(len(onsets))



    # run IF model
    V,spikes = run_IF(Isat[onsets[idx]:offsets[idx]],free_params_d,const_params)[:2]
    if plot_tgl:
        plt.close('all')
        plt.plot(V)
        plt.show()
        plt.pause(0.1)

    # calculate cost
    cost = cost_function(y[onsets[idx]:offsets[idx]],spikes,tau=5*pq.ms)
    print('Cost is: {:0.4f}'.format(cost))
    # print(free_params)

    return(cost)



def convert_free_params(params,X,nfilts,out_type='dict'):
    """
    A utility function that freely converts the free parameters between
    a list and a dictionary as hardcoded in the parameter list

    :param params: a list or a dictionary of the parameters
    :param X: the design matrix, gives us the dimensions of K
    :param [out_type]: optional, defaults to dictionary output. Switch between what
                        kind of output type is desired. Can be: 'dict','list'

    :return out_params: the converted parameter data structure
    """
    param_list = ['K','tau','A0','A1','I0','a']
    ndims = X.shape[1]*nfilts
    # convert a list to a dictionary
    if out_type=='dict':
        if type(params) is dict:
            print('params is already a dictionary, skipping...')
            return(params)
        else:
            out_params = {}
            for ii,param in enumerate(param_list):
                if ii==0:
                    out_params[param] = np.array([x for x in params[:ndims]])
                else:
                    out_params[param] = params[ndims+ii-1]
        out_params['K'] = out_params['K'].reshape([nfilts,-1])

    # convert a dictionary to a list
    elif out_type=='list':
        if type(params) is list:
            print('params is already a list, skipping...')
            return(params)
        out_params= []
        for param in param_list:
           if type(params[param])==np.ndarray:
               p = params[param].ravel()
               for x in p:
                   out_params.append(x)
           else:
               out_params.append(params[param])
    else:
        raise ValueError("out_type is not a valid keyword. Use 'dict' or 'list'")

    return(out_params)


def test_fake_input():
    I = 1. # in nA?
    I_inj = np.ones(10000)*I
    X = np.empty([10,10])
    params = init_params(X,1)
    V,spikes=run_IF(I_inj,params)


def test_real_data():
    nfilts=3
    p_smooth = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\smooth')
    fname = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\rat2017_08_FEB15_VG_D1_NEO.h5')
    X,y,cbool = get_X_y(fname,p_smooth)
    free_params = init_free_params(X,nfilts)
    free_params = convert_free_params(free_params,X,nfilts,out_type='list')
    const_params = init_constrained_params()

    print('Beginning Optimization...')

    sol = scipy.optimize.minimize(optim_func,
                            x0=free_params,
                            args=(X,y,nfilts,const_params),
                            method='COBYLA',
                            )
def init_constraints(free_params,X,nfilts):
    constraints = []
    for factor in range(len(free_params)):
        if factor==nfilts*X.shape[1]:
            l = {'type':'ineq','fun':lambda x:x}
            constraints.append(l)
        elif factor==(nfilts*X.shape[1]+1):
            l = {'type':'ineq','fun':lambda x:-x}
            constraints.append(l)
        elif factor==(nfilts*X.shape[1]+2):
            l = {'type':'ineq','fun':lambda x:-x}
            constraints.append(l)
        else:
            l = {'type':'ineq','fun':lambda x:np.abs(x)}
            constraints.append(l)
    return(constraints)

def main(fname,p_smooth,nfilts=3):
    print('loading in {}'.format(fname))
    blk = neoUtils.get_blk(fname)
    save_dir = os.path.split(fname)[0]
    num_units = len(blk.channel_indexes[-1].units)
    for unit_num in range(num_units):
        id = neoUtils.get_root(blk,unit_num)
        save_file = os.path.join(save_dir,'{}_IF_model.npz'.format(id))
        if os.path.isfile(save_file):
            print('File already found. Aborting...')
        X,y,cbool = get_X_y(fname,p_smooth,unit_num=unit_num)
        free_params = init_free_params(X,nfilts)
        free_params = convert_free_params(free_params,X,nfilts,out_type='list')
        const_params = init_constrained_params()
        cons = init_constraints(free_params,X,nfilts)


        print('Beginning Optimization...')

        solution = scipy.optimize.minimize(optim_func,
                                x0=free_params,
                                args=(X,y,nfilts,const_params),
                                method='COBYLA',
                                constraints=cons,
                                )
        np.savez(save_file,
                 X=X,
                 y=y,
                 cbool=cbool,
                 solution=solution,
                 free_params=free_params,
                 const_params=const_params,
                 nfilts=nfilts)


def analyze_result(fname):
    dat = np.load(fname)
    sol = dat['solution'].item()
    X = dat['X']
    y = dat['y']
    nfilts = dat['nfilts']
    const_params = dat['const_params'].item()
    x = sol.x
    free_params = convert_free_params(x,X,nfilts)
    I = calc_IInj(X,free_params['K'])
    Iinj = saturation(I,free_params['I0'])*10000
    V,spikes,THETA,I_ind = run_IF(Iinj,free_params,const_params)
    yhat = np.zeros_like(y)
    yhat[spikes.times.magnitude.astype('int')]=1
    return(y,yhat)

if __name__=='__main__':
    fname = sys.argv[1]
    p_smooth = sys.argv[2]
    if len(sys.argv)==4:
        nfilts=sys.argv[3]
    else:
        nfilts=3
    print('File in: {}'.format(fname))
    print('p_smooth: {}'.format(p_smooth))
    main(fname,p_smooth,nfilts)



