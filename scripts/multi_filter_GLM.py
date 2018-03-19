import sklearn
import theano
import neoUtils
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from GLM import get_blk_smooth
import GLM

def get_X_y(fname,p_smooth,unit_num,pca_tgl=False,n_pcs=6):
    varlist = ['M', 'F', 'TH', 'PHIE']
    blk = neoUtils.get_blk(fname)
    blk_smooth = get_blk_smooth(fname,p_smooth)

    cbool = neoUtils.get_Cbool(blk)
    X = GLM.create_design_matrix(blk,varlist)
    Xdot,Xsmooth = GLM.get_deriv(blk,blk_smooth,varlist,[9])
    # if using the PCA decomposition of the inputs:
    if pca_tgl:

        X = neoUtils.replace_NaNs(X,'pchip')
        X = neoUtils.replace_NaNs(X,'interp')

        Xsmooth = neoUtils.replace_NaNs(Xsmooth,'pchip')
        Xsmooth = neoUtils.replace_NaNs(Xsmooth,'interp')

        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        X = scaler.fit_transform(X)

        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        Xsmooth = scaler.fit_transform(Xsmooth)

        pca = sklearn.decomposition.PCA()
        X_pc = pca.fit_transform(X)[:,:n_pcs]
        pca = sklearn.decomposition.PCA()
        Xs_pc = pca.fit_transform(Xsmooth)[:,:n_pcs]
        zero_pad = np.zeros([1,n_pcs])
        Xd_pc = np.diff(np.concatenate([zero_pad,Xs_pc],axis=0),axis=0)
        X = np.concatenate([X_pc,Xd_pc],axis=1)

        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        X = scaler.fit_transform(X)
    else:
        X = np.concatenate([X,Xdot],axis=1)
        X = neoUtils.replace_NaNs(X,'pchip')
        X = neoUtils.replace_NaNs(X,'interp')
        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        X = scaler.fit_transform(X)


    y = neoUtils.get_rate_b(blk,unit_num)[1][:,np.newaxis]
    yhat = np.zeros_like(y)
    return(X,y,cbool)
def logit(x):
    return(tf.divide(x,(1-tf.log(x))))

def neglogliklihood(z,y):
    z = tf.reshape(z,[-1,1])
    cost = -tf.matmul(tf.transpose(y),tf.log(z))+tf.reduce_sum(z)
    return(cost)

def neglogliklihood_bernoulli(z,y):
    z = tf.reshape(z,[-1,1])
    L = tf.matmul(tf.transpose(y),logit(z))-tf.matmul(tf.transpose((1-y)),logit(1-z))
    cost = -L
    return(cost)

def X_to_pillow(X):
    B = GLM.make_bases(5,[0,10])
    Xb = GLM.apply_bases(X,B[0])
    scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
    return(scaler.fit_transform(Xb))

def build_GLM_model(Xraw,yraw,savefile, nfilts=4,hist=False,learning_rate=1e-5,epochs=100,batch_size=256,family='p'):
    if batch_size is None:
        batch_size=Xraw.shape[0]
    if hist:
        B = GLM.make_bases(8,[0,25],1)
        yhistraw = GLM.add_spike_history(Xraw,yraw,B)[:,Xraw.shape[1]:]

    # make data a multiple of batchsize and batch it
    n_del = Xraw.shape[0] % batch_size
    X = Xraw[n_del:,:]
    y = yraw[n_del:]
    n_batches = X.shape[0]/batch_size
    batched_x = np.split(X,n_batches)
    batched_y = np.split(y,n_batches)
    if hist:
        yhist = yhistraw[n_del:,:]
        batched_yhist = np.split(yhist,n_batches)

    # init vars
    mdl_input = tf.placeholder(tf.float32,[None,X.shape[1]])
    mdl_output = tf.placeholder(tf.float32,[None,1])

    if hist:
        mdl_yhist = tf.placeholder(tf.float32,[None,yhist.shape[1]])
    # init weights
    if hist:
        H = tf.Variable(tf.zeros([yhist.shape[1],1]),name='HistoryFilters')
        tf.add_to_collection('H',H)

    K = tf.Variable(
        tf.random_normal([X.shape[1],nfilts], stddev=0.003),
        name='StimFilters')
    tf.add_to_collection('K',K)

    b = tf.Variable(
        tf.random_normal([1]),
        name = 'bias')
    tf.add_to_collection('b',b)

    #### The model ###
    # Hidden Layer
    hidden_out = tf.matmul(mdl_input,K)
    hidden_out = tf.nn.relu(hidden_out)

    Ksum = tf.reduce_sum(hidden_out,axis=1)
    if hist:
        H = tf.clip_by_value(H,-np.inf,0.)
        Ksum =tf.add(tf.squeeze(tf.matmul(mdl_yhist,H)),Ksum)
    Ksum = tf.add(Ksum,b)

    # define cost function as negative log liklihood of Poisson spiking
    if family=='p':
        conditional_intensity = tf.exp(Ksum)
        cost = neglogliklihood(conditional_intensity,mdl_output)
    elif family=='b':
        conditional_intensity = tf.sigmoid(Ksum)
        #cost = tf.reduce_mean(-tf.reduce_sum(mdl_output*conditional_intensity), reduction_indices=1)
        cost = neglogliklihood_bernoulli(conditional_intensity,mdl_output)


    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    # loop over entire dataset multiple times
    for epoch in range(epochs):
        # loop over sub_batches
        avg_cost = 0.
        for ii in range(n_batches):
            if hist:
                _,c = sess.run([optimizer,cost],
                         feed_dict={mdl_input:batched_x[ii],mdl_output:batched_y[ii],mdl_yhist:batched_yhist[ii]})
            else:
                _,c = sess.run([optimizer,cost],
                         feed_dict={mdl_input:batched_x[ii],mdl_output:batched_y[ii]})
            avg_cost +=c/n_batches
        print('Epoch:{}\t, Cost={}'.format(epoch,avg_cost))
    print('Done!')
    print('saving to {}'.format(savefile))
    saver = tf.train.Saver()
    saver.save(sess,savefile)
    print('Saved session to {}'.format(savefile))

def run_model(fname,p_smooth,unit_num,savepath,param_dict):
    X,y,cbool = get_X_y(fname,p_smooth,unit_num)
    blk = neoUtils.get_blk(fname)
    root = neoUtils.get_root(blk,unit_num)
    savefile = os.path.join(savepath,'{}_tensorflow.ckpt'.format(root))
    X[np.invert(cbool),:] = 0
    y[np.invert(cbool),:] = 0
    Xb = X_to_pillow(X[:,:8])
    print(param_dict)
    build_GLM_model(Xb,y,savefile, **param_dict)

def simulate(X,y,p_model):
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(p_model+'.meta')
    new_saver.restore(sess,p_model)
    K = tf.get_collection('K')[0]
    H = tf.get_collection('H')[0]
    b = tf.get_collection('b')[0]
    K = sess.run(K)
    H = sess.run(H)
    b = sess.run(b)    
    stim_curr = np.sum(np.dot(X,K),axis=1)+b
    #TODO: here we iterate through the time step to get the individual spikes
#    as in the pillow code 
    sess.close()

if __name__=='__main__': 
    fname =  '/media/nbush/Dante/Users/NBUSH/Box Sync/Box Sync/__VG3D/_deflection_trials/_NEO/rat2017_08_FEB15_VG_B3_NEO.h5'
    p_smooth = '/media/nbush/Dante/Users/NBUSH/Box Sync/Box Sync/__VG3D/_deflection_trials/_NEO/smooth'
    savepath = '/home/nbush/Desktop/models'
    param_dict={'family':'p',
                'hist':True,
                'nfilts':4,
                'learning_rate':1e-8,
                'batch_size':None,
                'epochs':100}
    run_model(fname,p_smooth,0,savepath,param_dict)
