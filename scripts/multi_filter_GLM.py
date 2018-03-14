import sklearn
import theano
import neoUtils
import os
import sys
import numpy as np
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
    # Xc = X[cbool,:]
    # yc = y[cbool]
    yhat = np.zeros_like(y)
    return(X,y,cbool)


def neglogliklihood(z,y):
    z = tf.multiply(z,1000.)
    z = tf.reshape(z,[-1,1])
    cost = tf.matmul(tf.transpose(y),tf.log(z))+tf.reduce_sum(z)
    return(cost)


def build_GLM_model(X,y,cbool,nfilts=4,hist=False):
    epochs = 10
    batch_size = 64
    learning_rate = 0.5
    # make data a multiple of batchsize
    n_del = X.shape[0] % batch_size
    X = X[n_del:,:]
    y = y[n_del:]
    # Create Batch Indices


    # map X and y to tensor
    X = tf.reshape(X,X.shape)
    y = tf.reshape(y,y.shape)
    X = tf.cast(X,'float32')
    y = tf.cast(y,'float32')

    #### The model ###
    # Input Layer
    L1 = tf.layers.dense(X,nfilts,
                         activation=tf.nn.relu)
    # Starting without history. Will add history term later
    Ksum = tf.reduce_sum(L1,axis=1)
    conditional_intensity = tf.exp(Ksum)

    # define cost function as negative log liklihood of Poisson spiking
    cost = neglogliklihood(conditional_intensity,y)
    optimizer = tf.train.GradientDescentOptimizer().minimize(cost)
    init = tf.global_variables_initializer()

    batched_x = tf.split(X,batch_size)
    batched_y = tf.split(y,batch_size)

    with tf.Session() as sess:
        sess.run(init)
        # loop over entire dataset multiple times
        for epoch in range(epochs):
            # loop over sub_batches
            for ii in range(total_batch):
                sess.run([optimizer,cost],
                         feed_dict={x:batched_x[ii],y:batched_y[ii]})












