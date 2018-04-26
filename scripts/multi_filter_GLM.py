import sklearn
import neoUtils
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from GLM import get_blk_smooth
import GLM
import glob

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
    return(cost[0][0])

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

def build_GLM_model(Xraw,yraw,savefile, nfilts=4,hist=False,learning_rate=1e-5,epochs=100,batch_size=256,family='p',min_delta=0.1,patience=8):
    tf.reset_default_graph()
    if batch_size is None:
        batch_size=Xraw.shape[0]
    if hist:
        B = GLM.make_bases(3, [0, 3], 1)
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
    # hidden_out = tf.nn.relu(hidden_out)

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


    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    # optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost)
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    sess = tf.Session()

    sess.run(init)
    # loop over entire dataset multiple times
    all_cost = [np.Inf]


    patience_cnt=0
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

            # Early Stopping
        # print('AVG:{}, Most Recent:{}'.format(avg_cost, all_cost[-1]))

        if epoch>0 and ((all_cost[-1]-avg_cost) > min_delta):
            patience_cnt=0
        else:
            patience_cnt+=1


        if patience_cnt>=patience:
            print('Early Stopping...')
            break
        all_cost.append(avg_cost)

        print('Epoch:{}\t, Cost={}'.format(epoch,avg_cost))
    print('Done!')
    # plt.plot(all_cost)
    # plt.show()
    print('saving to {}'.format(savefile))
    saver = tf.train.Saver()
    saver.save(sess,savefile)
    sess.close()
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

def simulate(X,y,p_model,cbool,n_sims=50):
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(p_model+'.meta')
    new_saver.restore(sess,p_model)
    K = tf.get_collection('K')[0]
    H = tf.get_collection('H')[0]
    b = tf.get_collection('b')[0]

    K = sess.run(K)
    H = sess.run(H)
    b = sess.run(b)

    stim_curr = np.dot(X,K)
    stim_curr = np.sum(stim_curr,axis=1)+b

    # Warning! The spike history basis is hard coded.
    B = GLM.make_bases(3,[0,3],1)
    H = GLM.map_bases(H,B)[0].ravel()
    g = np.zeros([X.shape[0]+len(H),n_sims]) # total current?
    ysim = np.zeros([X.shape[0],n_sims])# response vector (simulated spiketrains)
    hcurr = np.zeros([X.shape[0]+len(H),n_sims])# history current
    rsim = np.zeros_like(g)


    for runNum in range(n_sims):
        print('Simulation number {} of {}'.format(runNum+1,n_sims))
        g[:,runNum]=np.concatenate([stim_curr,np.zeros([len(H)])])
        for t in xrange(stim_curr.shape[0]):
            rsim[t,runNum] = np.exp(g[t,runNum])
            if not cbool[t]:
                continue
            if np.random.rand()<(1-np.exp(-rsim[t,runNum])):
                ysim[t,runNum]=1
                g[t+1:t+len(H)+1,runNum] += H
                hcurr[t+1:t+len(H)+1,runNum]+= H
    hcurr = hcurr[:X.shape[0],:]
    rsim = rsim[:X.shape[0],:]

    output = {}
    output['K'] = K
    output['H'] = H
    output['ysim'] = ysim
    output['rsim'] = rsim
    output['b'] = b
    output['Basis'] = B
    output['stim_curr'] = stim_curr
    output['hcurr'] = hcurr
    sess.close()
    return(output)

def main(fname,p_smooth,p_save,param_dict,mask=None,model_name='tensorflow'):
    """
    Run the multi-filter GLM on a given file
    :param fname:
    :param p_smooth:
    :param p_save:
    :return:  Saves a numpy file to p_save
    """

    blk = neoUtils.get_blk(fname)
    n_sims = param_dict.pop('n_sims')
    num_units = len(blk.channel_indexes[-1].units)
    for unit_num in range(num_units):
        X,y,cbool = get_X_y(fname,p_smooth,unit_num)
        root = neoUtils.get_root(blk,unit_num)
        model_fname = os.path.join(p_save,'{}_{}.ckpt'.format(root,model_name))
        X[np.invert(cbool),:] = 0
        y[np.invert(cbool),:] = 0
        # Drop some variables as indicated by mask
        if mask is not None:
            X = X[:,mask]
        # Train
        build_GLM_model(X,y,model_fname,**param_dict)
        #Simulate
        output = simulate(X,y,model_fname,cbool,n_sims)
        print('Saving...')
        np.savez(os.path.join(p_save,'{}_multi_filter.npz'.format(root)),
                 X=X,
                 y=y,
                 cbool=cbool,
                 model_out=output,
                 param_dict=param_dict)
        print('Saved')

def batch(p_load,p_save,p_smooth,param_dict):
    badfiles=[]
    for fname in glob.glob(os.path.join(p_load,'*.h5')):
        print('working on {}'.format(os.path.basename(fname)))
        try:
            main(fname,p_smooth,p_save)
        except:
            badfiles.append(fname)
            continue
    print(badfiles)
    return 0

def make_mask():
    mask = {}
    mask['full'] = np.ones(16,dtype='bool')
    mask['no_M'] = np.array([0,0,0,1,1,1,1,1]*2,dtype='bool')
    mask['no_F'] = np.array([1,1,1,0,0,0,1,1]*2,dtype='bool')
    mask['no_R'] = np.array([1,1,1,1,1,1,0,0]*2,dtype='bool')
    mask['no_D'] = np.concatenate([np.ones(8,dtype='bool'),np.zeros(8,dtype='bool')])

    mask['just_M'] = np.array([1,1,1,0,0,0,0,0]*2,dtype='bool')
    mask['just_F'] = np.array([0,0,0,1,1,1,0,0]*2,dtype='bool')
    mask['just_R'] = np.array([0,0,0,0,0,0,1,1]*2,dtype='bool')
    mask['just_D'] = np.concatenate([np.zeros(8,dtype='bool'),np.ones(8,dtype='bool')])
    return(mask)

if __name__=='__main__':
    # Main will run a number of possibilities.
    # TODO: make the main accept either: test, batch (run all files with same params), or drop (one file with multiple variable type--this is to be used on quest)
    # system arguments:
        # fname (or p_load), p_save, p_smooth

    param_dict={'family':'p',
                'hist':True,
                'nfilts':3,
                'learning_rate':3e-4,
                'batch_size':4096,
                'epochs':5000,
                'min_delta':0.01,
                'patience':8,
                'n_sims':100}

    if sys.argv[1] == 'test':
        main('/media/nbush/GanglionData/VG3D/_rerun_with_pad/_deflection_trials/_NEO/rat2017_08_FEB15_VG_D1_NEO.h5',
             '/media/nbush/GanglionData/VG3D/_rerun_with_pad/_deflection_trials/_NEO/smooth',
             '/home/nbush/Desktop/models',
             param_dict)

    elif sys.argv[1] == 'batch':
        batch(sys.argv[2],
              sys.argv[3],
              sys.argv[4],
              param_dict)

    elif sys.argv[1]== 'drop':
        masks = make_mask()
        for model_name,mask in masks.iteritems():
           main(sys.argv[2],
                sys.argv[3],
                sys.argv[4],
                mask,
                'tensorflow_'+model_name)
    else:
        print('Invalid first argument passed')
