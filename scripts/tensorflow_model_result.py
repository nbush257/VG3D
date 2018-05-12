import numpy as np
import glob
import scipy
import spikeAnalysis
import pandas as pd
import neoUtils
import neo
import elephant
import quantities as pq
import os


def load_dat(fname):
    dat = np.load(fname)

    mdl = dat['model_out'].item()
    y = dat['y']
    cbool = dat['cbool']

    return(y,cbool,mdl)
def yhat2trains(mdl,cbool):
    """
    Map the predicted spikes to a list of spike trains and extract a contact epoch
    :param mdl:
    :param cbool:
    :return:
    """
    ysim = mdl['ysim']
    trains = []
    for pred in ysim.T:
        train = neo.SpikeTrain(times=np.where(pred)*pq.ms,
                               t_stop=len(pred)*pq.ms)
        trains.append(train[0])

    starts,stops = neoUtils.Cbool_to_cc(cbool)
    dur = stops-starts
    epoch = neo.Epoch(starts*pq.ms,dur*pq.ms)
    return(trains,epoch)

def correlate_simulated_rate(mdl,y,cbool):
    spt = spikeAnalysis.binary_to_neo_train(y)
    kernels = []
    for ii in np.power(2,np.arange(1,10)):
        kernels.append(elephant.kernels.GaussianKernel(ii*pq.ms))

    r = [elephant.statistics.instantaneous_rate(spt,sampling_period=pq.ms,kernel=x) for x in kernels]

    pred_rate = np.mean(mdl['ysim'],axis=1)*1000
    R = []
    for rate in r:
        R.append(scipy.corrcoef(rate.magnitude[cbool].ravel(),pred_rate[cbool])[0,1])

    return(R)

def spike_train_similarities(trains,epoch,y,algo='VP'):
    """
    Calculates the spike train distances on a per contact basis
    :param trains: list of neo trains from simulations
    :param epoch: neo epoch of contact periods
    :param y: observed spike train as a boolean array
    :param algo: 'VP' to use Victo Purpura, otherwise use vanrossum
    :return: D
    """
    # Convert obseved spiketrain from bool to neo
    spt = spikeAnalysis.binary_to_neo_train(y)
    # init loop vars
    D = []
    count = 0
    #Loop over all contacts
    for start,dur in zip(epoch,epoch.durations):
        if count%100==0:
            print('{} of {}'.format(count,len(epoch)))
        count+=1
        D_contact = []

        # Calculate the distance between a simulated contact and the observed train contact
        # for all trains
        for train in trains:
            # make a 2 length list with the current predicted train and the observed train
            t_slice = []
            t_slice.append(train.time_slice(start,start+dur))
            t_slice.append(spt.time_slice(start,start+dur))
            # Use either VP or Vanrossum
            if algo=='VP':
                D_contact.append(elephant.spike_train_dissimilarity.victor_purpura_dist(t_slice)[0,1])
            else:
                D_contact.append(elephant.spike_train_dissimilarity.van_rossum_dist(t_slice)[0,1])
            # output to master list
        D.append(D_contact)
    return(np.array(D))


def main(fname):
    y,cbool,mdl = load_dat(fname)
    trains,epoch = yhat2trains(mdl,cbool)
    D = spike_train_similarities(trains,epoch,y)
    return(D)

def batch():
    p_load = os.path.join(os.environ['BOX_PATH'],r'__VG3D\_deflection_trials\_NEO\tensorflow')
    DF = pd.DataFrame()
    for f in glob.glob(os.path.join(p_load,'*.npz')):
        print('Working on {}'.format(os.path.basename(f)))
        try:
            df = pd.DataFrame()
            D = main(f)
            Dm = np.mean(D,axis=1)
            df['VP_distance'] = Dm
            df['id'] = os.path.basename(f)[:10]
        except:
            print('problem')
            continue
        DF = DF.append(df)

    DF.to_csv(os.path.join(p_load,'spiketrain_distances_tf_sims.csv'))



if __name__=='__main__':
    batch()
