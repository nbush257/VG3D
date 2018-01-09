import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import neoUtils
import spikeAnalysis
import worldGeometry

sns.set()
def plot3():
    f = plt.figure()
    return(f.add_subplot(111,projection='3d'))
def polar():
    f = plt.figure()
    return(f.add_subplot(111,projection='polar'))

def plot_spike_trains_by_direction(blk,unit_num):
    unit = blk.channel_indexes[-1].units[unit_num]
    _,_,trains = spikeAnalysis.get_contact_sliced_trains(blk,unit)
    b = spikeAnalysis.get_binary_trains(trains)

    idx,med_angle = worldGeometry.get_contact_direction(blk,plot_tgl=False)
    for dir in np.arange(np.arange():
        plt.plot(b[:,idx=])