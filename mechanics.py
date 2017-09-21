import numpy as np
from scipy.io.matlab import loadmat,savemat
import quantities as pq
import seaborn as sns
import sklearn
import matplotlib.ticker as ticker
from sklearn.preprocessing import scale


def get_analog_contact(var, cc):
    ''' this gets the mean and min-max of a given analog signal in each contact interval'''
    mean_var = np.empty([cc.shape[0], var.shape[1]])
    minmax_var = np.empty([cc.shape[0], var.shape[1]])

    for ii, contact in enumerate(cc):
        var_slice = var[contact[0]:contact[1], :]
        mean_var[ii, :] = np.mean(var_slice, 0)
        minmax_idx = np.argmax(np.abs(var_slice), 0)
        minmax_var[ii, :] = var_slice[minmax_idx, np.arange(len(minmax_idx))]

    return mean_var,minmax_var


def iterate_filtvar(filtvars,cc):
    ''' this loops through every variable in a structure (generally filtvars)
    and returns a dict of mean and minmax for each variable in each contact'''
    mean_filtvars = {}
    minmax_filtvars = {}

    for attr in filtvars._fieldnames:
        mean_filtvars[attr], minmax_filtvars[attr] = get_analog_contact(getattr(filtvars,attr),cc)
    return mean_filtvars,minmax_filtvars


def create_heatmap(var1,var2,bins,C,r):
    '''Need to figure out axes'''

    cmap = sns.cubehelix_palette(as_cmap=True)
    C = C.astype('bool').ravel()
    var1 = var1[C]
    var2 = var2[C]
    if type(r)!=np.ndarray:
        r = r.as_array()
    r = r.ravel()[C]
    H_prior,x_edges,y_edges = np.histogram2d(var1, var2, bins=bins)
    H_post = np.histogram2d(var1, var2, bins=bins, weights=r)[0]


    fig = sns.heatmap(H_post/H_prior,
                      vmin=0,
                      cmap=cmap,
                      robust=True,
                      cbar=True,
                      square=True,
                      xticklabels=20,
                      yticklabels=20)

    plt.colorbar()

def get_derivs():
    D = np.sqrt(TH ** 2 + PHIE ** 2 + Rcp ** 2)


def categorize_deflections(mean_filtvars,cc):
    dur = np.diff(cc,axis=1)
    TH = mean_filtvars['TH']
    PHIE = mean_filtvars['PHIE']
    Rcp = mean_filtvars['Rcp']



    max_vel = np.diff()
    X = np.hstack((dur,TH,PHIE,Rcp,cc[:,0].reshape([-1,1])))
    X= scale(X)
    clf = SC(assign_labels='discretize')
    labels = clf.fit_predict(X)

    if plot_tgl:
        ax=Axes3D(plt.figure())
        ax.scatter(mean_filtvars['TH'], mean_filtvars['PHIE'], mean_filtvars['Rcp'], c=labels, cmap='hsv')

